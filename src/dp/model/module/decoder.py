import torch
import torch.nn as nn
# from torch_cluster import knn

from src.dp.model.module.timm import Mlp, DropPath
from src.dp.model.diffusion_utils.sampling import dpm_sampler
from src.dp.model.diffusion_utils.sde import SDE, VPSDE_linear, VPSDE_LogSNR
from src.dp.utils.normalizer import ObservationNormalizer, StateNormalizer, ActionNormalizer
from src.dp.model.module.dit import TimestepEmbedder, DiTBlock, FinalLayer
# from src.dp.utils.train_utils import roll_out, batch_transform_trajs_to_local_frame
from src.dp.model.module.rel_emb import RelationEncoder

class Decoder(nn.Module):
    def __init__(self, config, rel_encoder=None):
        super().__init__()

        dpr = config.decoder_drop_path_rate
        self._predicted_neighbor_num = config.predicted_neighbor_num
        self._future_len = config.future_len
        self._action_len = config.action_len
        self._num_actions = self._future_len // self._action_len
        self._sde = VPSDE_linear()

        self._state_normalizer: StateNormalizer = StateNormalizer.from_json(config)
        self._observation_normalizer: ObservationNormalizer = ObservationNormalizer.from_json(config.normalization_file_path)
        self._action_normalizer = ActionNormalizer.from_json(config)
        
        self.rel_encoder = rel_encoder if rel_encoder is not None else RelationEncoder(
            hidden_dim=config.hidden_dim,
            num_freq_bands=64
        )

        self.dit = DiT(
            sde=self._sde, 
            action_normalizer=self._action_normalizer,
            depth=config.decoder_depth, 
            output_dim= self._future_len * 4, # [x, y, cos, sin] for each future frame
            hidden_dim=config.hidden_dim, 
            heads=config.num_heads, 
            dropout=dpr,
            model_type=config.diffusion_model_type,
            agent_num=config.agent_num,
        )
        
        self._guidance_fn = config.guidance_fn
        
    @property
    def sde(self):
        return self._sde
    
    def forward(self, encoder_outputs, inputs):
        """
        Diffusion decoder process.

        Args:
            encoder_outputs: Dict
                {
                    ...
                    "encoding": agents, static objects and lanes context encoding
                    ...
                }
            inputs: Dict
                {
                    ...
                    "ego_current_state": current ego states,            
                    "neighbor_agent_past": past and current neighbor states,  

                    [training-only] "sampled_trajectories": sampled current-future ego & neighbor states,        [B, P, 1 + V_future, 4]
                    [training-only] "diffusion_time": timestep of diffusion process $t \in [0, 1]$,              [B]
                    ...
                }

        Returns:
            decoder_outputs: Dict
                {
                    ...
                    [training-only] "score": Predicted future states, [B, P, 1 + V_future, 4]
                    [inference-only] "prediction": Predicted future states, [B, P, V_future, 4]
                    ...
                }

        """
        current_states = inputs['agents_history'][:, :self._predicted_neighbor_num+1, -1, :6]

        neighbor_current_mask = encoder_outputs['encoding_mask'][:, 1:self._predicted_neighbor_num+1]
        inputs["neighbor_current_mask"] = neighbor_current_mask

        B, P, _ = current_states.shape
        assert P == (1 + self._predicted_neighbor_num)

        # Extract context encoding
        context_encoding = encoder_outputs['encoding']
        relations = encoder_outputs['relations']
        encoding_mask = encoder_outputs['encoding_mask']

        P = 1 + self._predicted_neighbor_num
        rel_enc_pn = self.rel_encoder(relations[:, :P, :, :])

        if self.training:
            sampled_trajectories = inputs['sampled_trajectories'].reshape(B, P, -1)  # [B, P, future_len*4]
            diffusion_time = inputs['diffusion_time']

            return {
                    "score": self.dit(
                        sampled_trajectories, 
                        diffusion_time,
                        context_encoding,
                        current_states=current_states,
                        relation_encodings=rel_enc_pn,
                        encoding_mask=encoding_mask,
                    ).reshape(B, P, self._future_len, 4)  # [B, P, future_len, 4]
                }
        else:
            # [B, 1 + predicted_neighbor_num, future_len, 4]
            # xT = (torch.randn(B, P, self._future_len, 4).to(current_states.device) * 0.5).reshape(B, P, -1)
            xT = (torch.randn(B, P, self._future_len, 4).to(current_states.device)).reshape(B, P, -1)

            x0 = dpm_sampler(
                        self.dit,
                        xT,
                        other_model_params={
                            "cross_c": context_encoding, 
                            "current_states": current_states,
                            "relation_encodings": rel_enc_pn,
                            "encoding_mask": encoding_mask,
                        },
                        dpm_solver_params={},
                        model_wrapper_params={
                            "classifier_fn": self._guidance_fn,
                            "classifier_kwargs": {
                                "model": self.dit,
                                "model_condition": {
                                    "cross_c": context_encoding, 
                                    "current_states": current_states,
                                    "relation_encodings": rel_enc_pn,
                                    "encoding_mask": encoding_mask,
                                },
                                "inputs": inputs,
                                "observation_normalizer": self._observation_normalizer,
                                "state_normalizer": self._state_normalizer
                            },
                            "guidance_scale": 0.5,
                            "guidance_type": "classifier" if self._guidance_fn is not None else "uncond"
                        },
                )
            
            # No need for state normalizer inverse for trajectories
            x0 = x0.reshape(B, P, self._future_len, 4)
            x0 = self._state_normalizer.inverse(x0)

            return {
                    "prediction": x0
                }

class DiT(nn.Module):
    def __init__(self, 
                 sde: SDE, 
                 action_normalizer: ActionNormalizer,
                 depth, 
                 output_dim, 
                 hidden_dim=192, 
                 heads=6, 
                 dropout=0.1, 
                 mlp_ratio=4.0, 
                 traj_dim = 4,  # x, y, cos, sin 
                 future_len = 80,
                 model_type="x_start",
                 agent_num=64,
        ):
        super().__init__()
        
        assert model_type in ["score", "x_start"], f"Unknown model type: {model_type}"
        self._model_type = model_type
        self._agent_num = agent_num
        self._traj_dim = traj_dim
        self.preproj = Mlp(in_features=output_dim, hidden_features=512, out_features=hidden_dim, act_layer=nn.GELU, drop=0.)
        
        self.t_embedder = TimestepEmbedder(hidden_dim)

        self.agent_fusion_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.feat_tf_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=1, 
        )

        self.blocks = nn.ModuleList([DiTBlock(hidden_dim, heads, dropout, mlp_ratio) for i in range(depth)])
        self.final_layer = FinalLayer(hidden_dim, output_dim)
        
        self._sde = sde
        self.marginal_prob_std = self._sde.marginal_prob_std
        self._action_normalizer = action_normalizer

    @property
    def model_type(self):
        return self._model_type

    def forward(self, 
                x, 
                t, 
                cross_c, 
                current_states,
                relation_encodings=None,
                encoding_mask=None,
        ):
        """
        Forward pass of DiT.
        x: (B, P, output_dim)   -> Embedded out of DiT (actions)
        t: (B,)
        cross_c: (B, N, D)      -> Cross-Attention context
        current_states: (B, P, 6) -> Current states for rollout
        action_len: int -> Action length for rollout
        """
        B, P, _ = x.shape
        trajectories = x.reshape(B, P, -1, self._traj_dim)  # [B, P, future_len, traj_dim]

        valid_mask = ~encoding_mask[:, :P]

        traj = trajectories.view(B * P, -1, self._traj_dim)

        valid_indices = valid_mask.view(-1)  # [B*P]
        traj = traj[valid_indices]

        traj_flat = traj.reshape(traj.shape[0], -1)
        x_traj = self.preproj(traj_flat)

        x_traj_result = torch.zeros((B * P, x_traj.shape[-1]), device=x_traj.device, dtype=x_traj.dtype)
        x_traj_result[valid_indices] = x_traj
        x = x_traj_result.view(B, P, -1)

        y = self.t_embedder(t)

        attn_mask = encoding_mask[:, :P]
        
        cross_agent = cross_c[:, :self._agent_num, :]
        cross_map = cross_c[:, self._agent_num:, :]

        # Fuse each agent's own history with its noised trajectory (ref: denoising_decoder L413-415)
        agents_history = cross_agent[:, :P, :]
        y_expanded = y.unsqueeze(1).expand(-1, P, -1)
        x = self.agent_fusion_mlp(torch.cat((x, y_expanded, agents_history), dim=-1))
        x = self.feat_tf_encoder(x)

        encoding_mask_agent = encoding_mask[:, :self._agent_num] if encoding_mask is not None else None
        encoding_mask_map = encoding_mask[:, self._agent_num:] if encoding_mask is not None else None

        rel_enc_agent = None
        rel_enc_map = None
        if relation_encodings is not None:
            rel_enc_agent = relation_encodings[:, :, :self._agent_num, :]
            rel_enc_map = relation_encodings[:, :, self._agent_num:, :]

        for block in self.blocks:
            x = block(x, cross_agent, cross_map, y, attn_mask,
                      rel_enc_agent, rel_enc_map,
                      encoding_mask_agent, encoding_mask_map)
        
        x = self.final_layer(x, y)
        
        if self._model_type == "score":
            return x / (self.marginal_prob_std(t)[:, None, None] + 1e-6)
        elif self._model_type == "x_start":
            return x
        else:
            raise ValueError(f"Unknown model type: {self._model_type}")