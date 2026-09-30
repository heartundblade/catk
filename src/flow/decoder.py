import torch
import torch.nn as nn
# from torch_cluster import knn

from src.dp.model.module.timm import Mlp, DropPath
from src.dp.model.diffusion_utils.sampling import dpm_sampler
from src.dp.model.diffusion_utils.sde import SDE, VPSDE_linear, VPSDE_LogSNR
from src.dp.utils.normalizer import ObservationNormalizer, StateNormalizer, ActionNormalizer
from src.dp.model.module.dit import TimestepEmbedder, DiTBlock, FinalLayer, RefineBlock, ChunkEmbbeder
from src.dp.model.module.mixer import MixerBlock
from src.dp.utils.train_utils import roll_out, batch_transform_trajs_to_local_frame
from src.dp.model.module.rel_emb import RelationEncoder
from src.dp.utils.train_utils import wrap_angle

class Decoder(nn.Module):
    def __init__(self, config, rel_encoder=None):
        super().__init__()

        dpr = config.decoder_drop_path_rate
        self._predicted_neighbor_num = config.predicted_neighbor_num
        self._future_len = config.future_len
        self._action_len = config.get('action_len', 1)
        self._num_actions = self._future_len
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
            output_dim= self._future_len * 3, # [a_parallel, a_lateral, a_psi] for each step
            hidden_dim=config.hidden_dim, 
            heads=config.num_heads, 
            dropout=dpr,
            future_len=self._future_len,
            model_type=config.diffusion_model_type,
            agent_num=config.agent_num,
            num_chunks=config.num_chunks,
            action_len=config.action_len,
        )
        
        self._guidance_fn = config.guidance_fn
        self._diffusion_steps = getattr(config, 'diffusion_steps', 10)
        
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
        encoding_mask = encoder_outputs['encoding_mask']

        P = 1 + self._predicted_neighbor_num
        rel_enc_pn = None

        if self.training:
            sampled_actions = inputs['sampled_actions'].reshape(B, P, -1)  # [B, P, future_len*3]
            diffusion_time = inputs['diffusion_time']

            return {
                    "score": self.dit(
                        sampled_actions, 
                        diffusion_time,
                        context_encoding,
                        current_states=current_states,
                        relation_encodings=rel_enc_pn,
                        encoding_mask=encoding_mask,
                        action_len=1,
                    ).reshape(B, P, self._future_len, 3)  # [B, P, future_len, 3]
                }
        else:
            # [B, 1 + predicted_neighbor_num, future_len, 3]
            # xT = (torch.randn(B, P, self._future_len, 3).to(current_states.device) * 0.5).reshape(B, P, -1)
            xT = (torch.randn(B, P, self._future_len, 3).to(current_states.device)).reshape(B, P, -1)

            x0 = dpm_sampler(
                        self.dit,
                        xT,
                        other_model_params={
                            "cross_c": context_encoding, 
                            "current_states": current_states,
                            "relation_encodings": rel_enc_pn,
                            "encoding_mask": encoding_mask,
                            "action_len": self._action_len,
                        },
                        diffusion_steps=self._diffusion_steps,
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
                                    "action_len": self._action_len,
                                },
                                "inputs": inputs,
                                "observation_normalizer": self._observation_normalizer,
                                "state_normalizer": self._state_normalizer
                            },
                            "guidance_scale": 0.5,
                            "guidance_type": "classifier" if self._guidance_fn is not None else "uncond"
                        },
                )
            
            # No need for state normalizer inverse for actions
            x0 = x0.reshape(B, P, self._future_len, 3)

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
                 action_dim = 3,  # [a_parallel, a_lateral, a_psi] for each step
                 future_len = 80,
                 model_type="x_start",
                 agent_num=64,
                 num_chunks=4,  # Number of chunks C
                 action_len=2,
        ):
        super().__init__()
        
        assert model_type in ["score", "x_start", "noise", "velocity"], f"Unknown model type: {model_type}"
        self._model_type = model_type
        self._agent_num = agent_num
        self._action_dim = action_dim
        self._num_chunks = num_chunks
        self._future_len = future_len
        self._chunk_size = future_len // num_chunks
        
        self.step_emb = nn.Sequential(
            nn.Linear(action_dim, hidden_dim, bias=True),
            nn.LayerNorm(hidden_dim),
        )
        
        self.chunk_emb = ChunkEmbbeder(hidden_dim, num_heads=heads, dropout=dropout, chunk_size=self._chunk_size)
        
        self.step_pos_embed = nn.Parameter(torch.zeros(1, self._chunk_size, hidden_dim))
        self.chunk_pos_embed = nn.Parameter(torch.zeros(1, num_chunks, hidden_dim))
        
        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.t_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 3 * hidden_dim, bias=True)
        )

        # Stage 1: Chunk-level bidirectional self-attention
        self.chunk_blocks = nn.ModuleList([
            DiTBlock(hidden_dim, heads, dropout, mlp_ratio) 
            for _ in range(depth)
        ])
        
        self.chunk_context_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        # Stage 2: Step-level intra-chunk attention (refiner)
        self.refiner_blocks = nn.ModuleList([
            RefineBlock(hidden_dim, heads, dropout, self._chunk_size)
            for _ in range(depth)
        ])
        
        # Output head: per-chunk predictions, each chunk has its own head
        self.final_layer = FinalLayer(hidden_dim, action_dim, num_chunks=num_chunks)
        
        self._sde = sde
        self.marginal_prob_std = self._sde.marginal_prob_std
        self._action_normalizer = action_normalizer
        
        # Initialize positional embeddings
        nn.init.normal_(self.step_pos_embed, std=0.02)
        nn.init.normal_(self.chunk_pos_embed, std=0.02)

        self._init_weights()

    def _init_weights(self):
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

        self.apply(_basic_init)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        nn.init.constant_(self.t_modulation[-1].weight, 0)
        nn.init.constant_(self.t_modulation[-1].bias, 0)
        for block in self.chunk_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        for block in self.refiner_blocks:
            nn.init.constant_(block.agent_modulation[-1].weight, 0)
            nn.init.constant_(block.agent_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.head[-1].weight, 0)
        nn.init.constant_(self.final_layer.head[-1].bias, 0)

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
                action_len=2,
        ):
        """
        Forward pass of DiT with hierarchical two-stage attention.
        
        Args:
            x: (B, P, output_dim)   -> Noisy actions [B, P, future_len*3]
            t: (B,)                 -> Diffusion timestep
            cross_c: (B, N, D)      -> Cross-Attention context
            current_states: (B, P, 6) -> Current states (for potential use)
            action_len: int         -> Action length (not used in forward)
        
        Architecture:
            1. Input: [B, P, future_len*3] noisy actions
            2. Split into C chunks: [B, P, C, chunk_size, 3]
            3. Action embedding: each action -> hidden_dim
            4. Chunk aggregation: actions within chunk -> chunk token
            5. Stage 1: Chunk-level bidirectional self-attention
            6. Stage 2: Broadcast chunk features, action-level intra-chunk attention
            7. Output: per-action predictions [B, P, future_len, 3]
        """
        B, P, _ = x.shape
        # valid_mask = ~encoding_mask[:, :P]

        # Reshape to actions: [B, P, future_len, action_dim]
        actions = x.reshape(B, P, self._num_chunks * self._chunk_size, self._action_dim)
        action_chunks = actions.reshape(B, P, self._num_chunks, self._chunk_size, self._action_dim)
        action_emb = self.step_emb(action_chunks)  # [B, P, C, chunk_size, hidden_dim]

        y = self.t_embedder(t)  # [B, D]

        # adaLN modulate action features with timestep before GRU aggregation
        shift, scale, gate = self.t_modulation(y).chunk(3, dim=-1)
        shift = shift.view(B, 1, 1, 1, -1)
        scale = scale.view(B, 1, 1, 1, -1)
        gate = gate.view(B, 1, 1, 1, -1)
        action_emb = action_emb * (1 + scale * gate.sigmoid()) + shift
        action_emb = action_emb + self.step_pos_embed.view(1, 1, 1, self._chunk_size, -1)
        
        action_emb_flat = action_emb.reshape(B * P * self._num_chunks, self._chunk_size, -1)
        chunk_emb = self.chunk_emb(action_emb_flat)  # [B*P*C, D]
        chunk_emb = chunk_emb.reshape(B, P, self._num_chunks, -1)  # [B, P, C, D]
        
        chunk_emb = chunk_emb + self.chunk_pos_embed.view(1, 1, self._num_chunks, -1)
        chunk_feat = chunk_emb.reshape(B * P, self._num_chunks, -1)
        
        cross_agent = cross_c[:, :self._agent_num, :]  # [B, N_agent, D]
        cross_map = cross_c[:, self._agent_num:, :]    # [B, N_map, D]
        
        encoding_mask_agent = encoding_mask[:, :self._agent_num] if encoding_mask is not None else None
        encoding_mask_map = encoding_mask[:, self._agent_num:] if encoding_mask is not None else None
        
        rel_enc_agent = None
        rel_enc_map = None
        if relation_encodings is not None:
            rel_enc_agent = relation_encodings[:, :, :self._agent_num, :]
            rel_enc_map = relation_encodings[:, :, self._agent_num:, :]
        
        attn_mask = encoding_mask[:, :P]
        
        y = y.unsqueeze(1).expand(B, P, -1).reshape(B * P, -1)
        
        action_emb = action_emb.reshape(
            B * P * self._num_chunks, self._chunk_size, -1
        )
        agent_ctx = cross_agent[:, :P, :].unsqueeze(2).expand(
            B, P, self._num_chunks, -1
        ).reshape(B * P * self._num_chunks, -1)  # [B*P*C, D]

        # Interleaved: chunk block -> broadcast chunk context -> refiner block
        for i in range(len(self.chunk_blocks)):
            if i > 0:
                action_emb_flat = action_emb.reshape(
                    B * P * self._num_chunks, self._chunk_size, -1
                )
                chunk_emb = self.chunk_emb(action_emb_flat)
                chunk_feat = chunk_feat + chunk_emb.reshape(B * P, self._num_chunks, -1)

            # Stage 1: Chunk-level DiTBlock
            chunk_feat = self.chunk_blocks[i](
                chunk_feat,
                cross_agent, cross_map, y, attn_mask,
                rel_enc_agent, rel_enc_map,
                encoding_mask_agent, encoding_mask_map
            )

            chunk_context = self.chunk_context_proj(chunk_feat)  # [B*P, C, D]
            chunk_context = chunk_context.unsqueeze(2).expand(
                -1, -1, self._chunk_size, -1
            ).reshape(B * P * self._num_chunks, self._chunk_size, -1)
            action_emb = action_emb + chunk_context
            
            # Stage 2: Action-level RefineBlock with agent adaLN modulation
            action_emb = self.refiner_blocks[i](action_emb, agent_ctx)
            
        action_out = self.final_layer(action_emb)
        output = action_out.reshape(B, P, -1)
        
        if self._model_type == "score":
            return output / (self.marginal_prob_std(t)[:, None, None] + 1e-6)
        elif self._model_type == "x_start":
            return output
        elif self._model_type in ("noise", "velocity"):
            return output
        else:
            raise ValueError(f"Unknown model type: {self._model_type}")
