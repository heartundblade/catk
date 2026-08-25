import torch
import torch.nn as nn
# from torch_cluster import knn

from src.dp.model.module.timm import Mlp, DropPath
from src.dp.model.diffusion_utils.sampling import dpm_sampler
from src.dp.model.diffusion_utils.sde import SDE, VPSDE_linear, VPSDE_LogSNR
from src.dp.utils.normalizer import ObservationNormalizer, StateNormalizer, ActionNormalizer
from src.dp.model.module.dit import TimestepEmbedder, DiTBlock, FinalLayer, RefineBlock, ChunkAggregator
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
            output_dim= self._num_actions * 2, # [acc, yaw_rate] for each action
            hidden_dim=config.hidden_dim, 
            heads=config.num_heads, 
            dropout=dpr,
            model_type=config.diffusion_model_type,
            agent_num=config.agent_num,
            action_len=config.action_len,
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
            sampled_actions = inputs['sampled_actions'].reshape(B, P, -1)  # [B, P, num_actions*2]
            diffusion_time = inputs['diffusion_time']

            return {
                    "score": self.dit(
                        sampled_actions, 
                        diffusion_time,
                        context_encoding,
                        current_states=current_states,
                        relation_encodings=rel_enc_pn,
                        encoding_mask=encoding_mask,
                        action_len=self._action_len,
                    ).reshape(B, P, self._num_actions, 2)  # [B, P, num_actions, 2]
                }
        else:
            # [B, 1 + predicted_neighbor_num, num_actions, 2]
            # xT = (torch.randn(B, P, self._num_actions, 2).to(current_states.device) * 0.5).reshape(B, P, -1)
            xT = (torch.randn(B, P, self._num_actions, 2).to(current_states.device)).reshape(B, P, -1)

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
            x0 = x0.reshape(B, P, self._num_actions, 2)

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
                 action_dim = 2,  # [acc, yaw_rate] for each action
                 future_len = 80,
                 model_type="x_start",
                 agent_num=64,
                 num_chunks=4,  # Number of chunks C
                 action_len=2,
        ):
        super().__init__()
        
        assert model_type in ["score", "x_start", "noise"], f"Unknown model type: {model_type}"
        self._model_type = model_type
        self._agent_num = agent_num
        self._action_dim = action_dim
        self._num_chunks = num_chunks
        self._future_len = future_len
        self._chunk_size = future_len // (num_chunks*action_len)
        
        # Action embedding: embed each action
        self.step_embed = Mlp(
            in_features=action_dim, 
            hidden_features=hidden_dim, 
            out_features=hidden_dim, 
            act_layer=nn.GELU, 
            drop=0.
        )
        
        # Chunk aggregation: VectorMapBackbone-style (point_mlp + attn_pool)
        self.chunk_aggregator = ChunkAggregator(hidden_dim, num_heads=heads, dropout=dropout)
        
        # Positional encoding for steps within chunk
        self.step_pos_embed = nn.Parameter(torch.zeros(1, self._chunk_size, hidden_dim))
        # Positional encoding for chunks
        self.chunk_pos_embed = nn.Parameter(torch.zeros(1, num_chunks, hidden_dim))
        
        self.t_embedder = TimestepEmbedder(hidden_dim)
        # Timestep modulation for action features (adaLN)
        self.t_action_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        )

        # Stage 1: Chunk-level bidirectional self-attention
        self.chunk_blocks = nn.ModuleList([
            DiTBlock(hidden_dim, heads, dropout, mlp_ratio) 
            for _ in range(depth)
        ])
        
        # Projection for chunk context before broadcasting to action level
        self.chunk_context_proj = nn.Linear(hidden_dim, hidden_dim)

        # Stage 2: Step-level intra-chunk attention (refiner)
        self.refiner_blocks = nn.ModuleList([
            RefineBlock(hidden_dim, heads, dropout)
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
            x: (B, P, output_dim)   -> Noisy actions [B, P, num_actions*2]
            t: (B,)                 -> Diffusion timestep
            cross_c: (B, N, D)      -> Cross-Attention context
            current_states: (B, P, 6) -> Current states (for potential use)
            action_len: int         -> Action length (not used in forward)
        
        Architecture:
            1. Input: [B, P, num_actions*2] noisy actions
            2. Split into C chunks: [B, P, C, chunk_size, 2]
            3. Action embedding: each action -> hidden_dim
            4. Chunk aggregation: actions within chunk -> chunk token
            5. Stage 1: Chunk-level bidirectional self-attention
            6. Stage 2: Broadcast chunk features, action-level intra-chunk attention
            7. Output: per-action predictions [B, P, num_actions, 2]
        """
        B, P, _ = x.shape
        valid_mask = ~encoding_mask[:, :P]

        # Reshape to actions: [B, P, num_actions, 2]
        actions = x.reshape(B, P, self._num_chunks * self._chunk_size, 2)
        action_chunks = actions.reshape(B, P, self._num_chunks, self._chunk_size, 2)
        action_feat = self.step_embed(action_chunks)  # [B, P, C, chunk_size, hidden_dim]

        y = self.t_embedder(t)  # [B, D]

        # adaLN modulate action features with timestep before GRU aggregation
        shift, scale = self.t_action_modulation(y).chunk(2, dim=-1)
        shift = shift.view(B, 1, 1, 1, -1)
        scale = scale.view(B, 1, 1, 1, -1)
        action_feat = action_feat * (1 + scale) + shift
        action_feat = action_feat + self.step_pos_embed.view(1, 1, 1, self._chunk_size, -1)
        
        action_feat_flat = action_feat.reshape(B * P * self._num_chunks, self._chunk_size, -1)
        chunk_features = self.chunk_aggregator(action_feat_flat)  # [B*P*C, D]
        chunk_features = chunk_features.reshape(B, P, self._num_chunks, -1)  # [B, P, C, D]
        
        chunk_features = chunk_features + self.chunk_pos_embed
        
        # Reshape for processing: [B, P, C, D] -> [B*P, C, D]
        chunk_feat = chunk_features.reshape(B * P, self._num_chunks, -1)
        
        # Prepare cross-attention context
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
        
        # Expand y from [B, ...] to [B*P, ...]
        y_s1 = y.unsqueeze(1).expand(B, P, -1).reshape(B * P, -1)
        
        # Flatten action features for refiner: [B, P, C, S, D] -> [B*P*C, S, D]
        action_feat = action_feat.reshape(
            B * P * self._num_chunks, self._chunk_size, -1
        )

        # Prepare agent context for RefineBlock adaLN: [B, P, D] -> [B*P*C, D]
        agent_ctx = cross_agent[:, :P, :].unsqueeze(2).expand(
            B, P, self._num_chunks, -1
        ).reshape(B * P * self._num_chunks, -1)  # [B*P*C, D]

        # Interleaved: chunk block -> broadcast chunk context -> refiner block
        for i in range(len(self.chunk_blocks)):
            if i > 0:
                action_feat_flat = action_feat.reshape(
                    B * P * self._num_chunks, self._chunk_size, -1
                )
                chunk_feat_from_actions = self.chunk_aggregator(action_feat_flat)
                chunk_feat = chunk_feat + chunk_feat_from_actions.reshape(B * P, self._num_chunks, -1)

            # Stage 1: Chunk-level DiTBlock
            chunk_feat = self.chunk_blocks[i](
                chunk_feat,
                cross_agent, cross_map, y_s1, attn_mask,
                rel_enc_agent, rel_enc_map,
                encoding_mask_agent, encoding_mask_map
            )

            # Project chunk features before broadcasting to action level
            chunk_feat_proj = self.chunk_context_proj(chunk_feat)  # [B*P, C, D]
            chunk_context = chunk_feat_proj.view(B, P, self._num_chunks, -1).unsqueeze(3).expand(
                B, P, self._num_chunks, self._chunk_size, -1
            ).reshape(B * P * self._num_chunks, self._chunk_size, -1)
            
            # Add chunk context to action features (progressive refinement)
            action_feat = action_feat + chunk_context
            
            # Stage 2: Action-level RefineBlock with agent adaLN modulation
            action_feat = self.refiner_blocks[i](action_feat, agent_ctx)
            
        action_outputs = self.final_layer(action_feat)
        output = action_outputs.reshape(B, P, -1)
        
        if self._model_type == "score":
            return output / (self.marginal_prob_std(t)[:, None, None] + 1e-6)
        elif self._model_type == "x_start":
            return output
        elif self._model_type == "noise":
            return output
        else:
            raise ValueError(f"Unknown model type: {self._model_type}")