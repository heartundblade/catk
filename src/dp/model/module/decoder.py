import math
import torch
import torch.nn as nn

from src.dp.model.module.timm import Mlp, DropPath
from src.dp.model.diffusion_utils.sampling import dpm_sampler
from src.dp.model.diffusion_utils.sde import SDE, VPSDE_linear, VPSDE_LogSNR
from src.dp.utils.normalizer import ObservationNormalizer, StateNormalizer, ActionNormalizer
from src.dp.model.module.mixer import MixerBlock
from src.dp.model.module.dit import TimestepEmbedder, DiTBlock, FinalLayer
from src.dp.utils.train_utils import roll_out

class Decoder(nn.Module):
    def __init__(self, config):
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
        
        # Output dimension for action prediction: num_actions * 2 (accel, yaw_rate)
        self.dit = DiT(
            sde=self._sde, 
            action_normalizer=self._action_normalizer,
            # route_encoder = RouteEncoder(config.route_num, config.lane_len, drop_path_rate=config.encoder_drop_path_rate, hidden_dim=config.hidden_dim),
            depth=config.decoder_depth, 
            output_dim= self._num_actions * 2, # [acc, yaw_rate] for each action
            hidden_dim=config.hidden_dim, 
            heads=config.num_heads, 
            dropout=dpr,
            model_type=config.diffusion_model_type
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
        # Extract ego & neighbor current states
        # ego_current = inputs['ego_current_state'][:, None, :4]
        # neighbors_current = inputs["neighbor_agents_past"][:, :self._predicted_neighbor_num, -1, :4]
        neighbors_current = inputs["agents_history"][:, 1:self._predicted_neighbor_num+1, -1, :4]
        neighbor_current_mask = torch.sum(torch.ne(neighbors_current, 0), dim=-1) == 0
        inputs["neighbor_current_mask"] = neighbor_current_mask

        # current_states = torch.cat([ego_current, neighbors_current], dim=1) # [B, P, 4]

        current_states = inputs['agents_history'][:, :self._predicted_neighbor_num+1, -1, :6]

        B, P, _ = current_states.shape
        assert P == (1 + self._predicted_neighbor_num)

        # Extract context encoding
        ego_neighbor_encoding = encoder_outputs['encoding']
        # route_lanes = inputs['route_lanes']

        if self.training:
            sampled_actions = inputs['sampled_actions'].reshape(B, P, -1)  # [B, P, num_actions*2]
            diffusion_time = inputs['diffusion_time']

            return {
                    "score": self.dit(
                        sampled_actions, 
                        diffusion_time,
                        ego_neighbor_encoding,
                        # route_lanes,
                        neighbor_current_mask,
                        current_states=current_states,
                        action_len=self._action_len
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
                            "cross_c": ego_neighbor_encoding, 
                            # "route_lanes": route_lanes,
                            "neighbor_current_mask": neighbor_current_mask,
                            "current_states": current_states,
                            "action_len": self._action_len
                        },
                        dpm_solver_params={},
                        model_wrapper_params={
                            "classifier_fn": self._guidance_fn,
                            "classifier_kwargs": {
                                "model": self.dit,
                                "model_condition": {
                                    "cross_c": ego_neighbor_encoding, 
                                    # "route_lanes": route_lanes,
                                    "neighbor_current_mask": neighbor_current_mask,
                                    "current_states": current_states,
                                    "action_len": self._action_len
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


class RouteEncoder(nn.Module):
    def __init__(self, route_num, lane_len, drop_path_rate=0.3, hidden_dim=192, tokens_mlp_dim=32, channels_mlp_dim=64):
        super().__init__()

        self._channel = channels_mlp_dim

        self.channel_pre_project = Mlp(in_features=4, hidden_features=channels_mlp_dim, out_features=channels_mlp_dim, act_layer=nn.GELU, drop=0.)
        self.token_pre_project = Mlp(in_features=route_num * lane_len, hidden_features=tokens_mlp_dim, out_features=tokens_mlp_dim, act_layer=nn.GELU, drop=0.)

        self.Mixer = MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate)

        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(in_features=channels_mlp_dim, hidden_features=hidden_dim, out_features=hidden_dim, act_layer=nn.GELU, drop=drop_path_rate)

    def forward(self, x):
        '''
        x: B, P, V, D
        '''
        # only x and x->x' vector, no boundary, no speed limit, no traffic light
        x = x[..., :4]

        B, P, V, _ = x.shape
        mask_v = torch.sum(torch.ne(x[..., :4], 0), dim=-1).to(x.device) == 0
        mask_p = torch.sum(~mask_v, dim=-1) == 0
        mask_b = torch.sum(~mask_p, dim=-1) == 0
        x = x.view(B, P * V, -1)

        valid_indices = ~mask_b.view(-1) 
        x = x[valid_indices] 

        x = self.channel_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.token_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.Mixer(x)

        x = torch.mean(x, dim=1)

        x = self.emb_project(self.norm(x))

        x_result = torch.zeros((B, x.shape[-1]), device=x.device)
        x_result[valid_indices] = x  # Fill in valid parts
        
        return x_result.view(B, -1)


class DiT(nn.Module):
    def __init__(self, 
                 sde: SDE, 
                 action_normalizer: ActionNormalizer,
                #  route_encoder: nn.Module, 
                 depth, 
                 output_dim, 
                 hidden_dim=192, 
                 heads=6, 
                 dropout=0.1, 
                 mlp_ratio=4.0, 
                 traj_dim = 6, 
                 action_t = 40,
                 future_len = 80,
                 act_dim = 2,
                 model_type="x_start",
                 tokens_mlp_dim=64,
                 channels_mlp_dim=128
        ):
        super().__init__()
        
        assert model_type in ["score", "x_start"], f"Unknown model type: {model_type}"
        self._model_type = model_type
        # self.route_encoder = route_encoder
        # self.agent_embedding = nn.Embedding(2, hidden_dim)
        # self.preproj = Mlp(in_features=output_dim, hidden_features=512, out_features=hidden_dim, act_layer=nn.GELU, drop=0.)
        
        # Trajectory encoder (similar to encoder's processing)
        self.traj_channel_pre_proj = Mlp(in_features=traj_dim, hidden_features=channels_mlp_dim, out_features=channels_mlp_dim, act_layer=nn.GELU, drop=0.)
        self.traj_token_pre_proj = Mlp(in_features=future_len, hidden_features=tokens_mlp_dim, out_features=tokens_mlp_dim, act_layer=nn.GELU, drop=0.)
        self.traj_encoder = nn.ModuleList([MixerBlock(tokens_mlp_dim, channels_mlp_dim, dropout) for _ in range(3)])
        self.traj_norm = nn.LayerNorm(channels_mlp_dim)
        self.traj_emb_proj = Mlp(in_features=channels_mlp_dim, hidden_features=hidden_dim, out_features=hidden_dim, act_layer=nn.GELU, drop=dropout)
        
        # Action encoder (similar to encoder's processing)
        self.action_channel_pre_proj = Mlp(in_features=act_dim, hidden_features=channels_mlp_dim, out_features=channels_mlp_dim, act_layer=nn.GELU, drop=0.)
        self.action_token_pre_proj = Mlp(in_features=action_t, hidden_features=tokens_mlp_dim, out_features=tokens_mlp_dim, act_layer=nn.GELU, drop=0.)
        self.action_encoder = nn.ModuleList([MixerBlock(tokens_mlp_dim, channels_mlp_dim, dropout) for _ in range(3)])
        self.action_norm = nn.LayerNorm(channels_mlp_dim)
        self.action_emb_proj = Mlp(in_features=channels_mlp_dim, hidden_features=hidden_dim, out_features=hidden_dim, act_layer=nn.GELU, drop=dropout)
        self.time_embedding = nn.Embedding(action_t, 192)
        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.blocks = nn.ModuleList([DiTBlock(hidden_dim, heads, dropout, mlp_ratio) for i in range(depth)])
        self.final_layer = FinalLayer(hidden_dim, output_dim)
        
        self._sde = sde
        self.marginal_prob_std = self._sde.marginal_prob_std
        self._action_normalizer = action_normalizer

        self.register_buffer('time', torch.arange(action_t).unsqueeze(0))
    
    @property
    def model_type(self):
        return self._model_type

    def forward(self, 
                x, 
                t, 
                cross_c, 
                # route_lanes, 
                neighbor_current_mask,
                current_states,
                action_len,
                action_dim=2,
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
        
        # Convert actions to trajectories using roll_out (similar to VBD)
        # This helps the model learn trajectory-level denoising
        # x is actions: [B, P, num_actions * 2]
        actions = x.reshape(B, P, -1, action_dim)
        actions = self._action_normalizer.inverse(actions)
        trajectories = roll_out(
            current_states,  # [B, P, 6]
            actions,
            dt=0.1,
            action_len=action_len,
            global_frame=False
        )  # [B, P, future_len, 6]
        
        # Use trajectory encoder (similar to encoder's processing)
        # trajectories: [B, P, future_len, 6]
        traj = trajectories.view(B * P, -1, current_states.shape[-1])  # [B*P, future_len, traj_dim]
        
        x_traj = self.traj_channel_pre_proj(traj)
        x_traj = x_traj.permute(0, 2, 1)
        x_traj = self.traj_token_pre_proj(x_traj)
        x_traj = x_traj.permute(0, 2, 1)
        for block in self.traj_encoder:
            x_traj = block(x_traj)
        
        # Pooling
        x_traj = torch.mean(x_traj, dim=1)
        x_traj = self.traj_emb_proj(self.traj_norm(x_traj))
        x_traj = x_traj.view(B, P, -1)

        # Use action encoder (similar to encoder's processing)
        # actions: [B, P, num_actions, 2]
        act = actions.view(B * P, -1, action_dim)  # [B*P, num_actions, act_dim]
        
        x = self.action_channel_pre_proj(act)
        x = x.permute(0, 2, 1)
        x = self.action_token_pre_proj(x)
        x = x.permute(0, 2, 1)
        for block in self.action_encoder:
            x = block(x)
        
        # Pooling
        x = torch.mean(x, dim=1)
        x = self.action_emb_proj(self.action_norm(x))
        x = x.view(B, P, -1)
        # Add time embedding
        # x = x + self.time_embedding(self.time)
        # x = x.view(B, P, -1)
        # x = x.mean(dim=2)  # [B, P, dim]
        x = x + x_traj

        # x_embedding = torch.cat([self.agent_embedding.weight[0][None, :], self.agent_embedding.weight[1][None, :].expand(P - 1, -1)], dim=0)  # (P, D)
        # x_embedding = x_embedding[None, :, :].expand(B, -1, -1) # (B, P, D)
        # x = x + x_embedding

        # route_encoding = self.route_encoder(route_lanes)
        # y = route_encoding
        # y = y + self.t_embedder(t)
        y = self.t_embedder(t)

        attn_mask = torch.zeros((B, P), dtype=torch.bool, device=x.device)
        attn_mask[:, 1:] = neighbor_current_mask
        
        for block in self.blocks:
            x = block(x, cross_c, y, attn_mask)  
        
        x = self.final_layer(x, y)
        
        if self._model_type == "score":
            return x / (self.marginal_prob_std(t)[:, None, None] + 1e-6)
        elif self._model_type == "x_start":
            return x
        else:
            raise ValueError(f"Unknown model type: {self._model_type}")


class CrossTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        heads, dim, dropout = 8, 256, 0.1
        self.cross_attention = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Dropout(dropout), 
                                 nn.Linear(dim*4, dim), nn.Dropout(dropout))

    def forward(self, query, key, relations, attn_mask=None, key_mask=None):
        # add relations to key and value
        key = key + relations
        value = key

        if key_mask is not None:
            attention_output, _ = self.cross_attention(query, key, value, key_padding_mask=key_mask)
        elif attn_mask is not None:
            attention_output, _ = self.cross_attention(query, key, value, attn_mask=attn_mask)
        else:
            attention_output, _ = self.cross_attention(query, key, value)

        attention_output = self.norm_1(attention_output)
        output = self.norm_2(self.ffn(attention_output) + attention_output)

        return output


class TransformerDecoder(nn.Module):
    def __init__(self, future_len, agents_len, action_len, input_dim=5, ouptut_dim = 2,  causal = True):
        super().__init__()
        self._future_len = future_len
        self._action_len = action_len
        self._agents_len = agents_len
        self._future_len = future_len // action_len
        self._input_dim = input_dim
        self._output_dim = ouptut_dim

        self.time_embedding = nn.Embedding(self._future_len, 256)
        self.attention_layers = nn.ModuleList([CrossTransformer() for _ in range(4)])
        self.encoder = nn.Sequential(nn.Linear(self._input_dim, 128), nn.ReLU(), nn.Linear(128, 256))
        self.decoder = nn.Sequential(nn.Linear(256, 128), nn.ELU(), nn.Dropout(0.1), nn.Linear(128, self._output_dim))
        
        self.register_buffer('casual_mask', self.generate_casual_mask(causal))
        self.register_buffer('time', torch.arange(self._future_len).unsqueeze(0))

    def generate_casual_mask(self, causal=True):
        if not causal:
            return torch.zeros(self._agents_len, self._future_len, self._agents_len * self._future_len, dtype=bool)
        
        # Initialize a zero mask
        mask = torch.zeros(self._agents_len, self._future_len, self._agents_len * self._future_len)

        # An agent can attend to all of its own actions
        for i in range(self._agents_len):
            mask[i, :, i*self._future_len:(i+1)*self._future_len] = 1.0

        # An agent can attend to other agents from all previous timesteps but not future timesteps
        for i in range(self._agents_len):
            for j in range(self._agents_len):
                if i != j:
                    for t in range(self._future_len):
                        mask[i, t, j*self._future_len:j*self._future_len+t+1] = 1.0
        
        # Convert to boolean mask
        mask = mask.bool().logical_not()

        return mask

    def forward(self, noisy_trajectories, noise_level, encodings, relations, mask):
        '''
        noisy_trajectories: [B, Na, T_f, 5]
        '''
        # get query
        noisy_trajectories = torch.reshape(
            noisy_trajectories, 
            (
                -1, 
                self._agents_len, self._future_len, self._action_len, 
                self._input_dim
            )
        )
        future_states = self.encoder(noisy_trajectories)
        future_states = future_states.max(dim=3).values # [B, Na, T, 256]
        time_embedding = self.time_embedding(self.time) # [1, T, 256]
        query = future_states + time_embedding[:, None] # [B, Na, T, 256]
        query = query + noise_level[:, :, None, :] 

        # decode denoised actions
        query_content_list = []
        for i in range(self._agents_len):
            query_content = self.attention_layers[0](
                query[:, i], 
                query.reshape(-1, self._agents_len*self._future_len, 256), 
                relations[:, i, :self._agents_len].repeat_interleave(self._future_len, dim=1),
                attn_mask=self.casual_mask[i]) # [B, T, 256]
            query_content = self.attention_layers[1](query_content, encodings, relations[:, i], key_mask=mask) # [B, T, 256]
            query_content_list.append(query_content)

        query_content_stack = torch.stack(query_content_list, dim=1) # [B, Na, T, 256] 
        query_content_stack = query_content_stack + query
    
        query_content_list = []
        for i in range(self._agents_len):
            query_content = self.attention_layers[2](
                query_content_stack[:, i],
                query_content_stack.reshape(-1, self._agents_len*self._future_len, 256),
                relations[:, i, :self._agents_len].repeat_interleave(self._future_len, dim=1),
                attn_mask=self.casual_mask[i]) # [B, T, 256]
            query_content = self.attention_layers[3](query_content, encodings, relations[:, i], key_mask=mask) # [B, T, 256]
            query_content_list.append(query_content)
        
        query_content_stack = torch.stack(query_content_list, dim=1) # [B, Na, T, 256] 
        actions = self.decoder(query_content_stack) 

        return actions