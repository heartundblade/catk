import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import knn

from src.dp.model.module.mixer import MixerBlock
from src.dp.model.module.timm import Mlp, DropPath
from src.dp.model.module.local_attention import MultiheadAttentionLocal
from src.dp.utils.train_utils import batch_transform_trajs_to_local_frame, batch_calculate_relations, batch_transform_maps_to_local_frame
from src.dp.model.module.rel_emb import FourierEmbedding, RelationEncoder


def _apply_basic_init(module, fn, skip_types):
    """Pre-order traversal that skips entire subtrees of skip_types."""
    if isinstance(module, skip_types):
        return
    fn(module)
    for child in module.children():
        _apply_basic_init(child, fn, skip_types)


class Encoder(nn.Module):
    def __init__(self, config, rel_encoder=None):
        super().__init__()

        self.hidden_dim = config.hidden_dim

        self.token_num = config.agent_num + config.static_map_num + config.lane_num + config.roadline_num  # + config.traffic_light_num
        self.local_attn_k = getattr(config, 'local_attn_k', 16)

        # self.agents_encoder = AgentFusionEncoder(config.time_len, drop_path_rate=config.encoder_drop_path_rate, hidden_dim=config.hidden_dim, depth=config.encoder_depth)
        self.agents_encoder = AgentEncoder(
            config.time_len, 
            drop_path_rate=config.encoder_drop_path_rate, 
            hidden_dim=config.hidden_dim, 
            depth_self=config.encoder_depth, 
            depth_cross=1,
            num_heads=config.num_heads
        )

        self.shared_spatial_enc = ChannelPreProject(
            input_dim=4, channels_mlp_dim=128, num_freq_bands=32
        )
        self.static_map_encoder = StaticMapFusionEncoder(
            config.static_map_len, drop_path_rate=config.encoder_drop_path_rate, hidden_dim=config.hidden_dim,
            shared_spatial_enc=self.shared_spatial_enc
        )
        self.lane_encoder = LaneFusionEncoder(
            config.lane_len,
            drop_path_rate=config.encoder_drop_path_rate,
            hidden_dim=config.hidden_dim,
            depth=config.polyline_encoder_depth,
            shared_spatial_enc=self.shared_spatial_enc
        )
        self.roadline_encoder = RoadlineFusionEncoder(
            config.roadline_len,
            drop_path_rate=config.encoder_drop_path_rate,
            hidden_dim=config.hidden_dim,
            depth=config.polyline_encoder_depth,
            shared_spatial_enc=self.shared_spatial_enc
        )

        # self.traffic_light_encoder = TrafficLightEncoder(config.hidden_dim)

        _rel_kwargs = dict(hidden_dim=config.hidden_dim, num_freq_bands=64)
        self.rel_encoder_m2m = RelationEncoder(**_rel_kwargs)
        self.rel_encoder_a2h = RelationEncoder(**_rel_kwargs)
        self.rel_encoder_a2m = RelationEncoder(**_rel_kwargs)
        self.rel_encoder_a2a = RelationEncoder(**_rel_kwargs)

        self.fusion_m2m = FusionEncoder(
            hidden_dim=config.hidden_dim, 
            num_heads=config.num_heads, 
            drop_path_rate=config.encoder_drop_path_rate, 
            depth=1, 
            device=config.device,
            rel_encoder=self.rel_encoder_m2m,
        )

        self.fusion_layers = nn.ModuleList([
            nn.ModuleDict({
                'a2h': CrossFusionEncoder(
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    drop_path_rate=config.encoder_drop_path_rate,
                    depth=1,
                    device=config.device,
                    rel_encoder=self.rel_encoder_a2h,
                ),
                'a2m': CrossFusionEncoder(
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    drop_path_rate=config.encoder_drop_path_rate,
                    depth=1,
                    device=config.device,
                    rel_encoder=self.rel_encoder_a2m,
                ),
                'a2a': FusionEncoder(
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    drop_path_rate=config.encoder_drop_path_rate,
                    depth=1,
                    device=config.device,
                    rel_encoder=self.rel_encoder_a2a,
                ),
            })
            for _ in range(config.encoder_depth)
        ])

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
                nn.init.normal_(m.weight, mean=0.0, std=1.0)

        skip_types = (FourierEmbedding, RelationEncoder, ChannelPreProject)
        _apply_basic_init(self, _basic_init, skip_types)



    def forward(self, inputs):

        encoder_outputs = {}

        # agents
        agents = inputs['agents_history']
        agents_type = inputs['agents_type']

        # vector maps
        lanes = inputs['lanes']
        lanes_speed_limit = inputs['lanes_speed_limit']
        lanes_has_speed_limit = inputs['lanes_has_speed_limit']
        roadlines = inputs['roadlines']
        static_maps = inputs['static_maps']

        B = agents.shape[0]

        agents_local = batch_transform_trajs_to_local_frame(agents)

        encoding_agents_t, encoding_agents, agents_t_mask, agents_temporal_rel, agents_mask = self.agents_encoder(agents_local, agents_type)

        static_maps_local = batch_transform_maps_to_local_frame(static_maps)
        lanes_local = batch_transform_maps_to_local_frame(lanes)

        # Transform lanes_stop_point to the same local frame as lanes
        ref_idx = 0
        ref_x = lanes[:, :, ref_idx, 0]
        ref_y = lanes[:, :, ref_idx, 1]
        ref_cos = lanes[:, :, ref_idx, 2]
        ref_sin = lanes[:, :, ref_idx, 3]
        sp = inputs['lanes_stop_point']  # [B, N, 2]
        dx = sp[..., 0] - ref_x
        dy = sp[..., 1] - ref_y
        sp_local_x = dx * ref_cos + dy * ref_sin
        sp_local_y = -dx * ref_sin + dy * ref_cos
        sp_local = torch.stack([sp_local_x, sp_local_y], dim=-1)
        sp_local[torch.all(sp == 0, dim=-1)] = 0.0

        roadlines_local = batch_transform_maps_to_local_frame(roadlines)

        encoding_static, static_mask = self.static_map_encoder(static_maps_local)
        encoding_lanes, lanes_mask = self.lane_encoder(lanes_local, sp_local, lanes_speed_limit, lanes_has_speed_limit)
        encoding_roadlines, roadlines_mask = self.roadline_encoder(roadlines_local)

        # # traffic lights
        # traffic_lights = inputs['traffic_light_points']
        # encoding_traffic_lights, traffic_lights_mask = self.traffic_light_encoder(traffic_lights)

        encoding_input = torch.cat([encoding_agents, encoding_static, encoding_lanes, encoding_roadlines], dim=1)  # , encoding_traffic_lights
        encoding_mask = torch.cat([agents_mask, static_mask, lanes_mask, roadlines_mask], dim=1).view(-1)  # , traffic_lights_mask

        relations, all_elements = batch_calculate_relations(
            agents, static_maps, lanes, roadlines,  # traffic_lights,
            encoding_mask=encoding_mask.reshape(B, self.token_num),
            device=agents.device
        )
        # relations_pos = relations[..., :2]
        # relations_angle = relations[..., 2:]
        # encoded_pos = self.relation_pos_encoder(relations_pos)
        # encoded_angle = self.relation_angle_encoder(relations_angle)
        encoder_outputs['relations'] = relations

        n_agents = encoding_agents.shape[1]
        n_static = encoding_static.shape[1]
        n_lanes = encoding_lanes.shape[1]
        n_roadlines = encoding_roadlines.shape[1]
        # n_tls = encoding_traffic_lights.shape[1]
        

        map_start = n_agents
        map_end = n_agents + n_static + n_lanes + n_roadlines  # + n_tls
        N_map = map_end - map_start

        encoding_map = encoding_input[:, map_start:map_end, :]
        encoding_map_mask = encoding_mask.view(B, -1)[:, map_start:map_end]
        m2m_rel = relations[:, map_start:map_end, :, :][:, :, map_start:map_end]

        dist_m2m = torch.norm(m2m_rel[..., :2], dim=-1)
        dist_m2m = dist_m2m.masked_fill(encoding_map_mask.unsqueeze(1), 1e9)
        _, m2m_index_pair = torch.topk(dist_m2m, k=self.local_attn_k, dim=-1, largest=False)
        m2m_index_pair = m2m_index_pair.reshape(B * N_map, self.local_attn_k)

        encoding_map_fused = self.fusion_m2m(encoding_map, encoding_map_mask, m2m_index_pair, m2m_rel)

        encoding_agents_mask = encoding_mask.view(B, -1)[:, :n_agents]
        a2m_rel = relations[:, :n_agents, :, :][:, :, map_start:map_end]

        dist_a2m = torch.norm(a2m_rel[..., :2], dim=-1)
        dist_a2m = dist_a2m.masked_fill(encoding_agents_mask.unsqueeze(-1), 1e9)
        dist_a2m = dist_a2m.masked_fill(encoding_map_mask.unsqueeze(1), 1e9)
        _, a2m_index_pair = torch.topk(dist_a2m, k=32, dim=-1, largest=False)
        a2m_index_pair = a2m_index_pair.reshape(B * n_agents, 32)

        a2a_rel = relations[:, :n_agents, :, :][:, :, :n_agents]

        a2a_dist = torch.norm(a2a_rel[..., :2], dim=-1)
        a2a_dist = a2a_dist.masked_fill(encoding_agents_mask.unsqueeze(1), 1e9)
        _, a2a_index_pair = torch.topk(a2a_dist, k=self.local_attn_k, dim=-1, largest=False)
        a2a_index_pair = a2a_index_pair.reshape(B * n_agents, self.local_attn_k)

        V = encoding_agents_t.shape[2]
        a2h_kv = encoding_agents_t.reshape(B, n_agents * V, -1)
        a2h_kv_mask = agents_t_mask.reshape(B, n_agents * V)
        current_frame_idx = torch.arange(n_agents, device=encoding_agents.device) * V + (V - 1)
        a2h_kv_mask[:, current_frame_idx] = False

        a2h_index_pair = (torch.arange(n_agents, device=encoding_agents.device)[:, None] * V + torch.arange(V, device=encoding_agents.device)[None, :])
        a2h_index_pair = a2h_index_pair.unsqueeze(0).expand(B, -1, -1).reshape(B * n_agents, V)

        a2h_rel = torch.zeros(B, n_agents, n_agents * V, 4, device=encoding_agents.device)
        for i in range(n_agents):
            a2h_rel[:, i, i * V:(i + 1) * V, :] = agents_temporal_rel[:, i, :, :]

        a2h_self_rel_mask = torch.zeros(B * n_agents, V, dtype=torch.bool, device=encoding_agents.device)
        a2h_self_rel_mask[:, V - 1] = True

        for layer in self.fusion_layers:
            encoding_agents = layer['a2h'](
                encoding_agents, a2h_kv,
                encoding_agents_mask, a2h_kv_mask,
                a2h_index_pair, a2h_rel,
                self_rel_mask=a2h_self_rel_mask
            )
            encoding_agents = layer['a2m'](
                encoding_agents, encoding_map_fused,
                encoding_agents_mask, encoding_map_mask,
                a2m_index_pair, a2m_rel
            )
            encoding_agents = layer['a2a'](
                encoding_agents, encoding_agents_mask, a2a_index_pair, a2a_rel
            )


        encoding_fused = torch.cat([
            encoding_agents,
            encoding_map_fused,
        ], dim=1)
        
        # # ---- DEBUG: 保存 index_pair 数据到文件，确认后删除 ----
        # debug_index_pair(index_pair, all_elements, encoding_mask, B, N, self.local_attn_k,
        #                  n_agents=64, n_static=20,
        #                  n_lanes=256, n_roadlines=128,
        #                  n_traffic_lights=20)
        # # ---- DEBUG END ----

        encoder_outputs['encoding'] = encoding_fused
        # encoder_outputs['encoding'] = encoding_input
        encoder_outputs['encoding_mask'] = encoding_mask.view(B, -1)

        return encoder_outputs


def debug_index_pair(index_pair, all_elements, encoding_mask, B, N, local_attn_k,
                     n_agents, n_static, n_lanes, n_roadlines, n_traffic_lights=0,
                     save_dir='/home/zhanghailiang/Repos/catk/logs/debug_index_pair'):
    """
    将 index_pair 和 token 位置保存到 .npz 文件，方便离线可视化。
    同时打印统计信息，无需 matplotlib 也能判断邻居选择是否合理。
    """
    import numpy as np
    import os

    os.makedirs(save_dir, exist_ok=True)

    # ---- 保存核心数据 ----
    pos = all_elements[0, :, :2].cpu().numpy()          # [N, 2] x, y
    cos_sin = all_elements[0, :, 2:4].cpu().numpy()     # [N, 2] cos, sin
    mask = encoding_mask.view(B, N)[0].cpu().numpy()     # [N] True=invalid
    ip = index_pair[:N].cpu().numpy()                    # [N, L] 前 N 个 query

    # 类型边界
    type_boundaries = [0, n_agents, n_agents + n_static,
                       n_agents + n_static + n_lanes,
                       n_agents + n_static + n_lanes + n_roadlines, N]
    type_names = ['agent', 'static_map', 'lane', 'roadline', 'traffic_light']
    type_ids = np.zeros(N, dtype=np.int32)
    for t in range(len(type_boundaries) - 1):
        type_ids[type_boundaries[t]:type_boundaries[t + 1]] = t

    np.savez(os.path.join(save_dir, 'index_pair_debug.npz'),
             pos=pos, cos_sin=cos_sin, mask=mask, index_pair=ip,
             type_ids=type_ids, type_names=type_names,
             local_attn_k=local_attn_k)

    # ---- 打印统计信息 ----
    print(f'[DEBUG index_pair] ========================================')
    print(f'[DEBUG index_pair] N={N}, L={local_attn_k}, B={B}')
    print(f'[DEBUG index_pair] data saved to {save_dir}/index_pair_debug.npz')

    for t in range(len(type_boundaries) - 1):
        start, end = type_boundaries[t], type_boundaries[t + 1]
        valid_in_type = [i for i in range(start, end) if not mask[i]]
        if not valid_in_type:
            continue

        # 统计该类型 token 的邻居有效性
        nbr_type_counts = np.zeros(len(type_names), dtype=np.int32)
        total_valid_nbrs = 0
        total_nbrs = 0
        total_dist = 0.0
        sample_count = 0

        for qi in valid_in_type:
            qx, qy = pos[qi]
            for ni in ip[qi]:
                total_nbrs += 1
                if ni < 0 or ni >= N or mask[ni]:
                    continue
                total_valid_nbrs += 1
                nbr_type_counts[type_ids[ni]] += 1
                nx, ny = pos[ni]
                total_dist += np.sqrt((qx - nx)**2 + (qy - ny)**2)
                sample_count += 1

        avg_dist = total_dist / sample_count if sample_count > 0 else 0.0
        valid_ratio = total_valid_nbrs / total_nbrs if total_nbrs > 0 else 0.0

        print(f'[DEBUG index_pair] {type_names[t]:>14s}: {len(valid_in_type)} valid tokens, '
              f'valid_nbr_ratio={valid_ratio:.2%}, avg_dist={avg_dist:.2f}')
        nbr_str = ', '.join(f'{type_names[j]}={nbr_type_counts[j]}' for j in range(len(type_names)))
        print(f'[DEBUG index_pair]   neighbor types: {nbr_str}')

    print(f'[DEBUG index_pair] ========================================')


class ChannelPreProject(nn.Module):
    """
    Spatial encoding: FourierEmbedding for (r, theta, heading_angle).
    """
    def __init__(self, input_dim=4, channels_mlp_dim=128, num_freq_bands=64):
        super().__init__()
        assert input_dim == 4, "ChannelPreProject expects [x, y, sin, cos]"
        self.spatial_encoder = FourierEmbedding(input_dim=3, hidden_dim=channels_mlp_dim, num_freq_bands=num_freq_bands)

    def forward(self, x):
        r = torch.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2)
        r = torch.where(r < 1e-4, torch.zeros_like(r), r)
        theta = torch.atan2(x[..., 1], x[..., 0])
        angle = torch.atan2(x[..., 3], x[..., 2])
        return self.spatial_encoder(torch.stack([r, theta, angle], dim=-1))



class LocalSelfAttentionBlock(nn.Module):
    """
    Local version of SelfAttentionBlock using MultiheadAttentionLocal for sparse/local attention.
    Replaces full NxN self-attention with index_pair-based local attention to reduce memory.

    Accepts index_pair to specify which keys each query attends to.
    """
    def __init__(self, dim=192, heads=6, dropout=0.1, mlp_ratio=4.0, attention_mode="full"):
        super().__init__()

        self.num_heads = heads
        self.head_dim = dim // heads
        self.attn_dropout_rate = dropout
        self.attention_mode = attention_mode
        self.norm1 = nn.LayerNorm(dim)

        self.drop_path = DropPath(dropout) if dropout > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=dropout)
        self.rel_proj = nn.Linear(dim, dim)
        self.rel_v_proj = nn.Linear(dim, dim)

        self.to_g = nn.Linear(2 * dim, dim)
        self.to_s = nn.Linear(dim, dim)

        self.local_attn = MultiheadAttentionLocal(dim, heads, dropout)

    def forward(self, x, mask, index_pair, rel_enc_sparse=None):
        """
        Args:
            x: [B, N, D] input tokens
            mask: [B, N] bool, True=invalid/padded token
            index_pair: [B*N, L] local attention indices in [0, N-1], -1 for padding
            rel_enc_sparse: [B*N, L, D] pre-encoded sparse relations for index_pair neighbors
        """
        B, N, D = x.shape
        H = self.num_heads
        d = self.head_dim
        L = index_pair.shape[1]

        shortcut = x
        x_norm = self.norm1(x)

        batch_cnt = [N] * B

        local_rel = None
        local_rel_v = None
        if rel_enc_sparse is not None:
            local_rel = self.rel_proj(rel_enc_sparse).reshape(B * N, L, H, d)
            if self.attention_mode == "full":
                local_rel_v = self.rel_v_proj(rel_enc_sparse).reshape(B * N, L, H, d)

            query_idx = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1).reshape(-1)
            self_mask = (index_pair == query_idx.unsqueeze(-1))
            local_rel = local_rel.masked_fill(self_mask.unsqueeze(-1).unsqueeze(-1), 0.0)
            if local_rel_v is not None:
                local_rel_v = local_rel_v.masked_fill(self_mask.unsqueeze(-1).unsqueeze(-1), 0.0)

        local_attn_mask = None
        if mask is not None:
            batch_ids = torch.arange(B, device=x.device).repeat_interleave(N)
            local_attn_mask = mask[batch_ids[:, None], index_pair.clamp(min=0)]
            local_attn_mask = local_attn_mask | (index_pair == -1)

        value, _ = self.local_attn(
            query=x_norm.reshape(B * N, D),
            key=x_norm.reshape(B * N, D),
            value=x_norm.reshape(B * N, D),
            index_pair=index_pair,
            query_batch_cnt=batch_cnt,
            key_batch_cnt=batch_cnt,
            attn_mask=local_attn_mask,
            relation_encodings=local_rel,
            rel_v=local_rel_v,
        )
        value = value.reshape(B, N, D)

        g = torch.sigmoid(self.to_g(torch.cat([value, shortcut], dim=-1)))
        x = value + g * (self.to_s(shortcut) - value)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if mask is not None:
            x = x * (~mask).float().unsqueeze(-1)
        return x


class AgentFusionEncoder(nn.Module):
    def __init__(self, time_len, drop_path_rate=0.3, hidden_dim=192, depth=3, tokens_mlp_dim=64, channels_mlp_dim=128):
        super().__init__()

        self._hidden_dim = hidden_dim
        self._channel = channels_mlp_dim

        self.type_emb = nn.Embedding(4, channels_mlp_dim)

        self.channel_pre_project = Mlp(in_features=8+1, hidden_features=channels_mlp_dim, out_features=channels_mlp_dim, act_layer=nn.GELU, drop=0.)
        self.token_pre_project = Mlp(in_features=time_len, hidden_features=tokens_mlp_dim, out_features=tokens_mlp_dim, act_layer=nn.GELU, drop=0.)

        self.blocks = nn.ModuleList([MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate) for i in range(depth)])

        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(in_features=channels_mlp_dim, hidden_features=hidden_dim, out_features=hidden_dim, act_layer=nn.GELU, drop=drop_path_rate)


    def forward(self, x, agent_type):
        '''
        x: B, P, V, D (x, y, cos, sin, vx, vy, w, l, z)
        agent_type: B, P, 1 (type)
        '''
        # aggragate infos from history with timestep diff embedding
        # and spatial embedding to the last pos to form feature
        x = x[..., :8]

        pos = x[:, :, -1, :8].clone() # x, y, cos, sin
        # agent: [1,0,0,0]
        pos[..., -4:] = 0.0
        pos[..., -4] = 1.0
        
        B, P, V, _ = x.shape
        mask_v = torch.sum(torch.ne(x[..., :8], 0), dim=-1).to(x.device) == 0
        mask_p = torch.sum(~mask_v, dim=-1) == 0
        x = torch.cat([x, (~mask_v).float().unsqueeze(-1)], dim=-1)
        x = x.view(B * P, V, -1)

        valid_indices = ~mask_p.view(-1) 
        x = x[valid_indices] 

        x = self.channel_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.token_pre_project(x)
        x = x.permute(0, 2, 1)
        for block in self.blocks:
            x = block(x)

        # pooling
        x = torch.mean(x, dim=1)

        agent_type = agent_type.long().view(B * P)
        agent_type = agent_type[valid_indices]
        type_embedding = self.type_emb(agent_type)  # Type embedding for valid data
        x = x + type_embedding

        x = self.emb_project(self.norm(x))

        x_result = torch.zeros((B * P, x.shape[-1]), device=x.device)
        x_result[valid_indices] = x  # Fill in valid parts
        
        return x_result.view(B, P, -1) , mask_p.reshape(B, -1), pos.view(B, P, -1)


class AgentEncoder(nn.Module):
    def __init__(self, time_len, drop_path_rate=0.3, hidden_dim=192, depth_self=2,
                 depth_cross=1, num_heads=6):
        super().__init__()

        self._hidden_dim = hidden_dim
        self._channel = hidden_dim
        self._time_len = time_len
        self._num_heads = num_heads

        self.type_emb = nn.Embedding(4, hidden_dim)

        self.state_proj = FourierEmbedding(input_dim=8, hidden_dim=hidden_dim, num_freq_bands=32)
        self.input_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # ---- Phase 1: Causal Self-Attention with relative spatio-temporal encoding ----
        self.sa_layers = nn.ModuleList([
            MultiheadAttentionLocal(
                embed_dim=hidden_dim, num_heads=num_heads,
                dropout=drop_path_rate
            ) for _ in range(depth_self)
        ])

        self.a2h_rel_encoder = nn.Sequential(
            FourierEmbedding(input_dim=4, hidden_dim=hidden_dim, num_freq_bands=32),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.rel_attn_proj = nn.Linear(hidden_dim, hidden_dim)
        self.rel_v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.sa_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(depth_self)
        ])
        self.sa_ffn_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(depth_self)
        ])
        self.sa_ffn_layers = nn.ModuleList([
            Mlp(
                in_features=hidden_dim, hidden_features=hidden_dim * 4,
                out_features=hidden_dim, act_layer=nn.GELU, drop=drop_path_rate
            ) for _ in range(depth_self)
        ])

        self.emb_project = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, x, agent_type):
        '''
        x: B, P, V, D (x, y, cos, sin, vx, vy, w, l, z)
        agent_type: B, P, 1 (type)
        '''
        B, P, V, _ = x.shape

        mask_v = torch.sum(torch.ne(x[..., :8], 0), dim=-1) == 0  # [B, P, V]
        mask_p = torch.sum(~mask_v, dim=-1) == 0                  # [B, P]

        x = x.view(B * P, V, -1)                      # [B*P, V, 9]
        mask_v_flat = mask_v.view(B * P, V)           # [B*P, V]
        valid_indices = ~mask_p.view(-1)              # [B*P]
        x = x[valid_indices]                          # [N, V, 9]
        mask_v_valid = mask_v_flat[valid_indices]     # [N, V]
        x_raw = x                                     # [N, V, 9] save raw for temporal relation computation
        N = x.shape[0]

        if N == 0:
            x_res = torch.zeros(B * P, self._hidden_dim, device=x.device)
            x_res = torch.zeros(B * P, V, self._hidden_dim, device=x.device)
            mask_t = torch.ones(B * P, V, dtype=torch.bool, device=x.device)
            a2h_rel = torch.zeros(B * P, V, 4, device=x.device)
            return (x_res.view(B, P, V, -1),
                    x_res.view(B, P, -1),
                    mask_t.view(B, P, V),
                    a2h_rel.view(B, P, V, -1),
                    mask_p.reshape(B, -1))

        motion = x_raw[:, 1:, :2] - x_raw[:, :-1, :2]
        motion = torch.cat([x_raw[:, :1, 4:6], motion], dim=1)
        vel = x_raw[..., 4:6]
        hvec = x_raw[..., 2:4]

        motion_norm = torch.norm(motion, p=2, dim=-1)
        motion_heading = torch.atan2(
            hvec[..., 0] * motion[..., 1] - hvec[..., 1] * motion[..., 0],
            (hvec * motion).sum(-1)
        )
        vel_norm = torch.norm(vel, p=2, dim=-1)
        vel_heading = torch.atan2(
            hvec[..., 0] * vel[..., 1] - hvec[..., 1] * vel[..., 0],
            (hvec * vel).sum(-1)
        )
        heading = torch.atan2(x_raw[..., 3], x_raw[..., 2])
        x_state = self.state_proj(torch.stack([
            motion_norm, motion_heading,
            vel_norm, vel_heading,
            heading,
            x_raw[..., 6], x_raw[..., 7], x_raw[..., 8]
        ], dim=-1))
        x = self.input_proj(x_state)  # [N, V, C]

        agent_type = agent_type.long().view(B * P)
        agent_type = agent_type[valid_indices]  # [N, 1]

        x = x + self.type_emb(agent_type).unsqueeze(1)  # [N, V, C]
        x = self.emb_project(x)

        pos = x_raw[..., :2]                                    # [N, V, 2]
        heading = torch.atan2(x_raw[..., 3], x_raw[..., 2])  # [N, V]

        rel_pos = pos.unsqueeze(1) - pos.unsqueeze(2)            # [N, V, V, 2]: key_j - query_i
        dheading = heading.unsqueeze(1) - heading.unsqueeze(2)  # [N, V, V]: key_j - query_i

        rel_dist = torch.sqrt(rel_pos[..., 0] ** 2 + rel_pos[..., 1] ** 2)  # [N, V, V]
        rel_dist = torch.where(rel_dist < 1e-4, torch.zeros_like(rel_dist), rel_dist)
        rel_theta = torch.atan2(rel_pos[..., 1], rel_pos[..., 0])                   # [N, V, V]
        rel_theta = torch.where(rel_dist < 1e-4, torch.zeros_like(rel_theta), rel_theta)

        dt = (torch.arange(V, device=x.device, dtype=torch.float32).unsqueeze(0)
              - torch.arange(V, device=x.device, dtype=torch.float32).unsqueeze(1)).unsqueeze(0) / V  # [1, V, V]

        a2h_rel = torch.stack([
            rel_dist, rel_theta, dheading, dt.expand(N, -1, -1)
        ], dim=-1)                                              # [N, V, V, 4]

        a2h_rel_enc = self.a2h_rel_encoder(a2h_rel)  # [N, V, V, C]
        rel_attn = self.rel_attn_proj(a2h_rel_enc).reshape(N * V, V, self._num_heads, -1)
        rel_v = self.rel_v_proj(a2h_rel_enc).reshape(N * V, V, self._num_heads, -1)

        arange = torch.arange(V, device=x.device)
        index_pair = arange.unsqueeze(0).expand(V, -1).clone()            # [V, V]
        index_pair = index_pair.unsqueeze(0).expand(N, -1, -1).reshape(N * V, V)  # [N*V, V]

        # Causal mask: query i can only attend to key j <= i (past + current)
        causal_mask = arange.unsqueeze(1) < arange.unsqueeze(0)            # [V, V]: True where key > query
        causal_mask = causal_mask.unsqueeze(0).expand(N, -1, -1).reshape(N * V, V)  # [N*V, V]

        # Key padding mask expanded to [N*V, V]
        key_padding_2d = mask_v_valid.unsqueeze(1).expand(N, V, V).reshape(N * V, V)
        attn_mask = causal_mask | key_padding_2d  # [N*V, V], True = masked

        # Ensure self-key is never masked (prevents NaN when early frames are padding)
        self_mask = torch.eye(V, device=x.device, dtype=torch.bool).unsqueeze(0).expand(N, -1, -1).reshape(N * V, V)
        attn_mask = attn_mask & ~self_mask

        query_batch_cnt = [V] * N
        key_batch_cnt = [V] * N

        # ---- Phase 1: Causal Self-Attention with relative spatio-temporal encoding ----
        for i, attn in enumerate(self.sa_layers):
            residual = x
            x_norm = self.sa_norms[i](x)
            x_flat = x_norm.reshape(N * V, self._hidden_dim)
            x_out, _ = attn(
                query=x_flat, key=x_flat, value=x_flat,
                index_pair=index_pair,
                query_batch_cnt=query_batch_cnt,
                key_batch_cnt=key_batch_cnt,
                relation_encodings=rel_attn,
                rel_v=rel_v,
                attn_mask=attn_mask,
            )
            x = x_out.reshape(N, V, self._hidden_dim) + residual

            residual = x
            x_norm = self.sa_ffn_norms[i](x)
            x = self.sa_ffn_layers[i](x_norm)
            x = x + residual

        # ---- Temporal relations: current frame vs all historical frames ----
        # Relation format: [local_x, local_y, cos_theta_diff, sin_theta_diff]
        ref_x = x_raw[:, -1, 0].unsqueeze(-1)
        ref_y = x_raw[:, -1, 1].unsqueeze(-1)
        ref_cos = x_raw[:, -1, 2].unsqueeze(-1)
        ref_sin = x_raw[:, -1, 3].unsqueeze(-1)

        dx = ref_x - x_raw[..., 0]
        dy = ref_y - x_raw[..., 1]
        local_x = dx * ref_cos + dy * ref_sin
        local_y = -dx * ref_sin + dy * ref_cos
        dcos = ref_cos * x_raw[..., 2] + ref_sin * x_raw[..., 3]
        dsin = ref_sin * x_raw[..., 2] - ref_cos * x_raw[..., 3]
        a2h_rel = torch.stack([local_x, local_y, dcos, dsin], dim=-1)  # [N, V, 4]

        x_a_last = x[:, -1, :]  # [N, C]

        x_res = torch.zeros((B * P, V, x.shape[-1]), device=x.device)
        x_res[valid_indices] = x

        x_last = torch.zeros((B * P, x_a_last.shape[-1]), device=x.device)
        x_last[valid_indices] = x_a_last

        mask_t = torch.ones((B * P, V), dtype=torch.bool, device=x.device)
        mask_t[valid_indices] = mask_v_valid

        a2h_rel_res = torch.zeros((B * P, V, 4), device=x.device)
        a2h_rel_res[valid_indices] = a2h_rel

        return (x_res.view(B, P, V, -1),
                x_last.view(B, P, -1),
                mask_t.view(B, P, V),
                a2h_rel_res.view(B, P, V, -1),
                mask_p.reshape(B, -1))


class StaticMapFusionEncoder(nn.Module):
    def __init__(self, static_map_len, drop_path_rate=0.3, depth=3, tokens_mlp_dim=64, hidden_dim=192, channels_mlp_dim=128, device='cuda',
                 shared_spatial_enc: ChannelPreProject = None):
        super().__init__()

        self._hidden_dim = hidden_dim

        self._map_len = static_map_len
        self._channel = channels_mlp_dim

        self.type_emb = nn.Embedding(3, channels_mlp_dim)

        if shared_spatial_enc is not None:
            self.channel_pre_project = shared_spatial_enc
        else:
            self.channel_pre_project = ChannelPreProject(
                input_dim=4, channels_mlp_dim=channels_mlp_dim, num_freq_bands=32
            )

        self.point_norm = nn.LayerNorm(channels_mlp_dim)
        self.query_token = nn.Parameter(torch.randn(1, 1, channels_mlp_dim) * 0.02)
        self.attn_pool = nn.MultiheadAttention(
            embed_dim=channels_mlp_dim, num_heads=4, dropout=drop_path_rate, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(channels_mlp_dim)
        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(
            in_features=channels_mlp_dim, hidden_features=hidden_dim,
            out_features=hidden_dim, act_layer=nn.GELU, drop=drop_path_rate
        )

    def forward(self, x):
        '''
        x: B, P, V, D (x, y, cos, sin, type)
        '''
        B, P, V, _ = x.shape
        static_map_type = x[:, :, 0, 4]
        x = x[..., :4]

        mask_v = torch.sum(torch.ne(x[..., :4], 0), dim=-1).to(x.device) == 0
        mask_p = torch.sum(~mask_v, dim=-1) == 0
        x = x.view(B * P, V, -1)

        valid_indices = ~mask_p.view(-1) 
        x = x[valid_indices] 

        x = self.channel_pre_project(x)
        x = self._mix(x)

        static_map_type = static_map_type.long().view(B * P)
        static_map_type = static_map_type[valid_indices]
        type_embedding = self.type_emb(static_map_type)
        
        x = x + type_embedding
        x = self.emb_project(self.norm(x))

        x_result = torch.zeros((B * P, x.shape[-1]), device=x.device)
        x_result[valid_indices] = x  # Fill in valid parts
        
        return x_result.view(B, P, -1), mask_p.reshape(B, -1)

    def _mix(self, x):
        x = self.point_norm(x)
        N = x.shape[0]
        q = self.query_token.expand(N, -1, -1)
        x, _ = self.attn_pool(q, x, x)
        x = self.attn_norm(x.squeeze(1))
        return x


class LaneFusionEncoder(nn.Module):
    def __init__(self, lane_len, drop_path_rate=0.3, hidden_dim=192, depth=3, tokens_mlp_dim=64, channels_mlp_dim=128,
                 shared_spatial_enc: ChannelPreProject = None):
        super().__init__()

        self._lane_len = lane_len
        self._channel = channels_mlp_dim

        self.speed_limit_emb = FourierEmbedding(
            input_dim=1, hidden_dim=channels_mlp_dim, num_freq_bands=16
        )
        self.unknown_speed_emb = nn.Embedding(1, channels_mlp_dim)
        self.traffic_emb = nn.Embedding(9, channels_mlp_dim)
        self.type_emb = nn.Embedding(4, channels_mlp_dim)
        self.stop_point_emb = FourierEmbedding(
            input_dim=2, hidden_dim=channels_mlp_dim, num_freq_bands=32
        )

        if shared_spatial_enc is not None:
            self.channel_pre_project = shared_spatial_enc
        else:
            self.channel_pre_project = ChannelPreProject(
                input_dim=4, channels_mlp_dim=channels_mlp_dim, num_freq_bands=32
            )

        self.point_norm = nn.LayerNorm(channels_mlp_dim)
        self.query_token = nn.Parameter(torch.randn(1, 1, channels_mlp_dim) * 0.02)
        self.attn_pool = nn.MultiheadAttention(
            embed_dim=channels_mlp_dim, num_heads=4, dropout=drop_path_rate, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(channels_mlp_dim)
        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(
            in_features=channels_mlp_dim, hidden_features=hidden_dim,
            out_features=channels_mlp_dim, act_layer=nn.GELU, drop=drop_path_rate
        )

    def forward(self, x, stop_point, speed_limit, has_speed_limit):
        '''
        x: B, P, V, D (x, y, cos, sin, traffic, type)
        stop_point: B, P, 2 (stop_x, stop_y) in local lane frame
        speed_limit: B, P, 1
        has_speed_limit: B, P, 1
        '''
        B, P, V, _ = x.shape

        traffic = x[:, :, 0, 4]
        lane_type = x[:, :, 0, 5]
        x = x[..., :4]
        
        mask_v = torch.sum(torch.ne(x[..., :4], 0), dim=-1).to(x.device) == 0
        mask_p = torch.sum(~mask_v, dim=-1) == 0
        x = x.view(B * P, V, -1)

        valid_indices = ~mask_p.view(-1) 
        x = x[valid_indices]

        # === Early fusion: speed_limit and type (before attention) ===
        speed_limit = speed_limit.view(B * P, 1)
        has_speed_limit = has_speed_limit.view(B * P, 1)

        has_speed_limit = has_speed_limit[valid_indices].squeeze(-1)
        speed_limit = speed_limit[valid_indices].squeeze(-1)

        unknown_indices = torch.zeros(speed_limit.shape[0], dtype=torch.long, device=x.device)
        unknown_emb_all = self.unknown_speed_emb(unknown_indices)

        speed_limit_emb_all = self.speed_limit_emb(speed_limit.unsqueeze(-1))
        speed_limit_embedding = torch.where(
            has_speed_limit.unsqueeze(-1).bool(),
            speed_limit_emb_all,
            unknown_emb_all
        )

        lane_type = lane_type.long().view(B * P)
        lane_type = lane_type[valid_indices]
        type_embedding = self.type_emb(lane_type)

        x = self.channel_pre_project(x)
        x = x + speed_limit_embedding.unsqueeze(1) + type_embedding.unsqueeze(1)
        x = self._mix(x)

        # === Late fusion: traffic and stop_point (after attention) ===
        traffic = traffic.long().view(B * P)
        traffic = traffic[valid_indices]
        traffic_light_embedding = self.traffic_emb(traffic)

        stop_point = stop_point.view(B * P, 2)
        stop_point = stop_point[valid_indices]
        stop_point_embedding = self.stop_point_emb(stop_point)

        x = x + self.emb_project(self.norm(traffic_light_embedding + stop_point_embedding))

        x_result = torch.zeros((B * P, x.shape[-1]), device=x.device)
        x_result[valid_indices] = x  # Fill in valid parts
        
        return x_result.view(B, P, -1), mask_p.reshape(B, -1)

    def _mix(self, x):
        x = self.point_norm(x)
        N = x.shape[0]
        q = self.query_token.expand(N, -1, -1)
        x, _ = self.attn_pool(q, x, x)
        x = self.attn_norm(x.squeeze(1))
        return x


class RoadlineFusionEncoder(nn.Module):
    def __init__(self, roadline_len, drop_path_rate=0.3, hidden_dim=192, depth=3, tokens_mlp_dim=64, channels_mlp_dim=128,
                 shared_spatial_enc: ChannelPreProject = None):
        super().__init__()

        self._roadline_len = roadline_len
        self._channel = channels_mlp_dim

        self.type_emb = nn.Embedding(12, channels_mlp_dim)

        if shared_spatial_enc is not None:
            self.channel_pre_project = shared_spatial_enc
        else:
            self.channel_pre_project = ChannelPreProject(
                input_dim=4, channels_mlp_dim=channels_mlp_dim, num_freq_bands=32
            )

        self.point_norm = nn.LayerNorm(channels_mlp_dim)
        self.query_token = nn.Parameter(torch.randn(1, 1, channels_mlp_dim) * 0.02)
        self.attn_pool = nn.MultiheadAttention(
            embed_dim=channels_mlp_dim, num_heads=4, dropout=drop_path_rate, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(channels_mlp_dim)
        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(
            in_features=channels_mlp_dim, hidden_features=hidden_dim,
            out_features=hidden_dim, act_layer=nn.GELU, drop=drop_path_rate
        )

    def forward(self, x):
        '''
        x: B, P, V, D (x, y, cos, sin, type)
        '''
        B, P, V, _ = x.shape
        roadline_type = x[:, :, 0, 4]
        x = x[..., :4]

        mask_v = torch.sum(torch.ne(x[..., :4], 0), dim=-1).to(x.device) == 0
        mask_p = torch.sum(~mask_v, dim=-1) == 0
        x = x.view(B * P, V, -1)

        valid_indices = ~mask_p.view(-1) 
        x = x[valid_indices]

        x = self.channel_pre_project(x)
        x = self._mix(x)

        roadline_type = roadline_type.long().view(B * P)
        roadline_type = roadline_type[valid_indices]
        type_embedding = self.type_emb(roadline_type)
        
        x = x + type_embedding
        x = self.emb_project(self.norm(x))

        x_result = torch.zeros((B * P, x.shape[-1]), device=x.device)
        x_result[valid_indices] = x  # Fill in valid parts
        
        return x_result.view(B, P, -1), mask_p.reshape(B, -1)

    def _mix(self, x):
        x = self.point_norm(x)
        N = x.shape[0]
        q = self.query_token.expand(N, -1, -1)
        x, _ = self.attn_pool(q, x, x)
        x = self.attn_norm(x.squeeze(1))
        return x


class TrafficLightEncoder(nn.Module):
    def __init__(self, hidden_dim=192):
        super().__init__()
        self.type_embed = nn.Embedding(9, hidden_dim)

    def forward(self, inputs):
        # inputs [B, TL, 3] - (x, y, traffic_light_state)
        B, P, _ = inputs.shape

        traffic_light_type = inputs[:, :, 2].long().clamp(0, 8)
        mask = torch.eq(inputs.sum(-1), 0)

        valid_indices = ~mask.view(-1)
        traffic_light_type = traffic_light_type.view(-1)
        traffic_light_type = traffic_light_type[valid_indices]

        type_embed = self.type_embed(traffic_light_type)

        output = torch.zeros((B * P, type_embed.shape[-1]), device=inputs.device)
        output[valid_indices] = type_embed

        return output.view(B, P, -1), mask

class FusionEncoder(nn.Module):
    def __init__(self, hidden_dim=192, num_heads=6, drop_path_rate=0.3, depth=3, device='cuda', rel_encoder=None):
        super().__init__()

        dpr = drop_path_rate

        self.blocks = nn.ModuleList(
            [LocalSelfAttentionBlock(hidden_dim, num_heads, dropout=dpr) for i in range(depth)]
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.rel_encoder = rel_encoder

    def forward(self, x, mask, index_pair, relations=None):
        # mask = mask.clone()
        # mask[:, 0] = False

        rel_enc_sparse = None
        if relations is not None and self.rel_encoder is not None:
            B, N_total, _, _ = relations.shape
            N = mask.shape[1]
            L = index_pair.shape[1]
            index_pair_2d = index_pair.reshape(B, N, L)
            batch_idx = torch.arange(B, device=x.device)[:, None, None].expand(-1, N, L)
            query_idx = torch.arange(N, device=x.device)[None, :, None].expand(B, -1, L)
            rel_sparse = relations[batch_idx, query_idx, index_pair_2d.clamp(min=0), :]
            rel_enc_sparse = self.rel_encoder(rel_sparse).reshape(B * N, L, -1)

        for b in self.blocks:
            x = b(x, mask, index_pair, rel_enc_sparse)

        return self.norm(x)



class LocalCrossAttentionBlock(nn.Module):
    """
    Local cross-attention block: query tokens attend to key/value tokens via index_pair.
    query attends to a subset of key_value specified by index_pair.
    """
    def __init__(self, dim=192, heads=6, dropout=0.1, mlp_ratio=4.0, attention_mode="full"):
        super().__init__()
        self.num_heads = heads
        self.head_dim = dim // heads
        self.attn_dropout_rate = dropout
        self.attention_mode = attention_mode

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

        self.drop_path = DropPath(dropout) if dropout > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=dropout)
        self.rel_proj = nn.Linear(dim, dim)
        self.rel_v_proj = nn.Linear(dim, dim)

        self.to_g = nn.Linear(2 * dim, dim)
        self.to_s = nn.Linear(dim, dim)

        self.local_attn = MultiheadAttentionLocal(dim, heads, dropout)

    def forward(self, query, key_value, query_mask, kv_mask, index_pair, rel_enc_sparse=None):
        """
        Args:
            query: [B, N_q, D] query tokens (e.g., agents)
            key_value: [B, N_kv, D] key/value tokens (e.g., maps)
            query_mask: [B, N_q] bool, True=invalid/padded
            kv_mask: [B, N_kv] bool, True=invalid/padded
            index_pair: [B*N_q, L] local attention indices into key_value, -1 for padding
            rel_enc_sparse: [B*N_q, L, D] pre-encoded sparse relations
        """
        B, N_q, D = query.shape
        N_kv = key_value.shape[1]
        H = self.num_heads
        d = self.head_dim

        shortcut = query
        q_norm = self.norm_q(query)
        kv_norm = self.norm_kv(key_value)

        batch_cnt_q = [N_q] * B
        batch_cnt_kv = [N_kv] * B

        local_rel = None
        local_rel_v = None
        if rel_enc_sparse is not None:
            L = index_pair.shape[1]
            local_rel = self.rel_proj(rel_enc_sparse).reshape(B * N_q, L, H, d)
            if self.attention_mode == "full":
                local_rel_v = self.rel_v_proj(rel_enc_sparse).reshape(B * N_q, L, H, d)

        local_attn_mask = None
        if kv_mask is not None:
            batch_ids = torch.arange(B, device=query.device).repeat_interleave(N_q)
            local_attn_mask = kv_mask[batch_ids[:, None], index_pair.clamp(min=0)]
            local_attn_mask = local_attn_mask | (index_pair == -1)

        value, _ = self.local_attn(
            query=q_norm.reshape(B * N_q, D),
            key=kv_norm.reshape(B * N_kv, D),
            value=kv_norm.reshape(B * N_kv, D),
            index_pair=index_pair,
            query_batch_cnt=batch_cnt_q,
            key_batch_cnt=batch_cnt_kv,
            attn_mask=local_attn_mask,
            relation_encodings=local_rel,
            rel_v=local_rel_v,
        )
        value = value.reshape(B, N_q, D)

        g = torch.sigmoid(self.to_g(torch.cat([value, shortcut], dim=-1)))
        x = value + g * (self.to_s(shortcut) - value)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if query_mask is not None:
            x = x * (~query_mask).float().unsqueeze(-1)  # [B, N_q, 1]
        return x


class CrossFusionEncoder(nn.Module):
    """
    Cross-attention fusion encoder: query tokens attend to key/value tokens.
    Uses LocalCrossAttentionBlock for sparse/local cross-attention.
    """
    def __init__(self, hidden_dim=192, num_heads=6, drop_path_rate=0.3, depth=1, device='cuda', rel_encoder=None):
        super().__init__()

        dpr = drop_path_rate

        self.blocks = nn.ModuleList(
            [LocalCrossAttentionBlock(hidden_dim, num_heads, dropout=dpr) for i in range(depth)]
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.rel_encoder = rel_encoder

    def forward(self, query, key_value, query_mask, kv_mask, index_pair, relations=None, self_rel_mask=None):
        """
        Args:
            query: [B, N_q, D] query tokens (agents)
            key_value: [B, N_kv, D] key/value tokens (maps)
            query_mask: [B, N_q] bool, True=invalid
            kv_mask: [B, N_kv] bool, True=invalid
            index_pair: [B*N_q, L] indices into key_value for local attention
            relations: [B, N_q, N_kv, rel_dim] full relation matrix (agent→map)
            self_rel_mask: [B*N_q, L] bool, True=zero this relation encoding (for self-relation)
        """
        rel_enc_sparse = None
        if relations is not None and self.rel_encoder is not None:
            B, N_q_total, _, _ = relations.shape
            N_q = query_mask.shape[1]
            L = index_pair.shape[1]
            index_pair_2d = index_pair.reshape(B, N_q, L)
            batch_idx = torch.arange(B, device=query.device)[:, None, None].expand(-1, N_q, L)
            query_idx = torch.arange(N_q, device=query.device)[None, :, None].expand(B, -1, L)
            rel_sparse = relations[batch_idx, query_idx, index_pair_2d.clamp(min=0), :]
            rel_enc_sparse = self.rel_encoder(rel_sparse).reshape(B * N_q, L, -1)

        if self_rel_mask is not None and rel_enc_sparse is not None:
            rel_enc_sparse = rel_enc_sparse.masked_fill(self_rel_mask.unsqueeze(-1), 0.0)

        for b in self.blocks:
            query = b(query, key_value, query_mask, kv_mask, index_pair, rel_enc_sparse)

        return self.norm(query)