import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import knn

from src.dp.model.module.mixer import MixerBlock
from src.dp.model.module.timm import Mlp, DropPath
from src.dp.model.module.local_attention import MultiheadAttentionLocal
from src.dp.utils.train_utils import batch_transform_trajs_to_local_frame, batch_calculate_relations, batch_transform_maps_to_local_frame
from src.dp.model.module.rel_emb import RelationEncoder

class Encoder(nn.Module):
    def __init__(self, config, rel_encoder=None):
        super().__init__()

        self.hidden_dim = config.hidden_dim

        self.token_num = config.agent_num + config.static_map_num + config.lane_num + config.roadline_num + config.traffic_light_num
        self.local_attn_k = getattr(config, 'local_attn_k', 16)

        self.agents_encoder = AgentFusionEncoder(config.time_len, drop_path_rate=config.encoder_drop_path_rate, hidden_dim=config.hidden_dim, depth=config.encoder_depth)

        # Create shared backbone for static_maps, lanes and roadlines
        # This shares the position encoding matrix (channel_pre_project),
        # all MixerBlocks, and the final projection layer among the three encoders
        self.vector_map_backbone = VectorMapBackbone(
            channels_mlp_dim=128,
            tokens_mlp_dim=64,
            hidden_dim=config.hidden_dim,
            depth=config.polyline_encoder_depth,
            drop_path_rate=config.encoder_drop_path_rate
        )
        self.static_map_encoder = StaticMapFusionEncoder(
            config.static_map_len, drop_path_rate=config.encoder_drop_path_rate, hidden_dim=config.hidden_dim,
            shared_backbone=self.vector_map_backbone
        )
        self.lane_encoder = LaneFusionEncoder(
            config.lane_len,
            drop_path_rate=config.encoder_drop_path_rate,
            hidden_dim=config.hidden_dim,
            depth=config.polyline_encoder_depth,
            shared_backbone=self.vector_map_backbone
        )
        self.roadline_encoder = RoadlineFusionEncoder(
            config.roadline_len,
            drop_path_rate=config.encoder_drop_path_rate,
            hidden_dim=config.hidden_dim,
            depth=config.polyline_encoder_depth,
            shared_backbone=self.vector_map_backbone
        )

        self.traffic_light_encoder = TrafficLightEncoder(config.hidden_dim)

        self.rel_encoder = rel_encoder if rel_encoder is not None else RelationEncoder(
            hidden_dim=config.hidden_dim,
            num_freq_bands=64
        )

        self.fusion = FusionEncoder(
            hidden_dim=config.hidden_dim, 
            num_heads=config.num_heads, 
            drop_path_rate=config.encoder_drop_path_rate, 
            depth=config.encoder_depth, 
            device=config.device,
            rel_encoder=self.rel_encoder,
        )

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

        agents_pos = agents[:, :, -1, :8].clone()
        agents_pos[..., -4:] = 0.0
        agents_pos[..., -4] = 1.0

        agents_local = batch_transform_trajs_to_local_frame(agents)

        encoding_agents, agents_mask, _ = self.agents_encoder(agents_local, agents_type)
        static_map_len = self.static_map_encoder._map_len
        lane_len = self.lane_encoder._lane_len
        roadline_len = self.roadline_encoder._roadline_len

        static_spatial_pos = static_maps[:, :, int(static_map_len / 2), :4].clone()
        lane_spatial_pos = lanes[:, :, int(lane_len / 2), :4].clone()
        roadline_spatial_pos = roadlines[:, :, int(roadline_len / 2), :4].clone()

        static_maps_local = batch_transform_maps_to_local_frame(static_maps)
        lanes_local = batch_transform_maps_to_local_frame(lanes)
        roadlines_local = batch_transform_maps_to_local_frame(roadlines)

        encoding_static, static_mask, static_pos = self.static_map_encoder(static_maps_local)
        encoding_lanes, lanes_mask, lane_pos = self.lane_encoder(lanes_local, lanes_speed_limit, lanes_has_speed_limit)
        encoding_roadlines, roadlines_mask, roadline_pos = self.roadline_encoder(roadlines_local)

        # traffic lights
        traffic_lights = inputs['traffic_light_points']
        encoding_traffic_lights, traffic_lights_mask = self.traffic_light_encoder(traffic_lights)

        static_pos[..., :4] = static_spatial_pos
        lane_pos[..., :4] = lane_spatial_pos
        roadline_pos[..., :4] = roadline_spatial_pos

        encoding_input = torch.cat([encoding_agents, encoding_static, encoding_lanes, encoding_roadlines, encoding_traffic_lights], dim=1)

        # encoding_pos = torch.cat([agents_pos, static_pos, lane_pos, roadline_pos], dim=1).view(B * self.token_num, -1)
        encoding_mask = torch.cat([agents_mask, static_mask, lanes_mask, roadlines_mask, traffic_lights_mask], dim=1).view(-1)
        # encoding_pos = self.pos_emb(encoding_pos[~encoding_mask])
        # encoding_pos_result = torch.zeros((B * self.token_num, self.hidden_dim), device=encoding_pos.device)
        # encoding_pos_result[~encoding_mask] = encoding_pos  # Fill in valid parts

        # encoding_input = encoding_input + encoding_pos_result.view(B, self.token_num, -1)

        # Compute and encode relations
        relations, all_elements = batch_calculate_relations(
            agents, static_maps, lanes, roadlines, traffic_lights,
            encoding_mask=encoding_mask.reshape(B, self.token_num),
            device=agents.device
        )
        # relations_pos = relations[..., :2]
        # relations_angle = relations[..., 2:]
        # encoded_pos = self.relation_pos_encoder(relations_pos)
        # encoded_angle = self.relation_angle_encoder(relations_angle)
        encoder_outputs['relations'] = relations

        N = self.token_num
        dist = torch.norm(relations[..., :2], dim=-1)
        dist = dist + torch.eye(N, device=dist.device).unsqueeze(0) * 1e9
        mask_2d = encoding_mask.view(B, N)
        dist = dist.masked_fill(mask_2d.unsqueeze(1), 1e9)
        _, index_pair = torch.topk(dist, k=self.local_attn_k, dim=-1, largest=False)
        index_pair = index_pair.reshape(B * N, self.local_attn_k)

        # # ---- DEBUG: 保存 index_pair 数据到文件，确认后删除 ----
        # debug_index_pair(index_pair, all_elements, encoding_mask, B, N, self.local_attn_k,
        #                  n_agents=64, n_static=20,
        #                  n_lanes=256, n_roadlines=128,
        #                  n_traffic_lights=20)
        # # ---- DEBUG END ----

        encoder_outputs['encoding'] = self.fusion(encoding_input, encoding_mask.view(B, N), index_pair, relations)
        encoder_outputs['encoding_mask'] = encoding_mask.view(B, N)

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

class VectorMapBackbone(nn.Module):
    """
    Shared backbone for vector map encoders (lanes, roadlines, static_maps).
    Shares the spatial position encoding (channel_pre_project), MixerBlocks,
    normalization, and final projection between lane and roadline encoders.

    NOTE: token_pre_project is NOT shared because lane and roadline have
    different sequence lengths (lane_len vs roadline_len).
    """
    def __init__(self, channels_mlp_dim=128, tokens_mlp_dim=64, hidden_dim=192,
                 depth=3, drop_path_rate=0.3):
        super().__init__()

        self.channel_pre_project = Mlp(
            in_features=4, hidden_features=channels_mlp_dim,
            out_features=channels_mlp_dim, act_layer=nn.GELU, drop=0.
        )
        self.blocks = nn.ModuleList([
            MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(
            in_features=channels_mlp_dim, hidden_features=hidden_dim,
            out_features=hidden_dim, act_layer=nn.GELU, drop=drop_path_rate
        )

        self.global_fusion = nn.Sequential(
            nn.Linear(channels_mlp_dim * 2, channels_mlp_dim),
            nn.LayerNorm(channels_mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(channels_mlp_dim, channels_mlp_dim),
        )

    def mix(self, x):
        """
        MixerBlocks + mean pooling.
        Args:
            x: [N, V, channels_mlp_dim]  after channel_pre_project + token_pre_project
        Returns:
            x: [N, channels_mlp_dim]
        """
        for block in self.blocks:
            x = block(x)
        
        global_feat = torch.max(x, dim=1, keepdim=True)[0]
        x = torch.cat([x, global_feat.expand(-1, x.shape[1], -1)], dim=-1)
        x = self.global_fusion(x)
        x = torch.max(x, dim=1)[0]
        return x

    def proj(self, x):
        """
        LayerNorm + final projection.
        Args:
            x: [N, channels_mlp_dim]
        Returns:
            x: [N, hidden_dim]
        """
        return self.emb_project(self.norm(x))


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim=192, heads=6, dropout=0.1, mlp_ratio=4.0, attention_mode="part"):
        super().__init__()

        self.num_heads = heads
        self.head_dim = dim // heads
        self.attn_dropout_rate = dropout
        self.attention_mode = attention_mode
        self.norm1 = nn.LayerNorm(dim)

        self.in_proj = nn.Linear(dim, 3 * dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)

        self.drop_path = DropPath(dropout) if dropout > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=dropout)
        self.rel_proj = nn.Linear(dim, dim)
        self.rel_v_proj = nn.Linear(dim, dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.in_proj.bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, x, mask, relation_encodings=None):
        B, N, D = x.shape
        H = self.num_heads
        d = self.head_dim

        shortcut = x
        x = self.norm1(x)

        qkv = self.in_proj(x).reshape(B, N, H, 3 * d)
        q, k, v = torch.split(qkv, d, dim=-1)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn_mask = None
        rel_pos_v = None
        if relation_encodings is not None:
            attn_mask = torch.zeros(B, H, N, N, device=x.device, dtype=x.dtype)
            
            rel_enc = self.rel_proj(relation_encodings).reshape(B, N, N, H, d)
            rel_pos_q = rel_enc.permute(0, 3, 1, 4, 2)
            dot_score_rel = torch.matmul(q.unsqueeze(-2), rel_pos_q).squeeze(-2)
            attn_mask = attn_mask + dot_score_rel
            if self.attention_mode == "full":
                rel_pos_v = self.rel_v_proj(relation_encodings).reshape(B, N, N, H, d)
                rel_pos_v = rel_pos_v.permute(0, 3, 1, 2, 4)
            if mask is not None:
                attn_mask = attn_mask.masked_fill(mask[:, None, None, :], float('-inf'))

        if rel_pos_v is not None:
            score = (q @ k.transpose(-2, -1)) * (d ** -0.5)
            score = score + attn_mask
            attn = F.softmax(score, dim=-1)
            attn = F.dropout(attn, p=self.attn_dropout_rate, training=self.training)
            value = attn @ v
            value = value + torch.matmul(attn.unsqueeze(-2), rel_pos_v).squeeze(-2)
        else:
            value = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_dropout_rate if self.training else 0.0,
            )

        value = value.permute(0, 2, 1, 3).reshape(B, N, D)
        value = self.out_proj(value)

        x = shortcut + self.drop_path(value)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


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

        self.in_proj = nn.Linear(dim, 3 * dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)

        self.drop_path = DropPath(dropout) if dropout > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=dropout)
        self.rel_proj = nn.Linear(dim, dim)
        self.rel_v_proj = nn.Linear(dim, dim)

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

        qkv = self.in_proj(x_norm).reshape(B, N, H, 3 * d)
        q, k, v = torch.split(qkv, d, dim=-1)

        q = q.reshape(B * N, H, d)
        k = k.reshape(B * N, H, d)
        v = v.reshape(B * N, H, d)

        batch_cnt = [N] * B

        local_rel = None
        local_rel_v = None
        if rel_enc_sparse is not None:
            local_rel = self.rel_proj(rel_enc_sparse).reshape(B * N, L, H, d)
            if self.attention_mode == "full":
                local_rel_v = self.rel_v_proj(rel_enc_sparse).reshape(B * N, L, H, d)

        local_attn_mask = None
        if mask is not None:
            batch_ids = torch.arange(B, device=x.device).repeat_interleave(N)
            local_attn_mask = mask[batch_ids[:, None], index_pair.clamp(min=0)]
            local_attn_mask = local_attn_mask | (index_pair == -1)

        value, _ = self.local_attn(
            query=x.reshape(B * N, D),
            key=x.reshape(B * N, D),
            value=x.reshape(B * N, D),
            index_pair=index_pair,
            query_batch_cnt=batch_cnt,
            key_batch_cnt=batch_cnt,
            attn_mask=local_attn_mask,
            relation_encodings=local_rel,
            rel_v=local_rel_v,
            q_proj=q, k_proj=k, v_proj=v,
            skip_out_proj=True,
        )
        value = value.reshape(B, N, D)
        value = self.out_proj(value)

        x = shortcut + self.drop_path(value)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
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
        # TODO: improve time encoding
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


class StaticMapFusionEncoder(nn.Module):
    def __init__(self, static_map_len, drop_path_rate=0.3, depth=3, tokens_mlp_dim=64, hidden_dim=192, channels_mlp_dim=128, device='cuda',
                 shared_backbone: VectorMapBackbone = None):
        super().__init__()

        self._hidden_dim = hidden_dim

        self._map_len = static_map_len
        self._channel = channels_mlp_dim

        self.type_emb = nn.Embedding(3, channels_mlp_dim)

        self.token_pre_project = Mlp(
            in_features=static_map_len, hidden_features=tokens_mlp_dim,
            out_features=tokens_mlp_dim, act_layer=nn.GELU, drop=0.
        )

        if shared_backbone is not None:
            self.backbone = shared_backbone
        else:
            self.backbone = VectorMapBackbone(
                channels_mlp_dim, tokens_mlp_dim, hidden_dim, depth, drop_path_rate
            )

    def forward(self, x):
        '''
        x: B, P, V, D (x, y, cos, sin, type)
        '''
        B, P, V, _ = x.shape
        static_map_type = x[:, :, 0, 4]
        x = x[..., :4]

        spatial_pos = x[:, :, int(self._map_len / 2), :4].clone()
        
        pos = torch.zeros(B, P, 8, device=x.device)
        pos[..., :4] = spatial_pos
        # static: [0,1,0,0]
        pos[..., 5] = 1.0

        mask_v = torch.sum(torch.ne(x[..., :4], 0), dim=-1).to(x.device) == 0
        mask_p = torch.sum(~mask_v, dim=-1) == 0
        x = x.view(B * P, V, -1)

        valid_indices = ~mask_p.view(-1) 
        x = x[valid_indices] 

        x = self.backbone.channel_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.token_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.backbone.mix(x)

        static_map_type = static_map_type.long().view(B * P)
        static_map_type = static_map_type[valid_indices]
        type_embedding = self.type_emb(static_map_type)
        
        x = x + type_embedding
        x = self.backbone.proj(x)

        x_result = torch.zeros((B * P, x.shape[-1]), device=x.device)
        x_result[valid_indices] = x  # Fill in valid parts
        
        return x_result.view(B, P, -1) , mask_p.reshape(B, -1), pos.view(B, P, -1)


class LaneFusionEncoder(nn.Module):
    def __init__(self, lane_len, drop_path_rate=0.3, hidden_dim=192, depth=3, tokens_mlp_dim=64, channels_mlp_dim=128,
                 shared_backbone: VectorMapBackbone = None):
        super().__init__()

        self._lane_len = lane_len
        self._channel = channels_mlp_dim

        self.speed_limit_emb = nn.Linear(1, channels_mlp_dim)
        self.unknown_speed_emb = nn.Embedding(1, channels_mlp_dim)
        self.traffic_emb = nn.Embedding(9, channels_mlp_dim)
        self.type_emb = nn.Embedding(4, channels_mlp_dim)

        self.token_pre_project = Mlp(
            in_features=lane_len, hidden_features=tokens_mlp_dim,
            out_features=tokens_mlp_dim, act_layer=nn.GELU, drop=0.
        )

        if shared_backbone is not None:
            self.backbone = shared_backbone
        else:
            self.backbone = VectorMapBackbone(
                channels_mlp_dim, tokens_mlp_dim, hidden_dim, depth, drop_path_rate
            )

    def forward(self, x, speed_limit, has_speed_limit):
        '''
        x: B, P, V, D (x, y, cos, sin, traffic, type)
        speed_limit: B, P, 1
        has_speed_limit: B, P, 1
        '''
        B, P, V, _ = x.shape

        traffic = x[:, :, 0, 4]
        lane_type = x[:, :, 0, 5]
        x = x[..., :4]
        
        spatial_pos = x[:, :, int(self._lane_len / 2), :4].clone()
        
        pos = torch.zeros(B, P, 8, device=x.device)
        pos[..., :4] = spatial_pos
        # lane: [0,0,1,0]
        pos[..., 6] = 1.0

        mask_v = torch.sum(torch.ne(x[..., :4], 0), dim=-1).to(x.device) == 0
        mask_p = torch.sum(~mask_v, dim=-1) == 0
        x = x.view(B * P, V, -1)

        valid_indices = ~mask_p.view(-1) 
        x = x[valid_indices] 

        x = self.backbone.channel_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.token_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.backbone.mix(x)

        # Reshape speed_limit and traffic to match flattened dimensions
        speed_limit = speed_limit.view(B * P, 1)
        has_speed_limit = has_speed_limit.view(B * P, 1)
        traffic = traffic.long().view(B * P)

        # Apply embedding directly to valid speed limit data
        has_speed_limit = has_speed_limit[valid_indices].squeeze(-1)
        speed_limit = speed_limit[valid_indices].squeeze(-1)
        # speed_limit_embedding = torch.zeros((speed_limit.shape[0], self._channel), device=x.device)

        # if has_speed_limit.sum() > 0:
        #     speed_limit_with_limit = self.speed_limit_emb(speed_limit[has_speed_limit].unsqueeze(-1))
        #     speed_limit_embedding[has_speed_limit] = speed_limit_with_limit

        # if (~has_speed_limit).sum() > 0:
        #     speed_limit_no_limit = self.unknown_speed_emb.weight.expand(
        #         (~has_speed_limit).sum().item(), -1
        #     )
        #     speed_limit_embedding[~has_speed_limit] = speed_limit_no_limit
        
        unknown_indices = torch.zeros(speed_limit.shape[0], dtype=torch.long, device=x.device)
        unknown_emb_all = self.unknown_speed_emb(unknown_indices)  # [B*P, C]

        speed_limit_emb_all = self.speed_limit_emb(speed_limit.unsqueeze(-1))  # [V, C]

        speed_limit_embedding = torch.where(
            has_speed_limit.unsqueeze(-1).bool(),   # [V, 1] -> broadcast to [V, C]
            speed_limit_emb_all,
            unknown_emb_all
        )

        # Process traffic lights directly for valid positions
        traffic = traffic[valid_indices]
        traffic_light_embedding = self.traffic_emb(traffic)  # Traffic light embedding for valid data

        lane_type = lane_type.long().view(B * P)
        lane_type = lane_type[valid_indices]
        type_embedding = self.type_emb(lane_type)

        x = x + speed_limit_embedding + traffic_light_embedding + type_embedding
        x = self.backbone.proj(x)

        x_result = torch.zeros((B * P, x.shape[-1]), device=x.device)
        x_result[valid_indices] = x  # Fill in valid parts
        
        return x_result.view(B, P, -1) , mask_p.reshape(B, -1), pos.view(B, P, -1)

class RoadlineFusionEncoder(nn.Module):
    def __init__(self, roadline_len, drop_path_rate=0.3, hidden_dim=192, depth=3, tokens_mlp_dim=64, channels_mlp_dim=128,
                 shared_backbone: VectorMapBackbone = None):
        super().__init__()

        self._roadline_len = roadline_len
        self._channel = channels_mlp_dim

        self.type_emb = nn.Embedding(12, channels_mlp_dim)

        self.token_pre_project = Mlp(
            in_features=roadline_len, hidden_features=tokens_mlp_dim,
            out_features=tokens_mlp_dim, act_layer=nn.GELU, drop=0.
        )

        if shared_backbone is not None:
            self.backbone = shared_backbone
        else:
            self.backbone = VectorMapBackbone(
                channels_mlp_dim, tokens_mlp_dim, hidden_dim, depth, drop_path_rate
            )

    def forward(self, x):
        '''
        x: B, P, V, D (x, y, cos, sin, type)
        '''
        B, P, V, _ = x.shape
        roadline_type = x[:, :, 0, 4]
        x = x[..., :4]

        spatial_pos = x[:, :, int(self._roadline_len / 2), :4].clone()
        
        pos = torch.zeros(B, P, 8, device=x.device)
        pos[..., :4] = spatial_pos
        # roadline: [0,0,0,1]
        pos[..., 7] = 1.0

        mask_v = torch.sum(torch.ne(x[..., :4], 0), dim=-1).to(x.device) == 0
        mask_p = torch.sum(~mask_v, dim=-1) == 0
        x = x.view(B * P, V, -1)

        valid_indices = ~mask_p.view(-1) 
        x = x[valid_indices]

        x = self.backbone.channel_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.token_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.backbone.mix(x)

        roadline_type = roadline_type.long().view(B * P)
        roadline_type = roadline_type[valid_indices]
        type_embedding = self.type_emb(roadline_type)
        
        x = x + type_embedding
        x = self.backbone.proj(x)

        x_result = torch.zeros((B * P, x.shape[-1]), device=x.device)
        x_result[valid_indices] = x  # Fill in valid parts
        
        return x_result.view(B, P, -1) , mask_p.reshape(B, -1), pos.view(B, P, -1)

class TrafficLightEncoder(nn.Module):
    def __init__(self, hidden_dim=192):
        super().__init__()
        self.type_embed = nn.Embedding(9, hidden_dim)

    def forward(self, inputs):
        # inputs [B, TL, 3] - (x, y, traffic_light_state)
        traffic_light_type = inputs[:, :, 2].long().clamp(0, 8)
        type_embed = self.type_embed(traffic_light_type)
        output = type_embed

        # Generate mask: True means invalid/padded
        mask = torch.eq(inputs.sum(-1), 0)

        return output, mask

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
        mask = mask.clone()
        mask[:, 0] = False

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

        # TODO: split agent and map features
        for b in self.blocks:
            x = b(x, mask, index_pair, rel_enc_sparse)

        return self.norm(x)


class FourierEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_freq_bands=8):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.freqs = nn.Embedding(input_dim, num_freq_bands) if input_dim != 0 else None

        self.mlps = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(num_freq_bands * 2 + 1, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
            ) for _ in range(input_dim)])

        self.to_out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, continuous_inputs):
        B, N1, N2, D = continuous_inputs.shape
        flat_input = continuous_inputs.view(B * N1 * N2, D)

        x = flat_input.unsqueeze(-1) * self.freqs.weight * 2 * math.pi
        x = torch.cat([x.cos(), x.sin(), flat_input.unsqueeze(-1)], dim=-1)
        x = torch.stack([self.mlps[i](x[:, i]) for i in range(self.input_dim)]).sum(dim=0)

        x = self.to_out(x)
        return x.view(B, N1, N2, self.hidden_dim)