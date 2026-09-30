"""Packed continuous SMART-style map encoder for the diffusion planner."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.dp.model.module.local_attention import MultiheadAttentionLocal
from src.dp.model.module.rel_emb import FourierEmbedding, RelationEncoder
from src.dp.model.module.smart_attention import (
    AttentionLayer,
    build_batched_knn_edge_index,
    build_batched_map_radius_graph,
    build_batched_radius_edge_index,
    gather_smart_anchor_edge_relations,
)
from src.dp.utils.train_utils import (
    batch_transform_trajs_to_local_frame,
    packed_map_to_dense,
)
from src.dp.model.module.timm import Mlp, DropPath


# Keep in sync with the unified map-type vocabulary in data_preprocess.py.
NUM_MAP_TYPES = 21
LANE_MAP_TYPE_MIN = 1
LANE_MAP_TYPE_MAX = 4
NUM_MAP_LIGHT_TYPES = 9


def _apply_basic_init(module, fn, skip_types):
    if isinstance(module, skip_types):
        return
    fn(module)
    for child in module.children():
        _apply_basic_init(child, fn, skip_types)


class AgentEncoder(nn.Module):
    """Causal history encoder retained from the original planner encoder."""

    def __init__(self, time_len, drop_path_rate=0.3, hidden_dim=192,
                 depth_self=2, depth_cross=1, num_heads=6):
        super().__init__()
        self._hidden_dim = hidden_dim
        self._time_len = time_len
        self._num_heads = num_heads
        self.type_emb = nn.Embedding(4, hidden_dim)
        self.state_proj = FourierEmbedding(
            input_dim=8, hidden_dim=hidden_dim, num_freq_bands=32
        )
        self.input_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.sa_layers = nn.ModuleList([
            MultiheadAttentionLocal(
                embed_dim=hidden_dim, num_heads=num_heads, dropout=drop_path_rate
            ) for _ in range(depth_self)
        ])
        self.a2h_rel_encoder = nn.Sequential(
            FourierEmbedding(input_dim=4, hidden_dim=hidden_dim, num_freq_bands=32),
            nn.LayerNorm(hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.rel_attn_proj = nn.Linear(hidden_dim, hidden_dim)
        self.rel_v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.sa_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(depth_self)])
        self.sa_ffn_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(depth_self)])
        self.sa_ffn_layers = nn.ModuleList([
            Mlp(
                in_features=hidden_dim, hidden_features=hidden_dim * 4,
                out_features=hidden_dim, act_layer=nn.GELU, drop=drop_path_rate,
            ) for _ in range(depth_self)
        ])
        self.emb_project = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU()
        )

    def forward(self, x, agent_type):
        batch_size, num_agents, history_len, _ = x.shape
        temporal_mask = torch.sum(torch.ne(x[..., :8], 0), dim=-1) == 0
        agent_mask = torch.sum(~temporal_mask, dim=-1) == 0

        x = x.view(batch_size * num_agents, history_len, -1)
        temporal_mask_flat = temporal_mask.view(batch_size * num_agents, history_len)
        valid_indices = ~agent_mask.view(-1)
        x = x[valid_indices]
        valid_temporal_mask = temporal_mask_flat[valid_indices]
        x_raw = x
        valid_agent_count = x.shape[0]

        motion = x_raw[:, 1:, :2] - x_raw[:, :-1, :2]
        motion = torch.cat([x_raw[:, :1, 4:6], motion], dim=1)
        velocity = x_raw[..., 4:6]
        heading_vector = x_raw[..., 2:4]
        motion_norm = torch.norm(motion, p=2, dim=-1)
        motion_heading = torch.atan2(
            heading_vector[..., 0] * motion[..., 1] - heading_vector[..., 1] * motion[..., 0],
            (heading_vector * motion).sum(-1),
        )
        velocity_norm = torch.norm(velocity, p=2, dim=-1)
        velocity_heading = torch.atan2(
            heading_vector[..., 0] * velocity[..., 1] - heading_vector[..., 1] * velocity[..., 0],
            (heading_vector * velocity).sum(-1),
        )
        heading = torch.atan2(x_raw[..., 3], x_raw[..., 2])
        state = self.state_proj(torch.stack([
            motion_norm, motion_heading, velocity_norm, velocity_heading,
            heading, x_raw[..., 6], x_raw[..., 7], x_raw[..., 8],
        ], dim=-1))
        x = self.input_proj(state)
        valid_types = agent_type.long().view(batch_size * num_agents)[valid_indices]
        x = self.emb_project(x + self.type_emb(valid_types).unsqueeze(1))

        position = x_raw[..., :2]
        heading = torch.atan2(x_raw[..., 3], x_raw[..., 2])
        relative_position = position.unsqueeze(1) - position.unsqueeze(2)
        relative_heading = heading.unsqueeze(1) - heading.unsqueeze(2)
        relative_distance = torch.norm(relative_position, p=2, dim=-1)
        relative_distance = torch.where(
            relative_distance < 1e-4, torch.zeros_like(relative_distance), relative_distance
        )
        relative_bearing = torch.atan2(relative_position[..., 1], relative_position[..., 0])
        relative_bearing = torch.where(
            relative_distance < 1e-4, torch.zeros_like(relative_bearing), relative_bearing
        )
        timestep = torch.arange(history_len, device=x.device, dtype=torch.float32)
        relative_time = (timestep.unsqueeze(0) - timestep.unsqueeze(1)).unsqueeze(0) / history_len
        temporal_relations = torch.stack([
            relative_distance, relative_bearing, relative_heading,
            relative_time.expand(valid_agent_count, -1, -1),
        ], dim=-1)
        temporal_relation_features = self.a2h_rel_encoder(temporal_relations)
        relation_attention = self.rel_attn_proj(temporal_relation_features).reshape(
            valid_agent_count * history_len, history_len, self._num_heads, -1
        )
        relation_values = self.rel_v_proj(temporal_relation_features).reshape(
            valid_agent_count * history_len, history_len, self._num_heads, -1
        )

        arange = torch.arange(history_len, device=x.device)
        index_pair = arange.unsqueeze(0).expand(history_len, -1).clone()
        index_pair = index_pair.unsqueeze(0).expand(
            valid_agent_count, -1, -1
        ).reshape(valid_agent_count * history_len, history_len)
        causal_mask = arange.unsqueeze(1) < arange.unsqueeze(0)
        causal_mask = causal_mask.unsqueeze(0).expand(
            valid_agent_count, -1, -1
        ).reshape(valid_agent_count * history_len, history_len)
        key_padding_mask = valid_temporal_mask.unsqueeze(1).expand(
            valid_agent_count, history_len, history_len
        ).reshape(valid_agent_count * history_len, history_len)
        attention_mask = causal_mask | key_padding_mask
        self_mask = torch.eye(
            history_len, device=x.device, dtype=torch.bool
        ).unsqueeze(0).expand(valid_agent_count, -1, -1).reshape(
            valid_agent_count * history_len, history_len
        )
        attention_mask = attention_mask & ~self_mask

        counts = [history_len] * valid_agent_count
        for index, attention in enumerate(self.sa_layers):
            residual = x
            x_norm = self.sa_norms[index](x).reshape(
                valid_agent_count * history_len, self._hidden_dim
            )
            x_out, _ = attention(
                query=x_norm, key=x_norm, value=x_norm,
                index_pair=index_pair, query_batch_cnt=counts, key_batch_cnt=counts,
                relation_encodings=relation_attention, rel_v=relation_values,
                attn_mask=attention_mask,
            )
            x = x_out.reshape(valid_agent_count, history_len, self._hidden_dim) + residual
            x = self.sa_ffn_layers[index](self.sa_ffn_norms[index](x)) + x

        reference_x = x_raw[:, -1, 0].unsqueeze(-1)
        reference_y = x_raw[:, -1, 1].unsqueeze(-1)
        reference_cos = x_raw[:, -1, 2].unsqueeze(-1)
        reference_sin = x_raw[:, -1, 3].unsqueeze(-1)
        dx = reference_x - x_raw[..., 0]
        dy = reference_y - x_raw[..., 1]
        local_x = dx * reference_cos + dy * reference_sin
        local_y = -dx * reference_sin + dy * reference_cos
        relative_cos = reference_cos * x_raw[..., 2] + reference_sin * x_raw[..., 3]
        relative_sin = reference_sin * x_raw[..., 2] - reference_cos * x_raw[..., 3]
        current_to_history = torch.stack(
            [local_x, local_y, relative_cos, relative_sin], dim=-1
        )

        temporal_output = x.new_zeros(
            (batch_size * num_agents, history_len, self._hidden_dim)
        )
        temporal_output[valid_indices] = x
        current_output = x.new_zeros((batch_size * num_agents, self._hidden_dim))
        current_output[valid_indices] = x[:, -1]
        output_temporal_mask = torch.ones(
            (batch_size * num_agents, history_len), dtype=torch.bool, device=x.device
        )
        output_temporal_mask[valid_indices] = valid_temporal_mask
        relation_output = x.new_zeros((batch_size * num_agents, history_len, 4))
        relation_output[valid_indices] = current_to_history
        return (
            temporal_output.view(batch_size, num_agents, history_len, -1),
            current_output.view(batch_size, num_agents, -1),
            output_temporal_mask.view(batch_size, num_agents, history_len),
            relation_output.view(batch_size, num_agents, history_len, 4),
            agent_mask.reshape(batch_size, num_agents),
        )


class LocalCrossAttentionBlock(nn.Module):
    def __init__(self, dim=192, heads=6, dropout=0.1, mlp_ratio=4.0,
                 attention_mode="full"):
        super().__init__()
        self.num_heads = heads
        self.head_dim = dim // heads
        self.attention_mode = attention_mode
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.drop_path = DropPath(dropout) if dropout > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(
            in_features=dim, hidden_features=int(dim * mlp_ratio),
            act_layer=nn.GELU, drop=dropout,
        )
        self.rel_proj = nn.Linear(dim, dim)
        self.rel_v_proj = nn.Linear(dim, dim)
        self.to_g = nn.Linear(2 * dim, dim)
        self.to_s = nn.Linear(dim, dim)
        self.local_attn = MultiheadAttentionLocal(dim, heads, dropout)

    def forward(self, query, key_value, query_mask, kv_mask, index_pair,
                rel_enc_sparse=None):
        batch_size, num_queries, hidden_dim = query.shape
        num_keys = key_value.shape[1]
        shortcut = query
        query_norm = self.norm_q(query)
        key_value_norm = self.norm_kv(key_value)
        local_rel = local_rel_values = None
        if rel_enc_sparse is not None:
            neighbor_count = index_pair.shape[1]
            local_rel = self.rel_proj(rel_enc_sparse).reshape(
                batch_size * num_queries, neighbor_count,
                self.num_heads, self.head_dim,
            )
            if self.attention_mode == "full":
                local_rel_values = self.rel_v_proj(rel_enc_sparse).reshape(
                    batch_size * num_queries, neighbor_count,
                    self.num_heads, self.head_dim,
                )
        local_mask = None
        if kv_mask is not None:
            batch_ids = torch.arange(
                batch_size, device=query.device
            ).repeat_interleave(num_queries)
            local_mask = kv_mask[batch_ids[:, None], index_pair.clamp(min=0)]
            local_mask = local_mask | (index_pair == -1)
        value, _ = self.local_attn(
            query=query_norm.reshape(batch_size * num_queries, hidden_dim),
            key=key_value_norm.reshape(batch_size * num_keys, hidden_dim),
            value=key_value_norm.reshape(batch_size * num_keys, hidden_dim),
            index_pair=index_pair,
            query_batch_cnt=[num_queries] * batch_size,
            key_batch_cnt=[num_keys] * batch_size,
            attn_mask=local_mask,
            relation_encodings=local_rel,
            rel_v=local_rel_values,
        )
        value = value.reshape(batch_size, num_queries, hidden_dim)
        gate = torch.sigmoid(self.to_g(torch.cat([value, shortcut], dim=-1)))
        output = value + gate * (self.to_s(shortcut) - value)
        output = output + self.drop_path(self.mlp(self.norm2(output)))
        if query_mask is not None:
            output = output * (~query_mask).float().unsqueeze(-1)
        return output


class CrossFusionEncoder(nn.Module):
    def __init__(self, hidden_dim=192, num_heads=6, drop_path_rate=0.3,
                 depth=1, device="cuda", rel_encoder=None):
        super().__init__()
        self.blocks = nn.ModuleList([
            LocalCrossAttentionBlock(
                hidden_dim, num_heads, dropout=drop_path_rate
            ) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.rel_encoder = rel_encoder

    def forward(self, query, key_value, query_mask, kv_mask, index_pair,
                relations=None, self_rel_mask=None):
        relation_features = None
        if relations is not None and self.rel_encoder is not None:
            batch_size = relations.shape[0]
            num_queries = query_mask.shape[1]
            neighbor_count = index_pair.shape[1]
            local_indices = index_pair.reshape(
                batch_size, num_queries, neighbor_count
            )
            batch_indices = torch.arange(
                batch_size, device=query.device
            )[:, None, None].expand(-1, num_queries, neighbor_count)
            query_indices = torch.arange(
                num_queries, device=query.device
            )[None, :, None].expand(batch_size, -1, neighbor_count)
            local_relations = relations[
                batch_indices, query_indices, local_indices.clamp(min=0)
            ]
            relation_features = self.rel_encoder(local_relations).reshape(
                batch_size * num_queries, neighbor_count, -1
            )
        if self_rel_mask is not None and relation_features is not None:
            relation_features = relation_features.masked_fill(
                self_rel_mask.unsqueeze(-1), 0.0
            )
        for block in self.blocks:
            query = block(
                query, key_value, query_mask, kv_mask, index_pair,
                relation_features,
            )
        return self.norm(query)


def _edge_relations(
    source_states: torch.Tensor,
    target_states: torch.Tensor,
    source_batch: torch.Tensor,
    target_batch: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    return gather_smart_anchor_edge_relations(
        source_states[:, None, :], target_states,
        source_batch, target_batch, edge_index,
    ).squeeze(1)


class SMARTMapEncoder(nn.Module):
    """Encode three-point fragments and refine them with sparse radius attention."""

    def __init__(self, config, rel_encoder: RelationEncoder | None = None):
        super().__init__()
        hidden_dim = config.hidden_dim
        num_heads = config.num_heads
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.radius_m = float(getattr(config, "pl2pl_radius_m", 10.0))
        self.max_neighbors = int(getattr(config, "max_map_neighbors", 20))
        self.num_layers = int(getattr(config, "num_map_layers", 3))
        self.segment_length_m = float(getattr(config, "map_segment_length_m", 5.0))
        self.segment_points = int(getattr(config, "map_segment_points", 3))
        if self.segment_length_m <= 0 or self.segment_points != 3:
            raise ValueError("SMART map tokens require a positive segment length and exactly 3 points")

        self.geometry_mlp = nn.Sequential(
            nn.Linear(self.segment_points * 3, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.type_embedding = nn.Embedding(NUM_MAP_TYPES, hidden_dim)
        self.light_embedding = nn.Embedding(NUM_MAP_LIGHT_TYPES, hidden_dim)
        self.stop_point_embedding = FourierEmbedding(2, hidden_dim=hidden_dim, num_freq_bands=32)
        self.unknown_stop_point_embedding = nn.Parameter(torch.empty(hidden_dim))
        nn.init.normal_(self.unknown_stop_point_embedding, std=0.02)
        self.speed_embedding = FourierEmbedding(1, hidden_dim=hidden_dim, num_freq_bands=32)
        self.unknown_speed_embedding = nn.Parameter(torch.empty(hidden_dim))
        nn.init.normal_(self.unknown_speed_embedding, std=0.02)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.rel_encoder = rel_encoder or RelationEncoder(hidden_dim=hidden_dim, num_freq_bands=64)
        self.layers = nn.ModuleList([
            AttentionLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                head_dim=hidden_dim // num_heads,
                dropout=config.encoder_drop_path_rate,
                bipartite=False,
                has_pos_emb=True,
            ) for _ in range(self.num_layers)
        ])

    @staticmethod
    def local_geometry(geometry: torch.Tensor) -> torch.Tensor:
        if geometry.shape[0] == 0:
            return geometry.new_empty((0, geometry.shape[1] * 3))
        reference = geometry[:, 0]
        delta = geometry[..., :2] - reference[:, None, :2]
        cos_h, sin_h = torch.cos(reference[:, 2]), torch.sin(reference[:, 2])
        local_x = delta[..., 0] * cos_h[:, None] + delta[..., 1] * sin_h[:, None]
        local_y = -delta[..., 0] * sin_h[:, None] + delta[..., 1] * cos_h[:, None]
        heading_delta = geometry[..., 2] - reference[:, None, 2]
        local_heading = torch.atan2(torch.sin(heading_delta), torch.cos(heading_delta))
        local = torch.stack([local_x, local_y, local_heading], dim=-1)
        return local.flatten(start_dim=1)

    @staticmethod
    def local_stop_xy(geometry: torch.Tensor, stop_point: torch.Tensor) -> torch.Tensor:
        reference = geometry[:, 0]
        delta = stop_point - reference[:, :2]
        cos_h, sin_h = torch.cos(reference[:, 2]), torch.sin(reference[:, 2])
        return torch.stack([
            delta[:, 0] * cos_h + delta[:, 1] * sin_h,
            -delta[:, 0] * sin_h + delta[:, 1] * cos_h,
        ], dim=-1)

    def forward(self, inputs: dict, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        geometry = inputs["map_geometry"]
        if geometry.shape[1:] != (self.segment_points, 3):
            raise ValueError(
                f"map_geometry has shape {tuple(geometry.shape[1:])} per token; "
                f"expected ({self.segment_points}, 3)"
            )
        map_batch = inputs["map_batch"].long()
        reference = geometry[:, 0]
        states = torch.stack([
            reference[:, 0], reference[:, 1],
            torch.cos(reference[:, 2]), torch.sin(reference[:, 2]),
        ], dim=-1)

        features = self.geometry_mlp(self.local_geometry(geometry))
        features = features + self.type_embedding(inputs["map_type"].long())
        features = features + self.light_embedding(inputs["map_light_type"].long())
        stop_local = self.local_stop_xy(
            geometry, inputs["map_stop_point"].to(geometry.dtype)
        )
        stop_feature = self.stop_point_embedding(stop_local)
        has_stop = inputs["map_has_stop_point"].bool()[:, None]
        stop_feature = torch.where(
            has_stop, stop_feature, self.unknown_stop_point_embedding[None]
        )
        speed = self.speed_embedding(inputs["map_speed_limit"].to(geometry.dtype)[:, None])
        known = inputs["map_has_speed_limit"].bool()[:, None]
        speed = torch.where(known, speed, self.unknown_speed_embedding[None])
        features = self.input_norm(features + speed + stop_feature)

        edge_index = build_batched_map_radius_graph(
            states[:, :2], map_batch, self.radius_m, self.max_neighbors,
        )
        relations = self.rel_encoder(
            _edge_relations(states, states, map_batch, map_batch, edge_index)
        )
        for layer in self.layers:
            features = layer(features, relations, edge_index)
        return features, states


class Encoder(nn.Module):
    """Agent history encoder with sparse packed map-to-map/map-to-agent fusion."""

    def __init__(self, config, rel_encoder=None):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.agent_num = config.agent_num
        self.local_attn_k = int(getattr(config, "local_attn_k", 16))
        self.a2a_attn_k = int(getattr(config, "a2a_attn_k", 32))
        self.a2a_radius_m = float(getattr(config, "a2a_radius_m", 60.0))
        self.pl2a_radius_m = float(getattr(config, "pl2a_radius_m", 30.0))
        self.max_map_neighbors = int(getattr(config, "max_map_neighbors", 100))
        self.use_adaptive_edge_radius = bool(getattr(config, "use_adaptive_edge_radius", True))
        self.adaptive_edge_base_radius_m = float(getattr(config, "adaptive_edge_base_radius_m", 10.0))
        self.adaptive_edge_time_horizon_s = float(getattr(config, "adaptive_edge_time_horizon_s", 2.0))
        self.adaptive_edge_min_radius_m = float(getattr(config, "adaptive_edge_min_radius_m", 20.0))
        self.adaptive_edge_max_radius_m = float(getattr(config, "adaptive_edge_max_radius_m", 80.0))

        if config.hidden_dim % config.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        head_dim = config.hidden_dim // config.num_heads
        self.rel_encoder = rel_encoder or RelationEncoder(config.hidden_dim, 64)
        self.agents_encoder = AgentEncoder(
            config.time_len,
            drop_path_rate=config.encoder_drop_path_rate,
            hidden_dim=config.hidden_dim,
            depth_self=config.encoder_depth,
            depth_cross=1,
            num_heads=config.num_heads,
        )
        self.map_encoder = SMARTMapEncoder(config, self.rel_encoder)
        self.fusion_layers = nn.ModuleList([
            nn.ModuleDict({
                "a2h": CrossFusionEncoder(
                    hidden_dim=config.hidden_dim, num_heads=config.num_heads,
                    drop_path_rate=config.encoder_drop_path_rate, depth=1,
                    device=getattr(config, "device", "cuda"), rel_encoder=self.rel_encoder,
                ),
                "a2m": AttentionLayer(
                    hidden_dim=config.hidden_dim, num_heads=config.num_heads,
                    head_dim=head_dim, dropout=config.encoder_drop_path_rate,
                    bipartite=True, has_pos_emb=True,
                ),
                "a2a": AttentionLayer(
                    hidden_dim=config.hidden_dim, num_heads=config.num_heads,
                    head_dim=head_dim, dropout=config.encoder_drop_path_rate,
                    bipartite=False, has_pos_emb=True,
                ),
            }) for _ in range(config.encoder_depth)
        ])
        self._init_weights()

    def _init_weights(self) -> None:
        def basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.zeros_(module.bias)
                nn.init.ones_(module.weight)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=1.0)

        _apply_basic_init(self, basic_init, (FourierEmbedding, RelationEncoder))

    def _scenario_agent_radius(self, inputs: dict, batch_size: int) -> torch.Tensor:
        speed = inputs["map_speed_limit"]
        device, dtype = speed.device, speed.dtype
        radii = torch.full((batch_size,), self.adaptive_edge_min_radius_m, device=device, dtype=dtype)
        for batch_index in range(batch_size):
            valid = (
                (inputs["map_batch"] == batch_index)
                & (inputs["map_type"] >= LANE_MAP_TYPE_MIN)
                & (inputs["map_type"] <= LANE_MAP_TYPE_MAX)
                & inputs["map_has_speed_limit"].bool()
                & torch.isfinite(speed) & (speed > 0)
            )
            parents = inputs["map_parent_id"][valid]
            lane_speeds = speed[valid]
            if parents.numel():
                unique_speeds = torch.stack([
                    lane_speeds[torch.nonzero(parents == parent, as_tuple=False)[0, 0]]
                    for parent in torch.unique(parents)
                ])
                mean_mps = unique_speeds.mean() * 0.44704
                radii[batch_index] = (
                    self.adaptive_edge_base_radius_m
                    + self.adaptive_edge_time_horizon_s * mean_mps
                ).clamp(self.adaptive_edge_min_radius_m, self.adaptive_edge_max_radius_m)
        return radii

    def forward(self, inputs: dict) -> dict:
        agents = inputs["agents_history"]
        batch_size, num_agents = agents.shape[:2]
        if num_agents != self.agent_num:
            raise ValueError(
                f"agents_history contains {num_agents} agents, expected agent_num={self.agent_num}"
            )
        agents_local = batch_transform_trajs_to_local_frame(agents)
        (agents_t, agent_features, agents_t_mask,
         agents_temporal_rel, agent_mask) = self.agents_encoder(
            agents_local, inputs["agents_type"]
        )
        map_features, map_states = self.map_encoder(inputs, batch_size)

        agent_nodes = torch.nonzero(~agent_mask, as_tuple=False)
        agent_batch = agent_nodes[:, 0]
        agent_states = agents[agent_nodes[:, 0], agent_nodes[:, 1], -1, :4]
        map_batch = inputs["map_batch"].long()

        map_radius = agents.new_full((batch_size,), self.pl2a_radius_m)
        a2m_edges = build_batched_radius_edge_index(
            map_states[:, :2], agent_states[:, :2], map_batch, agent_batch,
            batch_radius=map_radius, max_num_neighbors=self.max_map_neighbors,
        )
        a2m_rel = self.rel_encoder(
            _edge_relations(map_states, agent_states, map_batch, agent_batch, a2m_edges)
        )

        if self.use_adaptive_edge_radius:
            agent_radius = self._scenario_agent_radius(inputs, batch_size)
            a2a_edges = build_batched_radius_edge_index(
                agent_states[:, :2], agent_states[:, :2], agent_batch, agent_batch,
                batch_radius=agent_radius, max_num_neighbors=self.a2a_attn_k,
                exclude_self=True,
            )
        else:
            a2a_radius = agents.new_full((batch_size,), self.a2a_radius_m)
            a2a_edges = build_batched_radius_edge_index(
                agent_states[:, :2], agent_states[:, :2], agent_batch, agent_batch,
                batch_radius=a2a_radius, max_num_neighbors=self.a2a_attn_k,
                exclude_self=True,
            )
        a2a_rel = self.rel_encoder(
            _edge_relations(agent_states, agent_states, agent_batch, agent_batch, a2a_edges)
        )

        history_len = agents_t.shape[2]
        a2h_kv = agents_t.reshape(batch_size, num_agents * history_len, -1)
        a2h_kv_mask = agents_t_mask.reshape(batch_size, num_agents * history_len)
        current = torch.arange(num_agents, device=agents.device) * history_len + history_len - 1
        a2h_kv_mask[:, current] = False
        a2h_index = (
            torch.arange(num_agents, device=agents.device)[:, None] * history_len
            + torch.arange(history_len, device=agents.device)[None]
        ).unsqueeze(0).expand(batch_size, -1, -1).reshape(batch_size * num_agents, history_len)
        a2h_rel = agents.new_zeros((batch_size, num_agents, num_agents * history_len, 4))
        for index in range(num_agents):
            a2h_rel[:, index, index * history_len:(index + 1) * history_len] = agents_temporal_rel[:, index]
        self_mask = torch.zeros(
            (batch_size * num_agents, history_len), dtype=torch.bool, device=agents.device
        )
        self_mask[:, -1] = True

        for layer in self.fusion_layers:
            agent_features = layer["a2h"](
                agent_features, a2h_kv, agent_mask, a2h_kv_mask,
                a2h_index, a2h_rel, self_rel_mask=self_mask,
            )
            valid_features = agent_features[~agent_mask]
            if valid_features.numel():
                if a2m_edges.numel():
                    map_updated = layer["a2m"](
                        (map_features, valid_features), a2m_rel, a2m_edges
                    )
                    has_map_neighbor = torch.zeros(
                        valid_features.size(0), dtype=torch.bool, device=agents.device
                    )
                    has_map_neighbor[a2m_edges[1]] = True
                    valid_features = torch.where(
                        has_map_neighbor[:, None], map_updated, valid_features
                    )
                valid_features = layer["a2a"](valid_features, a2a_rel, a2a_edges)
                agent_features = torch.zeros_like(agent_features).index_put(
                    (agent_nodes[:, 0], agent_nodes[:, 1]), valid_features
                )

        dense_map, map_mask = packed_map_to_dense(
            map_features, inputs["map_ptr"], batch_size
        )
        return {
            "encoding": torch.cat([agent_features, dense_map], dim=1),
            "encoding_mask": torch.cat([agent_mask, map_mask], dim=1),
        }


__all__ = ["Encoder", "SMARTMapEncoder"]