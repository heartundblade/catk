"""SMART-style sparse graph attention utilities.

The :class:`AttentionLayer` implementation is adapted from SMART:
https://github.com/rainmaker22/SMART/blob/main/smart/layers/attention_layer.py

SMART is distributed under the Apache License 2.0. This adapted version keeps
the same message-passing, positional key/value injection, gated residual, and
feed-forward structure while making graph sizes explicit for empty bipartite
graphs used by the diffusion planner.
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch_cluster import knn, radius
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax


def _weight_init(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class AttentionLayer(MessagePassing):
    """Edge-aware attention with SMART's graph message-passing semantics."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
        bipartite: bool,
        has_pos_emb: bool,
        **kwargs,
    ) -> None:
        super().__init__(aggr="add", node_dim=0, **kwargs)

        if hidden_dim <= 0 or num_heads <= 0 or head_dim <= 0:
            raise ValueError("hidden_dim, num_heads, and head_dim must be positive")
        if num_heads * head_dim != hidden_dim:
            raise ValueError(
                "SMART AttentionLayer requires num_heads * head_dim == hidden_dim; "
                f"got {num_heads} * {head_dim} != {hidden_dim}"
            )

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.has_pos_emb = has_pos_emb
        self.scale = head_dim**-0.5

        self.to_q = nn.Linear(hidden_dim, head_dim * num_heads)
        self.to_k = nn.Linear(hidden_dim, head_dim * num_heads, bias=False)
        self.to_v = nn.Linear(hidden_dim, head_dim * num_heads)

        if has_pos_emb:
            self.to_k_r = nn.Linear(hidden_dim, head_dim * num_heads, bias=False)
            self.to_v_r = nn.Linear(hidden_dim, head_dim * num_heads)

        self.to_s = nn.Linear(hidden_dim, head_dim * num_heads)
        self.to_g = nn.Linear(head_dim * num_heads + hidden_dim, head_dim * num_heads)
        self.to_out = nn.Linear(head_dim * num_heads, hidden_dim)

        self.attn_drop = nn.Dropout(dropout)
        self.ff_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        if bipartite:
            self.attn_prenorm_x_src = nn.LayerNorm(hidden_dim)
            self.attn_prenorm_x_dst = nn.LayerNorm(hidden_dim)
        else:
            self.attn_prenorm_x_src = nn.LayerNorm(hidden_dim)
            self.attn_prenorm_x_dst = self.attn_prenorm_x_src

        if has_pos_emb:
            self.attn_prenorm_r = nn.LayerNorm(hidden_dim)

        self.attn_postnorm = nn.LayerNorm(hidden_dim)
        self.ff_prenorm = nn.LayerNorm(hidden_dim)
        self.ff_postnorm = nn.LayerNorm(hidden_dim)
        self.attention_weight: Optional[torch.Tensor] = None

        self.apply(_weight_init)

    def forward(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        r: Optional[torch.Tensor],
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            x_src = x_dst = self.attn_prenorm_x_src(x)
            residual = x
        else:
            x_src, x_dst = x
            x_src = self.attn_prenorm_x_src(x_src)
            x_dst = self.attn_prenorm_x_dst(x_dst)
            residual = x[1]

        if self.has_pos_emb and r is not None:
            r = self.attn_prenorm_r(r)

        residual = residual + self.attn_postnorm(
            self._attn_block(x_src, x_dst, r, edge_index)
        )
        residual = residual + self.ff_postnorm(
            self._ff_block(self.ff_prenorm(residual))
        )
        return residual

    def message(
        self,
        q_i: torch.Tensor,
        k_j: torch.Tensor,
        v_j: torch.Tensor,
        r: Optional[torch.Tensor],
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.has_pos_emb and r is not None:
            k_j = k_j + self.to_k_r(r).view(
                -1, self.num_heads, self.head_dim
            )
            v_j = v_j + self.to_v_r(r).view(
                -1, self.num_heads, self.head_dim
            )

        similarity = (q_i * k_j).sum(dim=-1) * self.scale
        attention = softmax(similarity, index, ptr)
        self.attention_weight = attention.sum(dim=-1).detach()
        attention = self.attn_drop(attention)
        return v_j * attention.unsqueeze(-1)

    def update(self, inputs: torch.Tensor, x_dst: torch.Tensor) -> torch.Tensor:
        inputs = inputs.reshape(x_dst.size(0), self.num_heads * self.head_dim)
        gate = torch.sigmoid(self.to_g(torch.cat([inputs, x_dst], dim=-1)))
        return inputs + gate * (self.to_s(x_dst) - inputs)

    def _attn_block(
        self,
        x_src: torch.Tensor,
        x_dst: torch.Tensor,
        r: Optional[torch.Tensor],
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        query = self.to_q(x_dst).view(-1, self.num_heads, self.head_dim)
        key = self.to_k(x_src).view(-1, self.num_heads, self.head_dim)
        value = self.to_v(x_src).view(-1, self.num_heads, self.head_dim)
        aggregated = self.propagate(
            edge_index=edge_index,
            size=(x_src.size(0), x_dst.size(0)),
            x_dst=x_dst,
            q=query,
            k=key,
            v=value,
            r=r,
        )
        return self.to_out(aggregated)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff_mlp(x)


def compute_scenario_edge_radius(
    speed_limit_mph: torch.Tensor,
    has_speed_limit: torch.Tensor,
    lane_mask: torch.Tensor,
    base_radius_m: float = 10.0,
    time_horizon_s: float = 2.0,
    min_radius_m: float = 30.0,
    max_radius_m: float = 80.0,
) -> torch.Tensor:
    """Compute one bounded interaction radius per scenario.

    Only non-padded lanes with a known, finite, positive speed limit
    contribute to the arithmetic mean. Speed limits are converted from mph to
    m/s before applying the time-horizon rule. Scenarios without a usable
    speed limit receive ``min_radius_m``.
    """

    if min_radius_m < 0 or max_radius_m < min_radius_m:
        raise ValueError("radius bounds must satisfy 0 <= min_radius_m <= max_radius_m")
    if base_radius_m < 0 or time_horizon_s < 0:
        raise ValueError("base_radius_m and time_horizon_s must be non-negative")

    def _remove_trailing_singleton(x: torch.Tensor, name: str) -> torch.Tensor:
        if x.ndim == 3 and x.size(-1) == 1:
            return x.squeeze(-1)
        if x.ndim != 2:
            raise ValueError(f"{name} must have shape [B, L] or [B, L, 1]")
        return x

    speed_limit_mph = _remove_trailing_singleton(
        speed_limit_mph, "speed_limit_mph"
    )
    has_speed_limit = _remove_trailing_singleton(
        has_speed_limit, "has_speed_limit"
    )
    lane_mask = _remove_trailing_singleton(lane_mask, "lane_mask")
    if not (
        speed_limit_mph.shape == has_speed_limit.shape == lane_mask.shape
    ):
        raise ValueError("speed-limit tensors and lane_mask must have matching shapes")

    if not speed_limit_mph.is_floating_point():
        speed_limit_mph = speed_limit_mph.float()
    has_speed_limit = has_speed_limit.to(device=speed_limit_mph.device).bool()
    lane_mask = lane_mask.to(device=speed_limit_mph.device).bool()

    valid = (
        has_speed_limit
        & ~lane_mask
        & torch.isfinite(speed_limit_mph)
        & (speed_limit_mph > 0)
    )
    valid_count = valid.sum(dim=-1)
    speed_sum_mph = torch.where(
        valid, speed_limit_mph, torch.zeros_like(speed_limit_mph)
    ).sum(dim=-1)
    average_speed_mph = speed_sum_mph / valid_count.clamp_min(1).to(
        speed_limit_mph.dtype
    )
    average_speed_mps = average_speed_mph * 0.44704

    radius = base_radius_m + time_horizon_s * average_speed_mps
    radius = radius.clamp(min=min_radius_m, max=max_radius_m)
    fallback = torch.full_like(radius, min_radius_m)
    return torch.where(valid_count > 0, radius, fallback)


def build_batched_knn_edge_index(
    source_positions: torch.Tensor,
    target_positions: torch.Tensor,
    source_batch: torch.Tensor,
    target_batch: torch.Tensor,
    k: int,
    exclude_self: bool = False,
) -> torch.Tensor:
    """Build compact source-to-target KNN edges without crossing scenes.

    ``torch_cluster.knn`` returns ``[target, source]`` indices. This helper
    converts them to PyG's ``[source, target]`` convention. When
    ``exclude_self`` is true, source and target tensors must use the same
    compact node ordering.
    """

    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if source_positions.ndim != 2 or target_positions.ndim != 2:
        raise ValueError("source_positions and target_positions must be rank-2")
    if source_positions.size(-1) != target_positions.size(-1):
        raise ValueError("source and target position dimensions must match")
    if source_batch.numel() != source_positions.size(0):
        raise ValueError("source_batch must contain one id per source node")
    if target_batch.numel() != target_positions.size(0):
        raise ValueError("target_batch must contain one id per target node")
    if exclude_self and source_positions.size(0) != target_positions.size(0):
        raise ValueError("exclude_self requires aligned source and target nodes")

    device = source_positions.device
    if source_positions.size(0) == 0 or target_positions.size(0) == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    edge_parts = []
    for batch_id in torch.unique(target_batch).tolist():
        source_ids = torch.nonzero(source_batch == batch_id, as_tuple=False).flatten()
        target_ids = torch.nonzero(target_batch == batch_id, as_tuple=False).flatten()
        if source_ids.numel() == 0 or target_ids.numel() == 0:
            continue

        extra_neighbor = 1 if exclude_self else 0
        k_query = min(k + extra_neighbor, source_ids.numel())
        local_pairs = knn(
            x=source_positions[source_ids],
            y=target_positions[target_ids],
            k=k_query,
        )
        source = source_ids[local_pairs[1]]
        target = target_ids[local_pairs[0]]

        if exclude_self:
            keep = source != target
            source = source[keep]
            target = target[keep]

            # The extra query neighbor guarantees enough candidates after the
            # self-edge is removed. Truncate explicitly for coincident nodes.
            keep_parts = []
            for target_id in target_ids:
                candidates = torch.nonzero(target == target_id, as_tuple=False).flatten()
                keep_parts.append(candidates[:k])
            if keep_parts:
                keep_indices = torch.cat(keep_parts)
                source = source[keep_indices]
                target = target[keep_indices]

        if source.numel() > 0:
            edge_parts.append(torch.stack([source, target], dim=0))

    if not edge_parts:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    return torch.cat(edge_parts, dim=1)


def build_batched_radius_edge_index(
    source_positions: torch.Tensor,
    target_positions: torch.Tensor,
    source_batch: torch.Tensor,
    target_batch: torch.Tensor,
    batch_radius: torch.Tensor,
    max_num_neighbors: int,
    exclude_self: bool = False,
) -> torch.Tensor:
    """Build SMART-style radius edges with a per-scenario neighbor cap.

    Radius membership selects admissible query-key pairs. The
    ``max_num_neighbors`` argument only bounds the number of incoming edges per
    target. Different scenarios may use different scalar radii.
    """

    if max_num_neighbors <= 0:
        raise ValueError(
            f"max_num_neighbors must be positive, got {max_num_neighbors}"
        )
    if source_positions.ndim != 2 or target_positions.ndim != 2:
        raise ValueError("source_positions and target_positions must be rank-2")
    if source_positions.size(-1) != target_positions.size(-1):
        raise ValueError("source and target position dimensions must match")
    if source_batch.numel() != source_positions.size(0):
        raise ValueError("source_batch must contain one id per source node")
    if target_batch.numel() != target_positions.size(0):
        raise ValueError("target_batch must contain one id per target node")
    if exclude_self and source_positions.size(0) != target_positions.size(0):
        raise ValueError("exclude_self requires aligned source and target nodes")
    if batch_radius.ndim != 1:
        raise ValueError("batch_radius must have shape [B]")
    if target_batch.numel() > 0 and int(target_batch.max()) >= batch_radius.numel():
        raise ValueError("batch_radius does not cover every target batch id")

    device = source_positions.device
    batch_radius = batch_radius.to(device=device, dtype=source_positions.dtype)
    if not torch.isfinite(batch_radius).all() or (batch_radius < 0).any():
        raise ValueError("batch_radius must contain finite, non-negative values")
    if source_positions.size(0) == 0 or target_positions.size(0) == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    edge_parts = []
    for batch_id in torch.unique(target_batch).tolist():
        source_ids = torch.nonzero(source_batch == batch_id, as_tuple=False).flatten()
        target_ids = torch.nonzero(target_batch == batch_id, as_tuple=False).flatten()
        if source_ids.numel() == 0 or target_ids.numel() == 0:
            continue

        extra_neighbor = 1 if exclude_self else 0
        local_pairs = radius(
            x=source_positions[source_ids],
            y=target_positions[target_ids],
            r=float(batch_radius[batch_id]),
            max_num_neighbors=max_num_neighbors + extra_neighbor,
        )
        source = source_ids[local_pairs[1]]
        target = target_ids[local_pairs[0]]

        if exclude_self:
            keep = source != target
            source = source[keep]
            target = target[keep]

            # Match radius_graph(loop=False): request one extra candidate for
            # the self-edge, remove it, then restore the requested upper bound.
            keep_parts = []
            for target_id in target_ids:
                candidates = torch.nonzero(target == target_id, as_tuple=False).flatten()
                keep_parts.append(candidates[:max_num_neighbors])
            if keep_parts:
                keep_indices = torch.cat(keep_parts)
                source = source[keep_indices]
                target = target[keep_indices]

        if source.numel() > 0:
            edge_parts.append(torch.stack([source, target], dim=0))

    if not edge_parts:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    return torch.cat(edge_parts, dim=1)


def build_batched_anchor_edge_index(
    source_anchor_positions: torch.Tensor,
    target_positions: torch.Tensor,
    source_batch: torch.Tensor,
    target_batch: torch.Tensor,
    max_num_neighbors: int,
    batch_radius: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build source-to-target edges from the closest source anchor.

    Each source node is represented by a small, fixed set of anchors, such as
    the start, midpoint, and endpoint of a map polyline. A source is eligible
    for a target when the minimum distance over its anchors is inside the
    target scenario's radius. Eligible sources are ranked by that same minimum
    distance and capped per target. When ``batch_radius`` is ``None``, this is
    anchor-based KNN without radius filtering.
    """

    if max_num_neighbors <= 0:
        raise ValueError(
            f"max_num_neighbors must be positive, got {max_num_neighbors}"
        )
    if source_anchor_positions.ndim != 3:
        raise ValueError(
            "source_anchor_positions must have shape [num_sources, num_anchors, D]"
        )
    if source_anchor_positions.size(1) == 0:
        raise ValueError("source nodes must contain at least one anchor")
    if target_positions.ndim != 2:
        raise ValueError("target_positions must be rank-2")
    if source_anchor_positions.size(-1) != target_positions.size(-1):
        raise ValueError("source anchor and target position dimensions must match")
    if source_batch.numel() != source_anchor_positions.size(0):
        raise ValueError("source_batch must contain one id per source node")
    if target_batch.numel() != target_positions.size(0):
        raise ValueError("target_batch must contain one id per target node")

    device = source_anchor_positions.device
    if target_positions.device != device:
        raise ValueError("source anchors and target positions must share a device")

    if batch_radius is not None:
        if batch_radius.ndim != 1:
            raise ValueError("batch_radius must have shape [B]")
        if target_batch.numel() > 0 and int(target_batch.max()) >= batch_radius.numel():
            raise ValueError("batch_radius does not cover every target batch id")
        batch_radius = batch_radius.to(
            device=device, dtype=source_anchor_positions.dtype
        )
        if not torch.isfinite(batch_radius).all() or (batch_radius < 0).any():
            raise ValueError("batch_radius must contain finite, non-negative values")

    if source_anchor_positions.size(0) == 0 or target_positions.size(0) == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    edge_parts = []
    num_anchors = source_anchor_positions.size(1)
    for batch_id in torch.unique(target_batch).tolist():
        source_ids = torch.nonzero(source_batch == batch_id, as_tuple=False).flatten()
        target_ids = torch.nonzero(target_batch == batch_id, as_tuple=False).flatten()
        if source_ids.numel() == 0 or target_ids.numel() == 0:
            continue

        # [num_targets, num_sources * num_anchors] ->
        # [num_targets, num_sources]. With three anchors this is small enough
        # to avoid expanding every sampled polyline point into a graph node.
        anchor_distance = torch.cdist(
            target_positions[target_ids],
            source_anchor_positions[source_ids].reshape(-1, target_positions.size(-1)),
        )
        source_distance = anchor_distance.view(
            target_ids.numel(), source_ids.numel(), num_anchors
        ).amin(dim=-1)

        if batch_radius is not None:
            source_distance = source_distance.masked_fill(
                source_distance > batch_radius[batch_id], float("inf")
            )

        k = min(max_num_neighbors, source_ids.numel())
        nearest_distance, nearest_source = torch.topk(
            source_distance, k=k, dim=-1, largest=False, sorted=True
        )
        keep = torch.isfinite(nearest_distance)
        if keep.any():
            source = source_ids[nearest_source[keep]]
            target = target_ids[:, None].expand(-1, k)[keep]
            edge_parts.append(torch.stack([source, target], dim=0))

    if not edge_parts:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    return torch.cat(edge_parts, dim=1)


def gather_smart_edge_relations(
    relations: torch.Tensor,
    source_nodes: torch.Tensor,
    target_nodes: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """Gather and orient query-first relations for SMART source-to-target edges.

    ``source_nodes`` and ``target_nodes`` contain ``[batch, token_index]`` rows
    into the full relation tensor. The planner stores ``target - source`` in
    the target frame; SMART consumes ``source - target`` in that same frame.
    """

    if edge_index.numel() == 0:
        return relations.new_empty((0, relations.size(-1)))

    source = source_nodes[edge_index[0]]
    target = target_nodes[edge_index[1]]
    if not torch.equal(source[:, 0], target[:, 0]):
        raise ValueError("graph edges must not cross batch elements")

    query_first = relations[target[:, 0], target[:, 1], source[:, 1]]
    return torch.stack(
        [
            -query_first[:, 0],
            -query_first[:, 1],
            query_first[:, 2],
            -query_first[:, 3],
        ],
        dim=-1,
    )


def gather_smart_anchor_edge_relations(
    source_anchor_states: torch.Tensor,
    target_states: torch.Tensor,
    source_batch: torch.Tensor,
    target_batch: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """Compute SMART relations from every source anchor to its target node.

    States use ``[x, y, cos(heading), sin(heading)]``. Returned positions and
    heading differences point from each source anchor to the target node and
    are expressed in the target node's local frame. The output has shape
    ``[num_edges, num_anchors, 4]``.
    """

    if source_anchor_states.ndim != 3 or source_anchor_states.size(-1) != 4:
        raise ValueError("source_anchor_states must have shape [S, A, 4]")
    if target_states.ndim != 2 or target_states.size(-1) != 4:
        raise ValueError("target_states must have shape [T, 4]")
    if source_batch.numel() != source_anchor_states.size(0):
        raise ValueError("source_batch must contain one id per source node")
    if target_batch.numel() != target_states.size(0):
        raise ValueError("target_batch must contain one id per target node")

    if edge_index.numel() == 0:
        return source_anchor_states.new_empty(
            (0, source_anchor_states.size(1), source_anchor_states.size(-1))
        )

    source_index, target_index = edge_index
    if not torch.equal(source_batch[source_index], target_batch[target_index]):
        raise ValueError("graph edges must not cross batch elements")

    source = source_anchor_states[source_index]
    target = target_states[target_index]

    dx = source[..., 0] - target[:, None, 0]
    dy = source[..., 1] - target[:, None, 1]
    target_cos = target[:, None, 2]
    target_sin = target[:, None, 3]
    local_x = dx * target_cos + dy * target_sin
    local_y = -dx * target_sin + dy * target_cos

    source_cos = source[..., 2]
    source_sin = source[..., 3]
    relative_cos = target_cos * source_cos + target_sin * source_sin
    relative_sin = source_sin * target_cos - source_cos * target_sin

    return torch.stack(
        [local_x, local_y, relative_cos, relative_sin], dim=-1
    )


__all__ = [
    "AttentionLayer",
    "build_batched_anchor_edge_index",
    "build_batched_knn_edge_index",
    "build_batched_radius_edge_index",
    "compute_scenario_edge_radius",
    "gather_smart_anchor_edge_relations",
    "gather_smart_edge_relations",
]
