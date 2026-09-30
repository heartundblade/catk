# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling <CAT-K> or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

import multiprocessing
import pickle
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
# import pandas as pd
from tqdm import tqdm

MAX_NUM_OBJECTS = 64
CURRENT_INDEX = 10
NUM_POINTS_POLYLINE = 30
NUM_POINTS_STATIC_MAP = 10

MAP_SCHEMA_VERSION = 9
MAP_SEGMENT_LENGTH_M = 5.0
MAP_SEGMENT_POINTS = 3
MAP_TYPE = {
    "unknown": 0,
    "lane_unknown": 1,
    "lane_freeway": 2,
    "lane_surface_street": 3,
    "lane_bike_lane": 4,
    "road_line_unknown": 5,
    "road_line_broken_single_white": 6,
    "road_line_solid_single_white": 7,
    "road_line_solid_double_white": 8,
    "road_line_broken_single_yellow": 9,
    "road_line_broken_double_yellow": 10,
    "road_line_solid_single_yellow": 11,
    "road_line_solid_double_yellow": 12,
    "road_line_passing_double_yellow": 13,
    "road_edge_unknown": 14,
    "road_edge_boundary": 15,
    "road_edge_median": 16,
    "crosswalk": 17,
    "speed_bump": 18,
    "driveway": 19,
    "stop_sign": 20,
}
NUM_MAP_TYPES = len(MAP_TYPE)
NUM_MAP_LIGHT_TYPES = 9
_MAP_FEATURE_ORDER = (
    "lane", "road_line", "road_edge", "crosswalk", "speed_bump",
    "driveway", "stop_sign",
)
_MAP_SOURCE_TYPE = {
    # Tuple positions are the raw Waymo enum values, starting at zero.
    "lane": tuple(MAP_TYPE[name] for name in (
        "lane_unknown", "lane_freeway", "lane_surface_street", "lane_bike_lane",
    )),
    "road_line": tuple(MAP_TYPE[name] for name in (
        "road_line_unknown", "road_line_broken_single_white",
        "road_line_solid_single_white", "road_line_solid_double_white",
        "road_line_broken_single_yellow", "road_line_broken_double_yellow",
        "road_line_solid_single_yellow", "road_line_solid_double_yellow",
        "road_line_passing_double_yellow",
    )),
    "road_edge": tuple(MAP_TYPE[name] for name in (
        "road_edge_unknown", "road_edge_boundary", "road_edge_median",
    )),
}
_MAP_POLYGON_CATEGORIES = {"crosswalk", "speed_bump", "driveway"}


def empty_map_tokens() -> Dict[str, np.ndarray]:
    return {
        "map_geometry": np.empty((0, MAP_SEGMENT_POINTS, 3), dtype=np.float32),
        "map_type": np.empty((0,), dtype=np.int64),
        "map_light_type": np.empty((0,), dtype=np.int64),
        "map_stop_point": np.empty((0, 2), dtype=np.float32),
        "map_has_stop_point": np.empty((0,), dtype=np.bool_),
        "map_speed_limit": np.empty((0,), dtype=np.float32),
        "map_has_speed_limit": np.empty((0,), dtype=np.bool_),
        "map_parent_id": np.empty((0,), dtype=np.int64),
        "map_segment_index": np.empty((0,), dtype=np.int32),
        "map_schema_version": np.asarray(MAP_SCHEMA_VERSION, dtype=np.int32),
    }


def _global_map_type(category: str, source_type: Optional[int]) -> int:
    if category in _MAP_SOURCE_TYPE:
        subtypes = _MAP_SOURCE_TYPE[category]
        source_index = 0 if source_type is None else int(source_type)
        return subtypes[source_index] if 0 <= source_index < len(subtypes) else subtypes[0]
    return MAP_TYPE.get(category, MAP_TYPE["unknown"])


def _remove_consecutive_duplicates(points: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    if len(points) < 2:
        return points
    keep = np.r_[True, np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1) > eps]
    return points[keep]


def split_discontinuous_runs(
    points: np.ndarray,
    max_spatial_jump_m: float = 3.0,
) -> List[np.ndarray]:
    points = _remove_consecutive_duplicates(np.asarray(points, dtype=np.float32))
    if len(points) < 2:
        return []
    delta = np.diff(points[:, :2], axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    runs = []
    start = 0
    for edge_index, length in enumerate(lengths):
        if length > max_spatial_jump_m:
            # The edge itself is discontinuous: do not include it in either run.
            end = edge_index + 1
            if end - start >= 2:
                runs.append(points[start:end])
            start = end
    if len(points) - start >= 2:
        runs.append(points[start:])
    return runs


def _densify_polygon_edges(points: np.ndarray, spacing_m: float = 1.0) -> np.ndarray:
    """Sample valid polygon edges before applying SMART's 3 m gap rule."""
    sampled = []
    for start, end in zip(points[:-1], points[1:]):
        distance = float(np.linalg.norm(end[:2] - start[:2]))
        steps = max(1, int(np.ceil(distance / spacing_m)))
        alpha = np.arange(steps, dtype=np.float32) / steps
        sampled.extend(start[None] + alpha[:, None] * (end - start)[None])
    sampled.append(points[-1])
    return np.asarray(sampled, dtype=np.float32)


def _sample_map_run(run: np.ndarray, distances: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    xy = run[:, :2].astype(np.float64)
    delta = np.diff(xy, axis=0)
    segment_lengths = np.linalg.norm(delta, axis=1)
    cumulative = np.r_[0.0, np.cumsum(segment_lengths)]
    indices = np.searchsorted(cumulative, distances, side="right") - 1
    indices = np.clip(indices, 0, len(segment_lengths) - 1)
    alpha = np.divide(
        distances - cumulative[indices], segment_lengths[indices],
        out=np.zeros_like(distances, dtype=np.float64),
        where=segment_lengths[indices] > 0,
    )
    alpha = np.clip(alpha, 0.0, 1.0)
    sampled = xy[indices] + alpha[:, None] * delta[indices]
    heading = np.arctan2(delta[indices, 1], delta[indices, 0])
    return sampled.astype(np.float32), heading.astype(np.float32)


def fragment_polyline(
    points: np.ndarray,
    segment_length_m: float = MAP_SEGMENT_LENGTH_M,
) -> List[np.ndarray]:
    """Split a polyline into 5 m tokens sampled at start, midpoint, and end."""
    if segment_length_m <= 0:
        raise ValueError("segment_length_m must be positive")
    fragments = []
    for run in split_discontinuous_runs(points):
        length = float(np.linalg.norm(np.diff(run[:, :2], axis=0), axis=1).sum())
        start = 0.0
        while length - start > 1e-8:
            end = min(start + segment_length_m, length)
            sample_distances = np.array([start, (start + end) / 2, end], dtype=np.float64)
            xy, heading = _sample_map_run(run, sample_distances)
            fragments.append(np.column_stack([xy, heading]))
            start = end
    return fragments


def _map_feature_points(feature: Mapping, all_polylines: np.ndarray) -> np.ndarray:
    if "points" in feature:
        return np.asarray(feature["points"], dtype=np.float32)[..., :2]
    start, end = feature["polyline_index"]
    return np.asarray(all_polylines[start:end], dtype=np.float32)[..., :2]


def _traffic_light_by_lane(traffic_light: Mapping) -> Dict[int, int]:
    lane_ids = np.asarray(traffic_light.get("traffic_lane_ids", []), dtype=np.int64)
    states = np.asarray(traffic_light.get("traffic_light_states", []), dtype=np.int64)
    return {int(lane): int(state) for lane, state in zip(lane_ids, states)}


def _traffic_stop_point_by_lane(traffic_light: Mapping) -> Dict[int, np.ndarray]:
    lane_ids = np.asarray(traffic_light.get("traffic_lane_ids", []), dtype=np.int64)
    points = np.asarray(traffic_light.get("traffic_stop_points", []), dtype=np.float32).reshape(-1, 2)
    valid = np.asarray(
        traffic_light.get("traffic_has_stop_points", np.ones(len(points), dtype=np.bool_)),
        dtype=np.bool_,
    )
    return {
        int(lane_ids[index]): points[index]
        for index in range(min(len(lane_ids), len(points), len(valid)))
        if valid[index]
    }


def _append_map_token(
    records: Dict[str, list], geometry: np.ndarray, category: str,
    feature: Mapping, segment_index: int, light_by_lane: Mapping[int, int],
    stop_by_lane: Mapping[int, np.ndarray],
) -> None:
    parent_id = int(feature.get("id", -1))
    source_type = feature.get("type", feature.get("source_type", -1))
    speed_limit = float(feature.get("speed_limit_mph", feature.get("speed_limit", 0.0)))
    has_speed = bool(feature.get("has_speed_limit", speed_limit > 0.0))
    records["map_geometry"].append(geometry)
    records["map_type"].append(_global_map_type(category, source_type))
    records["map_light_type"].append(light_by_lane.get(parent_id, 0) if category == "lane" else 0)
    stop_point = stop_by_lane.get(parent_id) if category == "lane" else None
    records["map_stop_point"].append(
        stop_point if stop_point is not None else np.zeros(2, dtype=np.float32)
    )
    records["map_has_stop_point"].append(stop_point is not None)
    records["map_speed_limit"].append(speed_limit if has_speed else 0.0)
    records["map_has_speed_limit"].append(has_speed)
    records["map_parent_id"].append(parent_id)
    records["map_segment_index"].append(segment_index)


def build_smart_map_tokens(
    map_infos: Mapping,
    traffic_light: Mapping,
    segment_length_m: float = MAP_SEGMENT_LENGTH_M,
) -> Dict[str, np.ndarray]:
    """Convert every decoded map feature to the unfiltered cache-v9 token schema."""
    if segment_length_m <= 0:
        raise ValueError("segment_length_m must be positive")
    records = {
        key: [] for key in empty_map_tokens()
        if key != "map_schema_version"
    }
    all_polylines = np.asarray(map_infos.get("all_polylines", []), dtype=np.float32)
    light_by_lane = _traffic_light_by_lane(traffic_light)
    stop_by_lane = _traffic_stop_point_by_lane(traffic_light)
    for category in _MAP_FEATURE_ORDER:
        for feature in map_infos.get(category, []):
            points = _map_feature_points(feature, all_polylines)
            if category == "stop_sign":
                if len(points) == 0:
                    continue
                geometry = np.tile(
                    np.r_[points[0, :2], 0.0], (MAP_SEGMENT_POINTS, 1)
                ).astype(np.float32)
                _append_map_token(records, geometry, category, feature, 0, light_by_lane, stop_by_lane)
                continue
            if category in _MAP_POLYGON_CATEGORIES and len(points) > 2:
                if not np.allclose(points[0], points[-1]):
                    points = np.concatenate([points, points[:1]], axis=0)
                points = _densify_polygon_edges(points)
            for index, geometry in enumerate(fragment_polyline(
                points, segment_length_m
            )):
                _append_map_token(records, geometry, category, feature, index, light_by_lane, stop_by_lane)
    if not records["map_geometry"]:
        return empty_map_tokens()
    return {
        "map_geometry": np.asarray(records["map_geometry"], dtype=np.float32),
        "map_type": np.asarray(records["map_type"], dtype=np.int64),
        "map_light_type": np.asarray(records["map_light_type"], dtype=np.int64),
        "map_stop_point": np.asarray(records["map_stop_point"], dtype=np.float32),
        "map_has_stop_point": np.asarray(records["map_has_stop_point"], dtype=np.bool_),
        "map_speed_limit": np.asarray(records["map_speed_limit"], dtype=np.float32),
        "map_has_speed_limit": np.asarray(records["map_has_speed_limit"], dtype=np.bool_),
        "map_parent_id": np.asarray(records["map_parent_id"], dtype=np.int64),
        "map_segment_index": np.asarray(records["map_segment_index"], dtype=np.int32),
        "map_schema_version": np.asarray(MAP_SCHEMA_VERSION, dtype=np.int32),
    }


# agent_types = {0: "vehicle", 1: "pedestrian", 2: "cyclist"}
# agent_roles = {0: "ego_vehicle", 1: "interest", 2: "predict"}
# polyline_type = {
#     # for lane
#     "TYPE_FREEWAY": 0,
#     "TYPE_SURFACE_STREET": 1,
#     "TYPE_STOP_SIGN": 2,
#     "TYPE_BIKE_LANE": 3,
#     # for roadedge
#     "TYPE_ROAD_EDGE_BOUNDARY": 4,
#     "TYPE_ROAD_EDGE_MEDIAN": 5,
#     # for roadline
#     "BROKEN": 6,
#     "SOLID_SINGLE": 7,
#     "DOUBLE": 8,
#     # for crosswalk, speed bump and drive way
#     "TYPE_CROSSWALK": 9,
# }
_polygon_types = ["lane", "road_edge", "road_line", "crosswalk"]
_polygon_light_type = [
    "NO_LANE_STATE",
    "LANE_STATE_UNKNOWN",
    "LANE_STATE_STOP",
    "LANE_STATE_GO",
    "LANE_STATE_CAUTION",
]

polyline_type = {
    # for lane
    'TYPE_UNDEFINED': -1,
    'TYPE_FREEWAY': 1,
    'TYPE_SURFACE_STREET': 2,
    'TYPE_BIKE_LANE': 3,

    # for roadline
    'TYPE_UNKNOWN': -1,
    'TYPE_BROKEN_SINGLE_WHITE': 6,
    'TYPE_SOLID_SINGLE_WHITE': 7,
    'TYPE_SOLID_DOUBLE_WHITE': 8,
    'TYPE_BROKEN_SINGLE_YELLOW': 9,
    'TYPE_BROKEN_DOUBLE_YELLOW': 10,
    'TYPE_SOLID_SINGLE_YELLOW': 11,
    'TYPE_SOLID_DOUBLE_YELLOW': 12,
    'TYPE_PASSING_DOUBLE_YELLOW': 13,

    # for roadedge
    'TYPE_ROAD_EDGE_BOUNDARY': 15,
    'TYPE_ROAD_EDGE_MEDIAN': 16,

    # for stopsign
    'TYPE_STOP_SIGN': 17,

    # for crosswalk
    'TYPE_CROSSWALK': 18,

    # for speed bump
    'TYPE_SPEED_BUMP': 19
}

signal_state = {
        0: "LANE_STATE_UNKNOWN",
        #  States for traffic signals with arrows.
        1: "LANE_STATE_ARROW_STOP",
        2: "LANE_STATE_ARROW_CAUTION",
        3: "LANE_STATE_ARROW_GO",
        #  Standard round traffic signals.
        4: "LANE_STATE_STOP",
        5: "LANE_STATE_CAUTION",
        6: "LANE_STATE_GO",
        #  Flashing light signals.
        7: "LANE_STATE_FLASHING_STOP",
        8: "LANE_STATE_FLASHING_CAUTION",
    }

_polygon_to_polygon_types = ['NONE', 'PRED', 'SUCC', 'LEFT', 'RIGHT']


def sort_polygons_by_distance(
        pos_polygons, 
        reference_points,
    ):
    """
    Sort polygons by the distance to reference_points.

    Args:
        pos_polygons (np.ndarray): [num_polygons, 3, 2]; [start, end, center]; [x, y]
        reference_points (np.ndarray): [num_agents, 2] [x, y]
    Returns:
        index (np.ndarray): [num_polygons] index of polygons sorted by distance to reference_points
    """
    diff = pos_polygons[:, :, None, :] - reference_points[None, None, None, :]

    distances = np.linalg.norm(diff, axis=-1)
    min_dist_to_agents = distances.min(axis=1)
    final_min_dist = min_dist_to_agents.min(axis=1)

    index = np.argsort(final_min_dist)
    return index


def _compute_polygon_positions(polygons, all_polylines):
    """
    Compute position features (start, end, center) for a list of polygons.
    
    Args:
        polygons (list): List of polygon dictionaries with "polyline_index" key
        all_polylines (np.ndarray): All polylines data
    Returns:
        pos (np.ndarray): [num_polygons, 3, 2] - position array
    """
    num_polygons = len(polygons)
    pos = np.zeros([num_polygons, 3, 2])
    for i, polygon in enumerate(polygons):
        polyline_index = polygon["polyline_index"]
        pos[i, 0, :] = all_polylines[polyline_index[0], :2]
        pos[i, 1, :] = all_polylines[polyline_index[1]-1, :2]
        pos[i, 2, :] = np.mean(all_polylines[polyline_index[0]:polyline_index[1], :2], axis=0)
    return pos


def _filter_by_distance(positions, agents_data):
    """
    Sort polygons by distance to agents. No longer truncates here;
    truncation/padding is deferred to data loading (collate_fn).
    
    Args:
        positions (np.ndarray): [num_polygons, 3, 2] - position array
        agents_data (np.ndarray): Agent positions for reference
    Returns:
        index (np.ndarray): Indices of all polygons sorted by distance
    """
    index = sort_polygons_by_distance(positions, agents_data)
    return index


def get_map_features(
        map_infos, 
        tf_current_light, 
        sdc_coord,
        num_points_polyline=30,
        num_points_static_map=10,
    ):
    """
    Get polylines pos, heading, traffic light state and type from
    map_infos, filtered by agent_data. 
    Return lane and road edge together, and others together.
    """
    all_polylines = map_infos["all_polylines"]
    lane_segments = map_infos['lane']
    road_edges = map_infos['road_edge']
    road_lines = map_infos['road_line']
    stop_signs = map_infos['stop_sign']
    crosswalks = map_infos['crosswalk']
    speed_bumps = map_infos['speed_bump']

    # Extract IDs from each type
    # lane_segment_ids = [info["id"] for info in lane_segments]
    # road_edge_ids = [info["id"] for info in road_edges]
    # road_line_ids = [info["id"] for info in road_lines]
    # stop_sign_ids = [info["id"] for info in stop_signs]
    # cross_walk_ids = [info["id"] for info in crosswalks]
    # speed_bump_ids = [info["id"] for info in speed_bumps]

    # lane_ids = lane_segment_ids
    # roadline_ids = road_edge_ids + road_line_ids
    # static_map_ids = stop_sign_ids + cross_walk_ids + speed_bump_ids

    # Combine polygons by category
    lane_polygons = lane_segments
    roadline_polygons = road_edges + road_lines
    static_map_polygons = stop_signs + crosswalks + speed_bumps

    # Calculate positions and get indices
    pos_lanes = _compute_polygon_positions(lane_polygons, all_polylines)
    pos_roadlines = _compute_polygon_positions(roadline_polygons, all_polylines)
    pos_static_map = _compute_polygon_positions(static_map_polygons, all_polylines)

    index_lanes = _filter_by_distance(pos_lanes, sdc_coord[:2])
    index_roadlines = _filter_by_distance(pos_roadlines, sdc_coord[:2])
    index_static_maps = _filter_by_distance(pos_static_map, sdc_coord[:2])

    num_lanes = len(index_lanes)
    num_roadlines = len(index_roadlines)
    num_static_maps = len(index_static_maps)

    # initialization for lane points
    lane_points_position: List[Optional[np.ndarray]] = [None] * num_lanes
    lane_points_orientation: List[Optional[np.ndarray]] = [None] * num_lanes
    lane_points_type: List[Optional[np.ndarray]] = [None] * num_lanes
    lane_points_light_type: List[Optional[np.ndarray]] = [None] * num_lanes
    lane_points: List[Optional[np.ndarray]] = [None] * num_lanes
    
    lane_polylines = []
    lanes_speed_limit = []
    lanes_has_speed_limit = []
    lanes_stop_point = []
    for idx, polygon in enumerate([lane_polygons[i] for i in index_lanes]):
        polyline_index = polygon["polyline_index"]
        centerline = all_polylines[polyline_index[0]:polyline_index[1], :]
        centerline = centerline.astype(np.float32)
        
        lane_points_position[idx] = np.concatenate([centerline[:-1, :2]], axis=0)
        center_vectors = centerline[1:] - centerline[:-1]
        heading_angle = np.arctan2(center_vectors[:, 1], center_vectors[:, 0])
        lane_points_orientation[idx] = np.column_stack([np.cos(heading_angle), np.sin(heading_angle)])
        polyline_type_data = all_polylines[polyline_index[0]:polyline_index[1], 3]  # the 4th column is type
        lane_points_type[idx] = polyline_type_data[:-1].astype(np.int64)
        
        res = tf_current_light["traffic_light_states"][
            tf_current_light["traffic_lane_ids"] == polygon["id"]
        ]
        if len(res) != 0:
            lane_points_light_type[idx] = np.full_like(lane_points_type[idx], res)
        else:
            lane_points_light_type[idx] = np.full_like(lane_points_type[idx], 0)
        
        lane_points[idx] = np.column_stack(
            [
                lane_points_position[idx],
                lane_points_orientation[idx],  # now [cos, sin]
                lane_points_light_type[idx], 
                lane_points_type[idx]
            ]
        )
        lanes_speed_limit.append(polygon["speed_limit_mph"])
        if polygon["speed_limit_mph"] > 1e-5:
            lanes_has_speed_limit.append(1)
        else:
            lanes_has_speed_limit.append(0)

        stop_res = tf_current_light["traffic_stop_points"][
            tf_current_light["traffic_lane_ids"] == polygon["id"]
        ]
        if len(stop_res) != 0:
            lanes_stop_point.append(stop_res[0, :2])
        else:
            lanes_stop_point.append(np.zeros((2,), dtype=np.float32))

        polyline_len = lane_points[idx].shape[0]
        if polyline_len >= num_points_polyline:
            sampled_points = np.linspace(0, polyline_len-1, num_points_polyline, dtype=np.int32)
            cur_polyline = np.take(lane_points[idx], sampled_points, axis=0)
        else:
            cur_polyline = np.zeros((num_points_polyline, lane_points[idx].shape[1]), dtype=np.float32)
            x = np.linspace(0, polyline_len-1, num_points_polyline)
            for i in range(4):
                cur_polyline[:, i] = np.interp(x, np.arange(polyline_len), lane_points[idx][:, i])
            for i in range(4, lane_points[idx].shape[1]):
                cur_polyline[:, i] = lane_points[idx][0, i]
        lane_polylines.append(cur_polyline)
    
    # initialization for roadline points
    roadline_points_position: List[Optional[np.ndarray]] = [None] * num_roadlines
    roadline_points_orientation: List[Optional[np.ndarray]] = [None] * num_roadlines
    roadline_points_type: List[Optional[np.ndarray]] = [None] * num_roadlines

    roadline_points: List[Optional[np.ndarray]] = [None] * num_roadlines
    roadline_polylines = []
    for idx, polygon in enumerate([roadline_polygons[i] for i in index_roadlines]):
        polyline_index = polygon["polyline_index"]
        centerline = all_polylines[polyline_index[0]:polyline_index[1], :]
        centerline = centerline.astype(np.float32)
        
        roadline_points_position[idx] = np.concatenate([centerline[:-1, :2]], axis=0)
        center_vectors = centerline[1:] - centerline[:-1]
        heading_angle = np.arctan2(center_vectors[:, 1], center_vectors[:, 0])
        roadline_points_orientation[idx] = np.column_stack([np.cos(heading_angle), np.sin(heading_angle)])
        polyline_type_data = all_polylines[polyline_index[0]:polyline_index[1], 3]  # the 4th column is type
        roadline_points_type[idx] = polyline_type_data[:-1].astype(np.int64)

        roadline_points[idx] = np.column_stack(
            [
                roadline_points_position[idx],
                roadline_points_orientation[idx], 
                roadline_points_type[idx]
            ]
        )

        polyline_len = roadline_points[idx].shape[0]
        if polyline_len >= num_points_polyline:
            sampled_points = np.linspace(0, polyline_len-1, num_points_polyline, dtype=np.int32)
            cur_polyline = np.take(roadline_points[idx], sampled_points, axis=0)
        else:
            cur_polyline = np.zeros((num_points_polyline, roadline_points[idx].shape[1]), dtype=np.float32)
            x = np.linspace(0, polyline_len-1, num_points_polyline)
            for i in range(4):
                cur_polyline[:, i] = np.interp(x, np.arange(polyline_len), roadline_points[idx][:, i])
            for i in range(4, roadline_points[idx].shape[1]):
                cur_polyline[:, i] = roadline_points[idx][0, i]
        roadline_polylines.append(cur_polyline)

    # initialization for static map points
    static_map_points_position: List[Optional[np.ndarray]] = [None] * num_static_maps
    static_map_points_orientation: List[Optional[np.ndarray]] = [None] * num_static_maps
    static_map_points_type: List[Optional[np.ndarray]] = [None] * num_static_maps

    static_map_points: List[Optional[np.ndarray]] = [None] * num_static_maps
    static_map_polylines = []
    for idx, polygon in enumerate([static_map_polygons[i] for i in index_static_maps]):
        polyline_index = polygon["polyline_index"]
        centerline = all_polylines[polyline_index[0]:polyline_index[1], :]
        centerline = centerline.astype(np.float32)
        
        static_map_points_position[idx] = np.concatenate([centerline[:-1, :2]], axis=0)
        center_vectors = centerline[1:] - centerline[:-1]
        heading_angle = np.arctan2(center_vectors[:, 1], center_vectors[:, 0])
        static_map_points_orientation[idx] = np.column_stack([np.cos(heading_angle), np.sin(heading_angle)])
        polyline_type_data = all_polylines[polyline_index[0]:polyline_index[1], 3]  # the 4th column is type
        static_map_points_type[idx] = polyline_type_data[:-1].astype(np.int64)

        static_map_points[idx] = np.column_stack(
            [
                static_map_points_position[idx],
                static_map_points_orientation[idx], 
                static_map_points_type[idx]
            ]
        )

        polyline_len = static_map_points[idx].shape[0]
        if polyline_len >= num_points_static_map:
            sampled_points = np.linspace(0, polyline_len-1, num_points_static_map, dtype=np.int32)
            cur_polyline = np.take(static_map_points[idx], sampled_points, axis=0)
        else:
            cur_polyline = np.zeros((num_points_static_map, static_map_points[idx].shape[1]), dtype=np.float32)
            x = np.linspace(0, polyline_len-1, num_points_static_map)
            for i in range(4):
                cur_polyline[:, i] = np.interp(x, np.arange(polyline_len), static_map_points[idx][:, i])
            for i in range(4, static_map_points[idx].shape[1]):
                cur_polyline[:, i] = static_map_points[idx][0, i]
        static_map_polylines.append(cur_polyline)

    map_data = {}
    # map_data['map_point']: List[np.ndarray] [num_points, 5]
    # x, y, heading, traffic_light_state, type
    if len(lane_polylines) == 0:
        map_data['lanes'] = np.zeros((1, num_points_polyline, 6), dtype=np.float32)
        map_data['lanes_valid'] = np.zeros((1,), dtype=np.int32)
        map_data['lanes_speed_limit'] = np.zeros((1, ), dtype=np.float32)
        map_data['lanes_has_speed_limit'] = np.zeros((1, ), dtype=np.int32)
        map_data['lanes_stop_point'] = np.zeros((1, 2), dtype=np.float32)
    else:
        lane_polylines = np.stack(lane_polylines, axis=0).astype(np.float32)
        # lane_polylines[..., :4] = transform_coords_to_sdc_frame(lane_polylines[..., :4], sdc_coord)
        map_data['lanes'] = lane_polylines
        map_data['lanes_valid'] = np.ones((map_data['lanes'].shape[0],), dtype=np.int32)
        map_data['lanes_speed_limit'] = np.stack(lanes_speed_limit, axis=0).astype(np.float32)
        map_data['lanes_has_speed_limit'] = np.stack(lanes_has_speed_limit, axis=0).astype(np.int32)
        map_data['lanes_stop_point'] = np.stack(lanes_stop_point, axis=0).astype(np.float32)

    if len(roadline_polylines) == 0:
        map_data['roadlines'] = np.zeros((1, num_points_polyline, 5), dtype=np.float32)
        map_data['roadlines_valid'] = np.zeros((1,), dtype=np.int32)
    else:
        roadline_polylines = np.stack(roadline_polylines, axis=0).astype(np.float32)
        # roadline_polylines[..., :4] = transform_coords_to_sdc_frame(roadline_polylines[..., :4], sdc_coord)
        map_data['roadlines'] = roadline_polylines
        map_data['roadlines_valid'] = np.ones((map_data['roadlines'].shape[0],), dtype=np.int32)
    
    if len(static_map_polylines) == 0:
        map_data['static_maps'] = np.zeros((1, num_points_static_map, 5), dtype=np.float32)
        map_data['static_maps_valid'] = np.zeros((1,), dtype=np.int32)
    else:
        static_map_polylines = np.stack(static_map_polylines, axis=0).astype(np.float32)
        # static_map_polylines[..., :4] = transform_coords_to_sdc_frame(static_map_polylines[..., :4], sdc_coord)
        map_data['static_maps'] = static_map_polylines
        map_data['static_maps_valid'] = np.ones((map_data['static_maps'].shape[0],), dtype=np.int32)
    
    map_data['all_polylines'] = all_polylines
    return map_data


from collections import defaultdict


def _proto_polyline(points, feature_type, feature_id, min_points=2):
    """Return the legacy five-column polyline, or skip undersized geometry."""
    if len(points) < min_points:
        return None
    return np.asarray(
        [[point.x, point.y, point.z, feature_type, feature_id] for point in points],
        dtype=np.float32,
    )


def decode_map_features_from_proto(map_features):
    map_infos = {
        'lane': [],
        'road_line': [],
        'road_edge': [],
        'stop_sign': [],
        'crosswalk': [],
        'speed_bump': [],
        'driveway': [],
        'lane_dict': {},
        'lane2other_dict': {}
    }
    polylines = []

    point_cnt = 0
    lane2other_dict = defaultdict(list)

    for cur_data in map_features:
        cur_info = {'id': cur_data.id}
        
        if cur_data.lane.ByteSize() > 0:
            cur_polyline = _proto_polyline(
                cur_data.lane.polyline, cur_data.lane.type, cur_data.id
            )
            if cur_polyline is None:
                continue
            cur_info['speed_limit_mph'] = cur_data.lane.speed_limit_mph
            cur_info['type'] = cur_data.lane.type # 0: undefined, 1: freeway, 2: surface_street, 3: bike_lane after process
            cur_info['left_neighbors'] = [lane.feature_id for lane in cur_data.lane.left_neighbors]

            cur_info['right_neighbors'] = [lane.feature_id for lane in cur_data.lane.right_neighbors]

            cur_info['interpolating'] = cur_data.lane.interpolating
            cur_info['entry_lanes'] = list(cur_data.lane.entry_lanes)
            cur_info['exit_lanes'] = list(cur_data.lane.exit_lanes)

            cur_info['left_boundary_type'] = [x.boundary_type + 5 for x in cur_data.lane.left_boundaries]
            cur_info['right_boundary_type'] = [x.boundary_type + 5 for x in cur_data.lane.right_boundaries]

            cur_info['left_boundary'] = [x.boundary_feature_id for x in cur_data.lane.left_boundaries]
            cur_info['right_boundary'] = [x.boundary_feature_id for x in cur_data.lane.right_boundaries]
            cur_info['left_boundary_start_index'] = [lane.lane_start_index for lane in cur_data.lane.left_boundaries]
            cur_info['left_boundary_end_index'] = [lane.lane_end_index for lane in cur_data.lane.left_boundaries]
            cur_info['right_boundary_start_index'] = [lane.lane_start_index for lane in cur_data.lane.right_boundaries]
            cur_info['right_boundary_end_index'] = [lane.lane_end_index for lane in cur_data.lane.right_boundaries]

            lane2other_dict[cur_data.id].extend(cur_info['left_boundary'])
            lane2other_dict[cur_data.id].extend(cur_info['right_boundary'])

            map_infos['lane'].append(cur_info)
            map_infos['lane_dict'][cur_data.id] = cur_info

        elif cur_data.road_line.ByteSize() > 0:
            cur_polyline = _proto_polyline(
                cur_data.road_line.polyline, cur_data.road_line.type, cur_data.id
            )
            if cur_polyline is None:
                continue
            cur_info['type'] = cur_data.road_line.type
            map_infos['road_line'].append(cur_info)

        elif cur_data.road_edge.ByteSize() > 0:
            cur_polyline = _proto_polyline(
                cur_data.road_edge.polyline, cur_data.road_edge.type, cur_data.id
            )
            if cur_polyline is None:
                continue
            cur_info['type'] = cur_data.road_edge.type
            map_infos['road_edge'].append(cur_info)

        elif cur_data.stop_sign.ByteSize() > 0:
            cur_info['lane_ids'] = list(cur_data.stop_sign.lane)
            for i in cur_info['lane_ids']:
                lane2other_dict[i].append(cur_data.id)
            point = cur_data.stop_sign.position
            cur_info['position'] = np.array([point.x, point.y, point.z])
            # global_type = polyline_type['TYPE_STOP_SIGN']
            global_type = polyline_type['TYPE_STOP_SIGN'] - 17
            cur_polyline = np.array([point.x, point.y, point.z, global_type, cur_data.id]).reshape(1, 5)
            cur_info['type'] = 0
            map_infos['stop_sign'].append(cur_info)
        elif cur_data.crosswalk.ByteSize() > 0:
            global_type = polyline_type['TYPE_CROSSWALK'] - 17
            cur_polyline = _proto_polyline(
                cur_data.crosswalk.polygon, global_type, cur_data.id, min_points=3
            )
            if cur_polyline is None:
                continue
            map_infos['crosswalk'].append(cur_info)

        elif cur_data.speed_bump.ByteSize() > 0:
            global_type = polyline_type['TYPE_SPEED_BUMP'] - 17
            cur_polyline = _proto_polyline(
                cur_data.speed_bump.polygon, global_type, cur_data.id, min_points=3
            )
            if cur_polyline is None:
                continue
            map_infos['speed_bump'].append(cur_info)

        elif hasattr(cur_data, 'driveway') and cur_data.driveway.ByteSize() > 0:
            cur_polyline = _proto_polyline(
                cur_data.driveway.polygon, 0, cur_data.id, min_points=3
            )
            if cur_polyline is None:
                continue
            cur_info['type'] = 0
            map_infos['driveway'].append(cur_info)

        else:
            # print(cur_data)
            continue
        polylines.append(cur_polyline)
        cur_info['polyline_index'] = (point_cnt, point_cnt + len(cur_polyline))
        point_cnt += len(cur_polyline)

    map_infos['all_polylines'] = (
        np.concatenate(polylines, axis=0) if polylines
        else np.empty((0, 5), dtype=np.float32)
    )
    map_infos['lane2other_dict'] = lane2other_dict
    return map_infos


def process_agents(
        scenario,
        max_num_objects=64,
        num_steps=91,
        current_index=10,  # current timestamp
        remove_history=False,
    ):
    tracks = scenario.tracks
    sdc_idx = scenario.sdc_track_index

    tracks_to_predict = scenario.tracks_to_predict
    tracks_to_predict_idx = [t.track_index for t in tracks_to_predict if t.track_index != sdc_idx]

    # select agents based on distance to sdc
    sdc_coord = np.array(
        [
            tracks[sdc_idx].states[current_index].center_x, 
            tracks[sdc_idx].states[current_index].center_y,
            tracks[sdc_idx].states[current_index].heading
        ], dtype=np.float32
    )

    valid_indices = [i for i, t in enumerate(tracks) if t.states[current_index].valid]

    agents_positions = []
    agents_velocities = []
    for idx in valid_indices:
        cur_data = tracks[idx]
        agents_positions.append(
            [
                cur_data.states[current_index].center_x,
                cur_data.states[current_index].center_y
            ]
        )
        agents_velocities.append(
            [
                cur_data.states[current_index].velocity_x,
                cur_data.states[current_index].velocity_y
            ]
        )
    
    distance_to_sdc = np.linalg.norm(
        np.array(agents_positions) - sdc_coord[:2], axis=-1
    )
    
    velocity_magnitude = np.linalg.norm(np.array(agents_velocities), axis=-1)
    distance_to_sdc[velocity_magnitude < 1e-5] += 100

    # agents_idx = np.argsort(distance_to_sdc)[:max_num_objects]
    # agents_idx = np.sort(agents_idx)

    sorted_idx = np.argsort(distance_to_sdc)     
    sorted_original_idx = [valid_indices[idx] for idx in sorted_idx]
    
    remaining_idx = [idx for idx in sorted_original_idx if idx not in [sdc_idx] + tracks_to_predict_idx]
    combined_idx = [sdc_idx] + tracks_to_predict_idx + remaining_idx
    
    # agents_idx = np.array(combined_idx[:max_num_objects])
    agents_idx = combined_idx[:max_num_objects]
    agents_idx_remaining = []
    if len(combined_idx) > max_num_objects:
        agents_idx_remaining = combined_idx[max_num_objects:]
    
    agents_history = np.zeros((max_num_objects, current_index+1, 9), dtype=np.float32)
    agents_type = np.zeros((max_num_objects,), dtype=np.int32)
    agents_interested = np.zeros((max_num_objects,), dtype=np.int32)
    agents_future = np.zeros((max_num_objects, num_steps-current_index, 9), dtype=np.float32)
    agents_id = np.ones((max_num_objects,), dtype=np.int32)*(-1)
    
    # agents_idx_list = agents_idx.tolist()
    for i, cur_data in enumerate([tracks[idx] for idx in agents_idx]):
        agent_type = cur_data.object_type
        agent_id = cur_data.id
        valid = cur_data.states[current_index].valid
        
        # leading to discontinuous result array
        if not valid:
            agents_interested[i] = 0
            continue
        
        if cur_data.id in scenario.objects_of_interest:
            agents_interested[i] = 10
        else:
            agents_interested[i] = 1
        
        agents_type[i] = agent_type
        agents_id[i] = agent_id
        
        step_state = []
        step_valid = []
        for s in cur_data.states:
            step_state.append(
                [
                    s.center_x,
                    s.center_y,
                    np.cos(s.heading),
                    np.sin(s.heading),
                    s.velocity_x,
                    s.velocity_y,
                    s.length,
                    s.width,
                    # s.height,
                    s.center_z,
                ]
            )
            step_valid.append(s.valid)

        step_state = np.array(step_state, dtype=np.float32)
        step_valid = np.array(step_valid, dtype=bool)
        
        agents_history[i] = step_state[:current_index+1]
        agents_history[i][~step_valid[:current_index+1]] = 0
        if step_state.shape[0] < num_steps:
            continue
        else:
            agents_future[i] = step_state[current_index:]
        agents_future[i][~step_valid[current_index:]] = 0

    agents_history_remaining = np.zeros((len(agents_idx_remaining), current_index+1, 9), dtype=np.float32)
    agents_id_remaining = np.ones((len(agents_idx_remaining),), dtype=np.int32)*(-1)
    for i, cur_data in enumerate([tracks[idx] for idx in agents_idx_remaining]):
        agent_id = cur_data.id
        agents_id_remaining[i] = agent_id

        step_state = []
        step_valid = []
        for s in cur_data.states:
            step_state.append(
                [
                    s.center_x,
                    s.center_y,
                    np.cos(s.heading),
                    np.sin(s.heading),
                    s.velocity_x,
                    s.velocity_y,
                    s.length,
                    s.width,
                    # s.height,
                    s.center_z,
                ]
            )
            step_valid.append(s.valid)
        
        step_state = np.array(step_state, dtype=np.float32)
        step_valid = np.array(step_valid, dtype=bool)

        agents_history_remaining[i] = step_state[:current_index+1]
        agents_history_remaining[i][~step_valid[:current_index+1]] = 0

    agents_future_valid = np.not_equal(
        np.sum(agents_future, axis=-1), 0
    )

    if remove_history:
        agents_history[:, :-1] = 0
    
    return {
        'history': agents_history,
        'future': agents_future,
        'future_valid': agents_future_valid,
        'interested': agents_interested,
        'type': agents_type,
        'ids': agents_id,
        'history_remaining': agents_history_remaining,
        'ids_remaining': agents_id_remaining,
        'sdc_coord': sdc_coord,
    }

def process_traffic_lights(
        dynamic_map_states,
        current_index=10,
    ):
    s = dynamic_map_states[current_index]
    lane_id, state, stop_point, has_stop_point = [], [], [], []
    for cur_signal in s.lane_states:  # (num_observed_signals)
        lane_id.append(cur_signal.lane)
        state.append(cur_signal.state)
        stop_point.append(
            [
                cur_signal.stop_point.x, cur_signal.stop_point.y
            ]
        )
        has_stop_point.append(cur_signal.HasField("stop_point"))

    traffic_lane_ids = np.array(lane_id, dtype=np.int32)  # [num_observed_signals]
    traffic_light_states = np.array(state, dtype=np.int32)  # [num_observed_signals]
    traffic_stop_points = np.array(stop_point).reshape(-1, 2)  # [num_observed_signals, 2]
    
    # here the process of catk (data_preprocess.py line 211-225) is ignored
    # the detailed traffic light states are saved
    traffic_light_points = np.concatenate(
        [traffic_stop_points, traffic_light_states[:, None]], axis=1
        )
    traffic_light_points = np.float32(traffic_light_points)

    # Truncation/padding deferred to data loading (collate_fn)

    return {
        'traffic_light_points': traffic_light_points,
        'traffic_lane_ids': traffic_lane_ids,
        'traffic_light_states': traffic_light_states,
        'traffic_stop_points': traffic_stop_points,
        'traffic_has_stop_points': np.asarray(has_stop_point, dtype=np.bool_),
    }

def process_roadgraph(
        scenario,
        traffic_light_data,
        map_segment_length_m=MAP_SEGMENT_LENGTH_M,
    ):
    """
    Process roadgraph data.
    
    Args:
        scenario: scenario data.
        traffic_light_data: traffic light data.
        
    Returns:
        roadgraph data.
    """
    map_infos = decode_map_features_from_proto(scenario.map_features)
    if len(map_infos['all_polylines']) == 0:
        print(f'empty polylines scenario id: {scenario.scenario_id}')
    return build_smart_map_tokens(
        map_infos=map_infos,
        traffic_light=traffic_light_data,
        segment_length_m=map_segment_length_m,
    )


def data_process_scenario(
        scenario: scenario_pb2.Scenario,
        max_num_objects: int=64,
        current_index: int=10,
        map_segment_length_m: float=MAP_SEGMENT_LENGTH_M,
        remove_history: bool=False,
        split: str='train',
    ) -> Dict[str, Any]:
    data = {}

    # if split == 'validation':
    #     agents_data = process_agents(
    #         scenario,
    #         max_num_objects=256,
    #         num_steps=91,
    #         current_index=current_index,
    #         select_agents=select_agents,
    #         remove_history=remove_history,
    #     )
    # else:
    agents_data = process_agents(
        scenario,
        max_num_objects=max_num_objects,
        num_steps=91,
        current_index=current_index,
        remove_history=remove_history,
    )
    
    traffic_light_data = process_traffic_lights(
        scenario.dynamic_map_states,
        current_index=current_index,
    )
    
    roadgraph_data = process_roadgraph(
        scenario, 
        traffic_light_data, 
        map_segment_length_m=map_segment_length_m,
    )

    # agents_data['history'][..., :6] = transform_coords_to_sdc_frame(agents_data['history'][..., :6], agents_data['sdc_coord'])

    data = {
        'sdc_coord': agents_data['sdc_coord'],
        'agents_history': agents_data['history'],
        'agents_future': agents_data['future'],
        'agents_future_valid': agents_data['future_valid'],
        'agents_interested': agents_data['interested'],
        'agents_type': agents_data['type'],
        'traffic_light_points': traffic_light_data['traffic_light_points'],
        **roadgraph_data,
        'agents_id': agents_data['ids'],
        'agents_history_remaining': agents_data['history_remaining'],
        'agents_id_remaining': agents_data['ids_remaining'],
    }
    return data

def wm2dp(
        file_path, 
        split, 
        output_dir,
        map_segment_length_m=MAP_SEGMENT_LENGTH_M,
    ):
    import tensorflow as tf
    from waymo_open_dataset.protos import scenario_pb2

    dataset = tf.data.TFRecordDataset(
        file_path, compression_type="", num_parallel_reads=3
    )

    for tf_data in dataset:
        tf_data = tf_data.numpy()
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(bytes(tf_data))

        data_dict = data_process_scenario(
            scenario,
            max_num_objects=MAX_NUM_OBJECTS,
            current_index=CURRENT_INDEX,
            map_segment_length_m=map_segment_length_m,
            split=split,
        )
        
        scenario_id = scenario.scenario_id
        data_dict["scenario_id"] = scenario_id
        with open(output_dir / f"{scenario_id}.pkl", "wb+") as f:
            pickle.dump(data_dict, f)

        # break

def batch_process9s_transformer(
        input_dir, output_dir, split, num_workers,
        map_segment_length_m=MAP_SEGMENT_LENGTH_M):
    if map_segment_length_m <= 0:
        raise ValueError("map segment length must be positive")
    output_dir = Path(output_dir)
    output_dir = output_dir / split
    output_dir.mkdir(exist_ok=True, parents=True)

    input_dir = Path(input_dir) / split
    packages = sorted([p.as_posix() for p in input_dir.glob("*")])
    packages = packages[:100]

    func = partial(
        wm2dp,
        split=split,
        output_dir=output_dir,
        map_segment_length_m=map_segment_length_m,
    )

    with multiprocessing.Pool(num_workers) as p:
        r = list(tqdm(p.imap_unordered(func, packages), total=len(packages)))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/root/workspace/data/womd/uncompressed/scenario",
    )
    parser.add_argument(
        "--output_dir", type=str, default="/root/workspace/data/SMART_new"
    )
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--map_segment_length_m", type=float, default=MAP_SEGMENT_LENGTH_M
    )
    args = parser.parse_args()

    batch_process9s_transformer(
        args.input_dir, args.output_dir, args.split, num_workers=args.num_workers,
        map_segment_length_m=args.map_segment_length_m,
    )
