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

import multiprocessing
import pickle
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
# import pandas as pd
import tensorflow as tf
import torch
from scipy.interpolate import interp1d
from tqdm import tqdm
from waymo_open_dataset.protos import scenario_pb2

from src.vbd.data_preprocess.utils import wrap_angle, wrap_to_pi
from src.dp.data_preprocess.utils import transform_coords_to_sdc_frame

MAX_NUM_OBJECTS = 64
CURRENT_INDEX = 10
NUM_POINTS_POLYLINE = 30
NUM_POINTS_STATIC_MAP = 10


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

def decode_map_features_from_proto(map_features):
    map_infos = {
        'lane': [],
        'road_line': [],
        'road_edge': [],
        'stop_sign': [],
        'crosswalk': [],
        'speed_bump': [],
        'lane_dict': {},
        'lane2other_dict': {}
    }
    polylines = []

    point_cnt = 0
    lane2other_dict = defaultdict(list)

    for cur_data in map_features:
        cur_info = {'id': cur_data.id}
        
        if cur_data.lane.ByteSize() > 0:
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

            global_type = cur_info['type']
            cur_polyline = np.stack(
                [np.array([point.x, point.y, point.z, global_type, cur_data.id]) for point in cur_data.lane.polyline],
                axis=0)
            cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline[:, 3:]), axis=-1)
            if cur_polyline.shape[0] <= 1:
                continue
            map_infos['lane'].append(cur_info)
            map_infos['lane_dict'][cur_data.id] = cur_info

        elif cur_data.road_line.ByteSize() > 0:
            # cur_info['type'] = cur_data.road_line.type + 5
            cur_info['type'] = cur_data.road_line.type

            global_type = cur_info['type']
            cur_polyline = np.stack([np.array([point.x, point.y, point.z, global_type, cur_data.id]) for point in
                                     cur_data.road_line.polyline], axis=0)
            cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline[:, 3:]), axis=-1)
            if cur_polyline.shape[0] <= 1:
                continue
            map_infos['road_line'].append(cur_info)

        elif cur_data.road_edge.ByteSize() > 0:
            # cur_info['type'] = cur_data.road_edge.type + 14
            cur_info['type'] = cur_data.road_edge.type + 9

            global_type = cur_info['type']
            cur_polyline = np.stack([np.array([point.x, point.y, point.z, global_type, cur_data.id]) for point in
                                     cur_data.road_edge.polyline], axis=0)
            cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline[:, 3:]), axis=-1)
            if cur_polyline.shape[0] <= 1:
                continue
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
            if cur_polyline.shape[0] <= 1:
                continue
            map_infos['stop_sign'].append(cur_info)
        elif cur_data.crosswalk.ByteSize() > 0:
            # global_type = polyline_type['TYPE_CROSSWALK']
            global_type = polyline_type['TYPE_CROSSWALK'] - 17
            cur_polyline = np.stack([np.array([point.x, point.y, point.z, global_type, cur_data.id]) for point in
                                     cur_data.crosswalk.polygon], axis=0)
            cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline[:, 3:]), axis=-1)
            if cur_polyline.shape[0] <= 1:
                continue
            map_infos['crosswalk'].append(cur_info)

        elif cur_data.speed_bump.ByteSize() > 0:
            # global_type = polyline_type['TYPE_SPEED_BUMP']
            global_type = polyline_type['TYPE_SPEED_BUMP'] - 17
            cur_polyline = np.stack([np.array([point.x, point.y, point.z, global_type, cur_data.id]) for point in
                                     cur_data.speed_bump.polygon], axis=0)
            cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline[:, 3:]), axis=-1)
            if cur_polyline.shape[0] <= 1:
                continue
            map_infos['speed_bump'].append(cur_info)

        else:
            # print(cur_data)
            continue
        polylines.append(cur_polyline)
        cur_info['polyline_index'] = (point_cnt, point_cnt + len(cur_polyline))
        point_cnt += len(cur_polyline)

    try:
        polylines = np.concatenate(polylines, axis=0).astype(np.float32)
    except:
        polylines = np.zeros((0, 8), dtype=np.float32)
        # print('Empty polylines: ')
    map_infos['all_polylines'] = polylines  # (num_polylines, 8)
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
    lane_id, state, stop_point = [], [], []
    for cur_signal in s.lane_states:  # (num_observed_signals)
        lane_id.append(cur_signal.lane)
        state.append(cur_signal.state)
        stop_point.append(
            [
                cur_signal.stop_point.x, cur_signal.stop_point.y
            ]
        )

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
        'traffic_stop_points': traffic_stop_points
    }

def process_roadgraph(
        scenario,
        traffic_light_data,
        sdc_coord,
        num_points_polyline=30,
        num_points_static_map=10,
    ):
    """
    Process roadgraph data.
    
    Args:
        scenario: scenario data.
        traffic_light_data: traffic light data.
        sdc_coord: sdc coordinate.
        num_points_polyline: number of points per polyline.
        num_points_static_map: number of points per static map.
        
    Returns:
        roadgraph data.
    """
    map_infos = decode_map_features_from_proto(scenario.map_features)
    if np.all(map_infos['all_polylines']==0):
        print(f'empty polylines scenario id: {scenario.scenario_id}')

    map_data = get_map_features(
        map_infos,
        traffic_light_data,
        sdc_coord,
        num_points_polyline=num_points_polyline,
        num_points_static_map=num_points_static_map,
    )

    lanes = map_data['lanes']
    lanes_valid = map_data['lanes_valid']
    lanes_speed_limit = map_data['lanes_speed_limit']
    lanes_has_speed_limit = map_data['lanes_has_speed_limit']
    lanes_stop_point = map_data['lanes_stop_point']
    
    roadlines = map_data['roadlines']
    roadlines_valid = map_data['roadlines_valid']
    static_maps = map_data['static_maps']
    static_maps_valid = map_data['static_maps_valid']

    # Truncation/padding deferred to data loading (collate_fn)

    return {
        'lanes': lanes,  # [num_lanes, num_points_polyline, 6]
        'lanes_valid': lanes_valid,  # [num_lanes,]
        'lanes_speed_limit': lanes_speed_limit,  # [num_lanes,]
        'lanes_has_speed_limit': lanes_has_speed_limit,  # [num_lanes,]
        'lanes_stop_point': lanes_stop_point,  # [num_lanes, 2]
        'roadlines': roadlines,  # [num_roadlines, num_points_polyline, 5]
        'roadlines_valid': roadlines_valid,  # [num_roadlines,]
        'static_maps': static_maps,  # [num_static_maps, num_points_static_map, 5]
        'static_maps_valid': static_maps_valid  # [num_static_maps,]
    }


def data_process_scenario(
        scenario: scenario_pb2.Scenario,
        max_num_objects: int=64,
        current_index: int=10,
        num_points_polyline: int=30,
        num_points_static_map: int=10,
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
    )
    
    roadgraph_data = process_roadgraph(
        scenario, 
        traffic_light_data, 
        agents_data['sdc_coord'], # sdc position
        num_points_polyline=num_points_polyline,
        num_points_static_map=num_points_static_map,
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
        'lanes': roadgraph_data['lanes'],
        'lanes_valid': roadgraph_data['lanes_valid'],
        'lanes_speed_limit': roadgraph_data['lanes_speed_limit'],
        'lanes_has_speed_limit': roadgraph_data['lanes_has_speed_limit'],
        'lanes_stop_point': roadgraph_data['lanes_stop_point'],
        'roadlines': roadgraph_data['roadlines'],
        'roadlines_valid': roadgraph_data['roadlines_valid'],
        'static_maps': roadgraph_data['static_maps'],
        'static_maps_valid': roadgraph_data['static_maps_valid'],
        'agents_id': agents_data['ids'],
        'agents_history_remaining': agents_data['history_remaining'],
        'agents_id_remaining': agents_data['ids_remaining'],
    }
    return data

def wm2dp(
        file_path, 
        split, 
        output_dir
    ):
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
            num_points_polyline=NUM_POINTS_POLYLINE,
            num_points_static_map=NUM_POINTS_STATIC_MAP,
            split=split,
        )
        
        scenario_id = scenario.scenario_id
        data_dict["scenario_id"] = scenario_id
        with open(output_dir / f"{scenario_id}.pkl", "wb+") as f:
            pickle.dump(data_dict, f)

        # break

def batch_process9s_transformer(input_dir, output_dir, split, num_workers):
    output_dir = Path(output_dir)
    output_dir = output_dir / split
    output_dir.mkdir(exist_ok=True, parents=True)

    input_dir = Path(input_dir) / split
    packages = sorted([p.as_posix() for p in input_dir.glob("*")])
    packages = packages[:200]

    func = partial(
        wm2dp,
        split=split,
        output_dir=output_dir,
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
    args = parser.parse_args()

    batch_process9s_transformer(
        args.input_dir, args.output_dir, args.split, num_workers=args.num_workers
    )