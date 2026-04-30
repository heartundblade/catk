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
from src.vbd.data_preprocess.utils import get_polylines_from_polygon, preprocess_map

MAX_NUM_OBJECTS = 64
MAX_POLYLINES = 256
MAX_TRAFFIC_LIGHTS = 16
CURRENT_INDEX = 10
NUM_POINTS_POLYLINE = 30

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

_point_types = ['DASH_SOLID_YELLOW', 'DASH_SOLID_WHITE', 'DASHED_WHITE', 'DASHED_YELLOW',
                'DOUBLE_SOLID_YELLOW', 'DOUBLE_SOLID_WHITE', 'DOUBLE_DASH_YELLOW', 'DOUBLE_DASH_WHITE',
                'SOLID_YELLOW', 'SOLID_WHITE', 'SOLID_DASH_WHITE', 'SOLID_DASH_YELLOW', 'EDGE',
                'NONE', 'UNKNOWN', 'CROSSWALK', 'CENTERLINE']

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
    diff = pos_polygons[:, :, None, :] - reference_points[None, None, :, :]

    distances = np.linalg.norm(diff, axis=-1)
    min_dist_to_agents = distances.min(axis=1)
    final_min_dist = min_dist_to_agents.min(axis=1)

    index = np.argsort(final_min_dist)
    return index

def get_map_features(
        map_infos, 
        tf_current_light, 
        agents_data,
        max_polylines=256,
        num_points_polyline=30,
    ):
    """
    Get polylines pos, heading, traffic light state and type from
    map_infos, filtered by agent_data.
    """
    lane_segments = map_infos['lane']
    all_polylines = map_infos["all_polylines"]
    crosswalks = map_infos['crosswalk']
    road_edges = map_infos['road_edge']
    road_lines = map_infos['road_line']

    lane_segment_ids = [info["id"] for info in lane_segments]
    cross_walk_ids = [info["id"] for info in crosswalks]
    road_edge_ids = [info["id"] for info in road_edges]
    road_line_ids = [info["id"] for info in road_lines]
    polygon_ids = lane_segment_ids + road_edge_ids + road_line_ids + cross_walk_ids
    num_polygons = len(lane_segment_ids) + len(road_edge_ids) + len(road_line_ids) + len(cross_walk_ids)

    pos_polygons = np.zeros([num_polygons, 3, 2])
    polygons = lane_segments + road_edges + road_lines + crosswalks
    for i, polygon in enumerate(polygons):
        polyline_index = polygon["polyline_index"]
        pos_polygons[i, 0, :] = all_polylines[polyline_index[0], :2]
        pos_polygons[i, 1, :] = all_polylines[polyline_index[1]-1, :2]
        pos_polygons[i, 2, :] = np.mean(all_polylines[polyline_index[0]:polyline_index[1], :2], axis=0)

    if num_polygons > max_polylines:
        index = sort_polygons_by_distance(
            pos_polygons, agents_data
        )
        index = index[:max_polylines]
    else:
        index = np.arange(num_polygons)

    # initialization
    points_position: List[Optional[np.ndarray]] = [None] * num_polygons
    points_orientation: List[Optional[np.ndarray]] = [None] * num_polygons
    points_type: List[Optional[np.ndarray]] = [None] * num_polygons
    points_light_type: List[Optional[np.ndarray]] = [None] * num_polygons

    points: List[Optional[np.ndarray]] = [None] * num_polygons
    polylines = []
    for polygon in [polygons[i] for i in index]:
        lane_segment_idx = polygon_ids.index(polygon["id"])
        polyline_index = polygon["polyline_index"]
        centerline = all_polylines[polyline_index[0]:polyline_index[1], :]
        centerline = centerline.astype(np.float32)
        # centerline = torch.from_numpy(centerline).float()
        
        # points_position[lane_segment_idx] = torch.cat([centerline[:-1, :dim]], dim=0)
        points_position[lane_segment_idx] = np.concatenate([centerline[:-1, :2]], axis=0)
        center_vectors = centerline[1:] - centerline[:-1]
        # points_orientation[lane_segment_idx] = torch.cat([torch.atan2(center_vectors[:, 1], center_vectors[:, 0])], dim=0)
        points_orientation[lane_segment_idx] = np.concatenate([np.arctan2(center_vectors[:, 1], center_vectors[:, 0])], axis=0)
        polyline_type_data = all_polylines[polyline_index[0]:polyline_index[1], 3]  # the 4th column is type
        # points_type[lane_segment_idx] = torch.from_numpy(polyline_type_data[:-1]).long() 
        points_type[lane_segment_idx] = polyline_type_data[:-1].astype(np.int64)
        
        if polygon["id"] in lane_segment_ids:
            res = tf_current_light["traffic_light_states"][
                tf_current_light["traffic_lane_ids"] == polygon["id"]
            ]
            if len(res) != 0:
                # point_light_type[lane_segment_idx] = torch.full_like(point_type[lane_segment_idx], res)
                points_light_type[lane_segment_idx] = np.full_like(points_type[lane_segment_idx], res)
            else:
                # point_light_type[lane_segment_idx] = torch.full_like(point_type[lane_segment_idx], 0)
                points_light_type[lane_segment_idx] = np.full_like(points_type[lane_segment_idx], 0)
        else:
            # point_light_type[lane_segment_idx] = torch.full_like(point_type[lane_segment_idx], 0)
            points_light_type[lane_segment_idx] = np.full_like(points_type[lane_segment_idx], 0)

        points[lane_segment_idx] = np.column_stack(
            [
                points_position[lane_segment_idx],
                points_orientation[lane_segment_idx], 
                points_light_type[lane_segment_idx], 
                points_type[lane_segment_idx]
            ]
        )

        polyline_len = points[lane_segment_idx].shape[0]
        sampled_points = np.linspace(0, polyline_len-1, num_points_polyline, dtype=np.int32)
        cur_polyline = np.take(points[lane_segment_idx], sampled_points, axis=0)
        polylines.append(cur_polyline)
    # num_points = torch.tensor([point.size(0) for point in point_position], dtype=torch.long)
    # num_points = np.array([point.shape[0] for point in point_position])
    
    map_data = {}
    # map_data['map_point']: List[np.ndarray] [num_points, 5]
    # x, y, heading, traffic_light_state, type
    if len(points_position) == 0:
        # map_data['map_point'] = torch.zeros([0, 5], dtype=torch.float)
        # map_data['map_point']['position'] = torch.tensor([], dtype=torch.float)
        # map_data['map_point']['orientation'] = torch.tensor([], dtype=torch.float)
        # map_data['map_point']['type'] = torch.tensor([], dtype=torch.int32)
        map_data['polylines'] = np.zeros((1, num_points_polyline, 5), dtype=np.float32)
        map_data['polylines_valid'] = np.zeros((1,), dtype=np.int32)
    else:
        # points_position = torch.cat(points_position, dim=0)
        # points_orientation = torch.cat(points_orientation, dim=0)
        # points_tf_light_state = torch.cat(points_light_type, dim=0)
        # points_type = torch.cat(points_type, dim=0)
        # points_position = np.concatenate(points_position, axis=0)
        # points_orientation = np.concatenate(points_orientation, axis=0)
        # points_tf_light_state = np.concatenate(points_light_type, axis=0)
        # points_type = np.concatenate(points_type, axis=0)
        
        # map_data['map_point'] = torch.cat(
        #     [points_position, points_orientation, points_tf_light_state, points_type], dim=0
        # )

        map_data['polylines'] = np.stack(polylines, axis=0).astype(np.float32)
        map_data['polylines_valid'] = np.ones((map_data['polylines'].shape[0],), dtype=np.int32)
    
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
            cur_info['type'] = cur_data.lane.type + 1  # 1: undefined, 2: freeway, 3: surface_street, 4: bike_lane after process
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
            cur_info['type'] = cur_data.road_line.type + 5

            global_type = cur_info['type']
            cur_polyline = np.stack([np.array([point.x, point.y, point.z, global_type, cur_data.id]) for point in
                                     cur_data.road_line.polyline], axis=0)
            cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline[:, 3:]), axis=-1)
            if cur_polyline.shape[0] <= 1:
                continue
            map_infos['road_line'].append(cur_info)

        elif cur_data.road_edge.ByteSize() > 0:
            cur_info['type'] = cur_data.road_edge.type + 14

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

            global_type = polyline_type['TYPE_STOP_SIGN']
            cur_polyline = np.array([point.x, point.y, point.z, global_type, cur_data.id]).reshape(1, 5)
            if cur_polyline.shape[0] <= 1:
                continue
            map_infos['stop_sign'].append(cur_info)
        elif cur_data.crosswalk.ByteSize() > 0:
            global_type = polyline_type['TYPE_CROSSWALK']
            cur_polyline = np.stack([np.array([point.x, point.y, point.z, global_type, cur_data.id]) for point in
                                     cur_data.crosswalk.polygon], axis=0)
            cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline[:, 3:]), axis=-1)
            if cur_polyline.shape[0] <= 1:
                continue
            map_infos['crosswalk'].append(cur_info)

        elif cur_data.speed_bump.ByteSize() > 0:
            global_type = polyline_type['TYPE_SPEED_BUMP']
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
        print('Empty polylines: ')
    map_infos['all_polylines'] = polylines  # (num_polylines, 8)
    map_infos['lane2other_dict'] = lane2other_dict
    return map_infos


def process_agents(
        scenario,
        max_num_objects=64,
        num_steps=91,
        current_index=10,  # current timestamp
        select_agents=None,
        remove_history=False,
    ):
    tracks = scenario.tracks
    sdc_idx = scenario.sdc_track_index

    # select agents based on distance to sdc
    sdc_position = np.array(
        [
            tracks[sdc_idx].states[current_index].center_x, 
            tracks[sdc_idx].states[current_index].center_y
        ]
    )
    agents_positions = []
    if select_agents is None:
        for i, cur_data in enumerate(tracks):
            agents_positions.append(
                [
                    cur_data.states[current_index].center_x,
                    cur_data.states[current_index].center_y
                ]
            )
        distance_to_sdc = np.linalg.norm(
            np.array(agents_positions) - sdc_position, axis=-1
            )
        agents_idx = np.argsort(distance_to_sdc)[:max_num_objects]
        agents_idx = np.sort(agents_idx)
    else:
        agents_idx = select_agents


    agents_history = np.zeros((max_num_objects, current_index+1, 9), dtype=np.float32)
    agents_type = np.zeros((max_num_objects,), dtype=np.int32)
    agents_interested = np.zeros((max_num_objects,), dtype=np.int32)
    agents_future = np.zeros((max_num_objects, num_steps-current_index, 9), dtype=np.float32)
    agents_id = np.zeros((max_num_objects,), dtype=np.int32)
    
    agents_idx_list = agents_idx.tolist()
    for i, cur_data in enumerate([tracks[idx] for idx in agents_idx_list]):
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
                    s.heading,
                    s.velocity_x,
                    s.velocity_y,
                    s.length,
                    s.width,
                    s.height,
                    s.center_z,
                ]
            )
            step_valid.append(s.valid)

        step_state = np.array(step_state, dtype=np.float32)
        step_valid = np.array(step_valid, dtype=bool)
        
        agents_history[i] = step_state[:current_index+1]
        agents_history[i][~step_valid[:current_index+1]] = 0
        if step_state.shape[0]<num_steps:
            continue
        else:
            agents_future[i] = step_state[current_index:]
        agents_future[i][~step_valid[current_index:]] = 0
    
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
        'ids': agents_id
    }

def process_traffic_lights(
        dynamic_map_states,
        max_num_traffic_lights=20,
        current_index=10,
        ):
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

    num_traffic_lights = traffic_light_points.shape[0]
    if num_traffic_lights >= max_num_traffic_lights:
        traffic_light_points = traffic_light_points[:max_num_traffic_lights]
    else:
        traffic_light_points = np.pad(
            traffic_light_points, 
            ((0, max_num_traffic_lights-num_traffic_lights), (0, 0))
        )

    return {
        'traffic_light_points': traffic_light_points,
        'traffic_lane_ids': traffic_lane_ids,
        'traffic_light_states': traffic_light_states
    }

def process_roadgraph(
        scenario,
        traffic_light_data,
        agents_pos,
        max_polylines=256,
        num_points_polyline=30,
    ):
    """
    Process roadgraph data.
    
    Args:
        scenario: scenario data.
        traffic_light_data: traffic light data.
        agents_pos: agent positions.
        max_polines: maximum number of polylines.
        num_points_polyline: number of points per polyline.
        
    Returns:
        roadgraph data.
    """
    map_infos = decode_map_features_from_proto(scenario.map_features)
    if np.all(map_infos['all_polylines']==0):
        print(f'empty polylines scenario id: {scenario.scenario_id}')

    map_data = get_map_features(
        map_infos,
        traffic_light_data,
        agents_pos,
        max_polylines=max_polylines,
        num_points_polyline=num_points_polyline,
    )

    polylines = map_data['polylines']
    polylines_valid = map_data['polylines_valid']

    # post-process polylines
    # if len(polylines) > 0:
    #     polylines = np.stack(polylines, axis=0)
    #     polylines_valid = np.ones((polylines.shape[0],), dtype=np.int32)
    # else:
    #     polylines = np.zeros((1, num_points_polyline, 5), dtype=np.float32)
    #     polylines_valid = np.zeros((1,), dtype=np.int32)
    
    if polylines.shape[0] >= max_polylines:
        polylines = polylines[:max_polylines]
        polylines_valid = polylines_valid[:max_polylines]
    else:
        polylines = np.pad(polylines, ((0, max_polylines-polylines.shape[0]), (0, 0), (0, 0)))
        polylines_valid = np.pad(polylines_valid, (0, max_polylines-polylines_valid.shape[0]))
    
    return {
        'polylines': polylines,  # [num_polylines, num_points_polyline, 5]
        'polylines_valid': polylines_valid  # [num_polylines,]
    }

def calculate_relations(
        agents_history, 
        polylines, 
        traffic_light_points
    ):
    """
    Calculate the relations between agents, polylines, and traffic lights.
    
    Args:
        agents_history(numpy.ndarray): Array of agent positions and orientations. [x, y, heading]
        polylines(numpy.ndarray): Array of polyline positions.
        traffic_light_points(numpy.ndarray): Array of traffic light positions.
        
    Returns:
        numpy.ndarray: Array of relations between elements in the scene.
    """
    n_agents = agents_history.shape[0]
    n_polylines = polylines.shape[0]
    n_traffic_lights = traffic_light_points.shape[0]
    n = n_agents + n_polylines + n_traffic_lights
    
    # Prepare a single array to hold all elements
    all_elements = np.concatenate([
        agents_history[:, -1, :3],
        polylines[:, 0, :3],
        np.concatenate([traffic_light_points[:, :2], np.zeros((n_traffic_lights, 1))], axis=1)
    ], axis=0)
    
    # Compute pairwise differences using broadcasting
    pos_diff = all_elements[:, :2][:, None, :] - all_elements[:, :2][None, :, :]
    
    # Compute local position and angle differences
    cos_theta = np.cos(all_elements[:, 2])[:, None]
    sin_theta = np.sin(all_elements[:, 2])[:, None]
    local_pos_x = pos_diff[..., 0] * cos_theta + pos_diff[..., 1] * sin_theta
    local_pos_y = -pos_diff[..., 0] * sin_theta + pos_diff[..., 1] * cos_theta
    theta_diff = wrap_to_pi(all_elements[:, 2][:, None] - all_elements[:, 2][None, :])
    
    # Set theta_diff to zero for traffic lights
    start_idx = n_agents + n_polylines
    theta_diff = np.where((np.arange(n) >= start_idx)[:, None] | (np.arange(n) >= start_idx)[None, :], 0, theta_diff)
    
    # Set the diagonal of the differences to a very small value
    diag_mask = np.eye(n, dtype=bool)
    epsilon = 0.01
    local_pos_x = np.where(diag_mask, epsilon, local_pos_x)
    local_pos_y = np.where(diag_mask, epsilon, local_pos_y)
    theta_diff = np.where(diag_mask, epsilon, theta_diff)
    
    # Conditions for zero coordinates
    zero_mask = np.logical_or(all_elements[:, 0][:, None] == 0, all_elements[:, 0][None, :] == 0)
    
    relations = np.stack([local_pos_x, local_pos_y, theta_diff], axis=-1)
    
    # Apply zero mask
    relations = np.where(zero_mask[..., None], 0.0, relations)
    
    return relations


def data_process_scenario(
        scenario: scenario_pb2.Scenario,
        max_num_objects: int=64,
        max_polylines: int=256,
        current_index: int=10,
        num_points_polyline: int=30,
        use_log: bool=True,
        select_agents: List[int] = None,
        remove_history: bool=False,
    ) -> Dict[str, Any]:
    data = {}

    agents_data = process_agents(
        scenario,
        max_num_objects=max_num_objects,
        num_steps=91,
        current_index=current_index,
        select_agents=select_agents,
        remove_history=remove_history,
    )
    
    traffic_light_data = process_traffic_lights(
        scenario.dynamic_map_states,
        max_num_traffic_lights=20,
        )
    
    roadgraph_data = process_roadgraph(
        scenario, 
        traffic_light_data, 
        agents_data['history'][:, -1, :2],
        max_polylines=max_polylines,
        num_points_polyline=num_points_polyline,
    )
    
    relations = calculate_relations(
        agents_data['history'], 
        roadgraph_data['polylines'], 
        traffic_light_data['traffic_light_points']
    )
    relations = relations.astype(np.float32)

    # TODO: add agents_future_valid
    data = {
        'agents_history': agents_data['history'],
        'agents_future': agents_data['future'],
        'agents_future_valid': agents_data['future_valid'],
        'agents_interested': agents_data['interested'],
        'agents_type': agents_data['type'],
        'traffic_light_points': traffic_light_data['traffic_light_points'],
        'polylines': roadgraph_data['polylines'],
        'polylines_valid': roadgraph_data['polylines_valid'],
        'relations': relations,
        'agents_id': agents_data['ids']
    }
    return data

def wm2vbd(
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
            max_polylines=MAX_POLYLINES,
            current_index=CURRENT_INDEX,
            num_points_polyline=NUM_POINTS_POLYLINE,
        )
        
        scenario_id = scenario.scenario_id
        data_dict["scenario_id"] = scenario_id
        with open(output_dir / f"{scenario_id}.pkl", "wb+") as f:
            pickle.dump(data_dict, f)

        break

def batch_process9s_transformer(input_dir, output_dir, split, num_workers):
    output_dir = Path(output_dir)
    output_dir = output_dir / split
    output_dir.mkdir(exist_ok=True, parents=True)

    input_dir = Path(input_dir) / split
    packages = sorted([p.as_posix() for p in input_dir.glob("*")])
    packages = packages[:5]

    func = partial(
        wm2vbd,
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