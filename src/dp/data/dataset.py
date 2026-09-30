import sys
import os
import torch
import pickle
import glob
import numpy as np
from torch.utils.data import Dataset
from .data_utils import *
from types import SimpleNamespace
import functools
import pickle

from pathlib import Path

class DPDataset(Dataset):
    def __init__(
            self, 
            dp_data_dir, 
            val_tfrecords_splitted=None,
            val_gt_scenario_dir=None,
            anchor_path=None,
        ):
        """
        Data class for transforming smart data to vbd input format.
        
        Args:
            dp_data_dir: DP processed data directory
        """
        self.dp_data_dir = dp_data_dir
        self.file_list = glob.glob(dp_data_dir+'/*') if dp_data_dir is not None else []

        self.anchors = pickle.load(open(anchor_path, "rb"))
        self._tfrecord_dir = Path(val_tfrecords_splitted) if val_tfrecords_splitted is not None else None
        self._gt_scenario_dir = Path(val_gt_scenario_dir) if val_gt_scenario_dir is not None else None
        
        self.__collate_fn__ = data_collate_fn
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        with open(self.file_list[idx], 'rb') as f:
            data = pickle.load(f)
        data = self.convert_to_tensor(data)
        if self._tfrecord_dir is not None:
            data["tfrecord_path"] = (
                self._tfrecord_dir / (data["scenario_id"] + ".tfrecords")
            ).as_posix()
        if self._gt_scenario_dir is not None:
            gt_path = self._gt_scenario_dir / f"{data['scenario_id']}.pkl"
            with open(gt_path, "rb") as handle:
                data["gt_scenario"] = SimpleNamespace(value=pickle.load(handle))
        return data

    def get_anchors(self, types):
        """
        Process the agent types and convert them into anchor vectors.

        Args:
            types (numpy.ndarray): Array of agent types.

        Returns:
            numpy.ndarray: Array of anchor vectors.
        """
        anchors = []

        for i in range(len(types)):
            if types[i] == 1:
                anchors.append(self.anchors['TYPE_VEHICLE'])
            elif types[i] == 2:
                anchors.append(self.anchors['TYPE_PEDESTRIAN'])
            elif types[i] == 3:
                anchors.append(self.anchors['TYPE_CYCLIST'])
            else:
                anchors.append(np.zeros_like(self.anchors['TYPE_VEHICLE']))

        return np.array(anchors, dtype=np.float32)

    def convert_to_tensor(self, data):
        """
        Convert numpy array to torch tensor.

        Args:
            data(dict): Input data dictionary of numpy arrays.

        Returns:
            torch tensor dictionary.
        """
        schema_version = int(np.asarray(data.get('map_schema_version', -1)))
        if schema_version != 9:
            raise RuntimeError(
                f"Unsupported map cache schema {schema_version}; expected version 9. "
                "Regenerate the train, validation, and test caches with the "
                "three-point continuous SMART-style map preprocessor."
            )
        sdc_coord = data['sdc_coord']
        agents_history = data['agents_history']
        agents_interested = data['agents_interested']
        agents_future = data['agents_future']
        agents_future_valid = data['agents_future_valid']
        agents_type = data['agents_type']
        traffic_light_points = data['traffic_light_points']
        agents_id = data['agents_id']
        agents_history_remaining = data['agents_history_remaining']
        agents_id_remaining = data['agents_id_remaining']
        anchors = self.get_anchors(agents_type)

        tensors = {
            "sdc_coord": torch.from_numpy(sdc_coord),
            "agents_history": torch.from_numpy(agents_history),
            "agents_interested": torch.from_numpy(agents_interested),
            "agents_future": torch.from_numpy(agents_future),
            "agents_future_valid": torch.from_numpy(agents_future_valid),
            "agents_type": torch.from_numpy(agents_type),
            "traffic_light_points": torch.from_numpy(traffic_light_points),
            "map_geometry": torch.from_numpy(data['map_geometry']),
            "map_type": torch.from_numpy(data['map_type']),
            "map_light_type": torch.from_numpy(data['map_light_type']),
            "map_stop_point": torch.from_numpy(data['map_stop_point']),
            "map_has_stop_point": torch.from_numpy(data['map_has_stop_point']),
            "map_speed_limit": torch.from_numpy(data['map_speed_limit']),
            "map_has_speed_limit": torch.from_numpy(data['map_has_speed_limit']),
            "map_parent_id": torch.from_numpy(data['map_parent_id']),
            "map_segment_index": torch.from_numpy(data['map_segment_index']),
            "map_schema_version": torch.tensor(schema_version, dtype=torch.int32),
            "anchors": torch.from_numpy(anchors),
            'agents_id': torch.from_numpy(agents_id),
            'agents_history_remaining': torch.from_numpy(agents_history_remaining),
            'agents_id_remaining': torch.from_numpy(agents_id_remaining),
            "scenario_id": data['scenario_id'],
        }
        return tensors
