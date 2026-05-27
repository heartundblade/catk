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
        ):
        """
        Data class for transforming smart data to vbd input format.
        
        Args:
            vbd_data_dir: VBD processed data directory
        """
        self.dp_data_dir = dp_data_dir
        self.file_list = glob.glob(dp_data_dir+'/*') if dp_data_dir is not None else []

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

    def convert_to_tensor(self, data):
        """
        Convert numpy array to torch tensor.

        Args:
            data(dict): Input data dictionary of numpy arrays.

        Returns:
            torch tensor dictionary.
        """
        agents_history = data['agents_history']
        agents_interested = data['agents_interested']
        agents_future = data['agents_future']
        agents_future_valid = data['agents_future_valid']
        agents_type = data['agents_type']
        traffic_light_points = data['traffic_light_points']
        polylines = data['polylines']
        polylines_valid = data['polylines_valid']
        relations = data['relations']
        agents_id = data['agents_id']
        agents_history_remaining = data['agents_history_remaining']
        agents_id_remaining = data['agents_id_remaining']

        tensors = {
            "agents_history": torch.from_numpy(agents_history),
            "agents_interested": torch.from_numpy(agents_interested),
            "agents_future": torch.from_numpy(agents_future),
            "agents_future_valid": torch.from_numpy(agents_future_valid),
            "agents_type": torch.from_numpy(agents_type),
            "traffic_light_points": torch.from_numpy(traffic_light_points),
            "polylines": torch.from_numpy(polylines),
            "polylines_valid": torch.from_numpy(polylines_valid),
            "relations": torch.from_numpy(relations),
            'agents_id': torch.from_numpy(agents_id),
            'agents_history_remaining': torch.from_numpy(agents_history_remaining),
            'agents_id_remaining': torch.from_numpy(agents_id_remaining),
            "scenario_id": data['scenario_id'],
        }
        return tensors
