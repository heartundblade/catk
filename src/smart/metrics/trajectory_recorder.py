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

import json
from pathlib import Path
from typing import Dict, List, Optional, Any

import hydra
import torch
from torch import Tensor
from torchmetrics.metric import Metric
from waymo_open_dataset.protos import sim_agents_submission_pb2

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=False)


def convert_tensor_to_native(obj: Any) -> Any:
    """
    Recursively convert Tensor objects to native Python types
    Reference: src/utils/wosac_utils.py
    """
    if isinstance(obj, torch.Tensor):
        # Convert tensor to numpy array first, then to native Python type
        return obj.cpu().numpy().tolist()
    elif isinstance(obj, dict):
        return {k: convert_tensor_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_tensor_to_native(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_tensor_to_native(item) for item in obj)
    else:
        return obj


def convert_tensor_to_native(obj: Any) -> Any:
    """
    Recursively convert Tensor objects to native Python types
    Reference: src/utils/wosac_utils.py
    """
    if isinstance(obj, torch.Tensor):
        # Convert tensor to numpy array first, then to native Python type
        return obj.cpu().numpy().tolist()
    elif isinstance(obj, dict):
        return {k: convert_tensor_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_tensor_to_native(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_tensor_to_native(item) for item in obj)
    else:
        return obj


class TrajectoryRecorder(Metric):
    def __init__(
        self,
        is_active: bool,
        save_format: str = "json",  # json, csv
        save_interval: int = 100,  # save every N scenarios
        save_dir: Optional[str] = None,
        record_predictions: bool = True,  # Whether to record predictions
        record_ground_truth: bool = True,  # Whether to record ground truth
        record_wosac_metrics: bool = True,  # Whether to record WOSAC metrics
    ) -> None:
        super().__init__()
        self.is_active = is_active
        if self.is_active:
            self.save_format = save_format
            self.save_interval = save_interval
            self.record_predictions = record_predictions
            self.record_ground_truth = record_ground_truth
            self.record_wosac_metrics = record_wosac_metrics
            
            # Set save directory
            if save_dir is None:
                self.save_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
                self.save_dir = Path(self.save_dir) / "trajectories"
            else:
                self.save_dir = Path(save_dir)
            self.save_dir.mkdir(exist_ok=True)
            
            # State variables
            self.data_keys = ["scenario_id", "agent_id"]
            
            if self.record_predictions:
                self.data_keys.extend(["pred_traj", "pred_z", "pred_head"])
            
            if self.record_ground_truth:
                self.data_keys.extend(["gt_traj", "gt_valid"])
            
            if self.record_wosac_metrics:
                self.data_keys.append("wosac_metrics")
            
            for k in self.data_keys:
                self.add_state(k, default=[], dist_reduce_fx=None)
            
            self.buffer_count = 0
            self.file_index = 0
    
    def update(
        self,
        scenario_id: List[str],
        agent_id: Tensor,
        pred_traj: Tensor,  # [n_ag, n_rollout, n_step, 2]
        pred_z: Tensor,      # [n_ag, n_rollout, n_step]
        pred_head: Tensor,   # [n_ag, n_rollout, n_step]
        gt_traj: Optional[Tensor] = None,  # [n_ag, n_step, 2]
        gt_valid: Optional[Tensor] = None, # [n_ag, n_step]
        wosac_metrics: Optional[Dict[str, Tensor]] = None,  # Pre-computed WOSAC metrics
        scenario_files: Optional[List[str]] = None,  # Scenario file paths (for compatibility)
        scenario_rollouts: Optional[Any] = None,  # Scenario rollouts (for compatibility)
    ) -> None:
        if not self.is_active:
            return
        
        # Store data - convert to CPU first as in wosac_utils.py - convert to CPU first as in wosac_utils.py
        self.scenario_id.append(scenario_id)
        
        # Convert agent_id to native types if it's a tensor
        converted_agent_id = []
        for agent in agent_id:
            if isinstance(agent, torch.Tensor):
                converted_agent = agent.cpu().numpy().tolist()
            else:
                converted_agent = agent
            converted_agent_id.append(converted_agent)
        
        # Convert agent_id to native types if it's a tensor
        converted_agent_id = []
        for agent in agent_id:
            if isinstance(agent, torch.Tensor):
                converted_agent = agent.cpu().numpy().tolist()
            else:
                converted_agent = agent
            converted_agent_id.append(converted_agent)
        self.agent_id.append(converted_agent_id)
        
        
        # Store predictions if enabled
        if self.record_predictions:
            self.pred_traj.append(pred_traj.cpu())
            self.pred_z.append(pred_z.cpu())
            self.pred_head.append(pred_head.cpu())
        
        # Store ground truth if enabled
        if self.record_ground_truth:
            if gt_traj is not None:
                self.gt_traj.append(gt_traj.cpu())
            if gt_valid is not None:
                self.gt_valid.append(gt_valid.cpu())
        
        # Store pre-computed WOSAC metrics if enabled
        if self.record_wosac_metrics and wosac_metrics is not None:
            # Convert metrics tensors to native types immediately
            converted_metrics = convert_tensor_to_native(wosac_metrics)
            self.wosac_metrics.append(converted_metrics)
        
        # Store scenario files and rollouts if provided (for future use)
        # These are not currently used but kept for compatibility
        # Store pre-computed WOSAC metrics
        if wosac_metrics is not None:
            # Convert metrics tensors to native types immediately
            converted_metrics = convert_tensor_to_native(wosac_metrics)
            self.wosac_metrics.append(converted_metrics)
        
        self.buffer_count += len(scenario_id)
        
        # Save if buffer reaches interval
        if self.buffer_count >= self.save_interval:
            self.save_trajectories()
    
    def save_trajectories(self) -> None:
        if not self.is_active or self.buffer_count == 0:
            return
        
        trajectories = []
        
        # Process trajectory data
        for i in range(len(self.scenario_id)):
            scenario_data = {
                "scenario_id": self.scenario_id[i],
                "agents": []
            }
            
            # Add pre-computed WOSAC metrics for this scenario if enabled and available
            if self.record_wosac_metrics and hasattr(self, 'wosac_metrics') and len(self.wosac_metrics) > i and self.wosac_metrics[i] is not None:
                scenario_data["wosac_metrics"] = self.wosac_metrics[i]
            # Add pre-computed WOSAC metrics for this scenario if available
            if hasattr(self, 'wosac_metrics') and len(self.wosac_metrics) > i and self.wosac_metrics[i] is not None:
                scenario_data["wosac_metrics"] = self.wosac_metrics[i]
            
            n_agents = len(self.agent_id[i])
            for j in range(n_agents):
                agent_data = {
                    "agent_id": self.agent_id[i][j]
                }
                
                # Add predictions if enabled
                if self.record_predictions:
                    agent_data["predictions"] = []
                    # Process each rollout
                    n_rollouts = self.pred_traj[i].shape[1]
                    for k in range(n_rollouts):
                        # Convert tensors to native types as in wosac_utils.py
                        rollout_data = {
                            "trajectory": self.pred_traj[i][j, k].numpy().tolist(),
                            "z": self.pred_z[i][j, k].numpy().tolist(),
                            "head": self.pred_head[i][j, k].numpy().tolist()
                        }
                        agent_data["predictions"].append(rollout_data)
                # Process each rollout
                n_rollouts = self.pred_traj[i].shape[1]
                for k in range(n_rollouts):
                    # Convert tensors to native types as in wosac_utils.py
                    rollout_data = {
                        "trajectory": self.pred_traj[i][j, k].numpy().tolist(),
                        "z": self.pred_z[i][j, k].numpy().tolist(),
                        "head": self.pred_head[i][j, k].numpy().tolist()
                    }
                    agent_data["predictions"].append(rollout_data)
                
                # Add ground truth if enabled and available
                if self.record_ground_truth and hasattr(self, 'gt_traj') and len(self.gt_traj) > i:
                    agent_data["ground_truth"] = {
                        "trajectory": self.gt_traj[i][j].numpy().tolist(),
                        "valid": self.gt_valid[i][j].numpy().tolist() if hasattr(self, 'gt_valid') and len(self.gt_valid) > i else None
                    }
                
                scenario_data["agents"].append(agent_data)
            
            trajectories.append(scenario_data)
        
        # Save to file
        if self.save_format == "json":
            output_file = self.save_dir / f"trajectories_{self.file_index:05d}.json"
            try:
                with open(output_file, 'w') as f:
                    json.dump(trajectories, f, indent=2)
                log.info(f"Saved trajectories to {output_file}")
            except Exception as e:
                log.error(f"Error saving trajectories: {e}")
                # Try to save with a more robust approach
                import pickle
                pickle_file = self.save_dir / f"trajectories_{self.file_index:05d}.pkl"
                with open(pickle_file, 'wb') as f:
                    pickle.dump(trajectories, f)
                log.info(f"Saved trajectories to {pickle_file} using pickle")
            try:
                with open(output_file, 'w') as f:
                    json.dump(trajectories, f, indent=2)
                log.info(f"Saved trajectories with WOSAC metrics to {output_file}")
            except Exception as e:
                log.error(f"Error saving trajectories: {e}")
                # Try to save with a more robust approach
                import pickle
                pickle_file = self.save_dir / f"trajectories_{self.file_index:05d}.pkl"
                with open(pickle_file, 'wb') as f:
                    pickle.dump(trajectories, f)
                log.info(f"Saved trajectories with WOSAC metrics to {pickle_file} using pickle")
        
        # Reset buffer
        for k in self.data_keys:
            setattr(self, k, [])
        self.buffer_count = 0
        self.file_index += 1
    
    def on_epoch_end(self) -> None:
        # Save any remaining trajectories at the end of epoch
        if self.is_active and self.buffer_count > 0:
            self.save_trajectories()
    
    
    def compute(self) -> Dict[str, Tensor]:
        # Implement required abstract method
        # Return empty dict as we don't need to compute metrics
        return {}