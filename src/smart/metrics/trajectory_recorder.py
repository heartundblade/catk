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
from typing import Dict, List, Optional

import hydra
import torch
from torch import Tensor
from torchmetrics.metric import Metric

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=False)


class TrajectoryRecorder(Metric):
    def __init__(
        self,
        is_active: bool,
        save_format: str = "json",  # json, csv
        save_interval: int = 100,  # save every N scenarios
        save_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.is_active = is_active
        if self.is_active:
            self.save_format = save_format
            self.save_interval = save_interval
            
            # Set save directory
            if save_dir is None:
                self.save_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
                self.save_dir = Path(self.save_dir) / "trajectories"
            else:
                self.save_dir = Path(save_dir)
            self.save_dir.mkdir(exist_ok=True)
            
            # State variables
            self.data_keys = [
                "scenario_id",
                "agent_id",
                "pred_traj",
                "pred_z",
                "pred_head",
                "gt_traj",
                "gt_valid",
                "wosac_metrics",
            ]
            for k in self.data_keys:
                self.add_state(k, default=[], dist_reduce_fx="cat")
            
            self.buffer_count = 0
            self.file_index = 0
    
    def update(
        self,
        scenario_id: List[str],
        agent_id: List[List[float]],
        pred_traj: Tensor,  # [n_ag, n_rollout, n_step, 2]
        pred_z: Tensor,      # [n_ag, n_rollout, n_step]
        pred_head: Tensor,   # [n_ag, n_rollout, n_step]
        gt_traj: Optional[Tensor] = None,  # [n_ag, n_step, 2]
        gt_valid: Optional[Tensor] = None, # [n_ag, n_step]
        wosac_metrics: Optional[Dict[str, Tensor]] = None,  # Pre-computed WOSAC metrics
    ) -> None:
        if not self.is_active:
            return
        
        # Store data
        self.scenario_id.append(scenario_id)
        self.agent_id.append(agent_id)
        self.pred_traj.append(pred_traj.cpu())
        self.pred_z.append(pred_z.cpu())
        self.pred_head.append(pred_head.cpu())
        
        if gt_traj is not None:
            self.gt_traj.append(gt_traj.cpu())
        if gt_valid is not None:
            self.gt_valid.append(gt_valid.cpu())
        
        # Store pre-computed WOSAC metrics
        if wosac_metrics is not None:
            self.wosac_metrics.append(wosac_metrics)
        
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
                "agents": [],
                "wosac_metrics": {}
            }
            
            # Add pre-computed WOSAC metrics for this scenario if available
            if hasattr(self, 'wosac_metrics') and len(self.wosac_metrics) > i and self.wosac_metrics[i] is not None:
                scenario_metrics = {}
                for key, value in self.wosac_metrics[i].items():
                    if isinstance(value, torch.Tensor):
                        scenario_metrics[key] = value.item()
                    else:
                        scenario_metrics[key] = value
                scenario_data["wosac_metrics"] = scenario_metrics
            
            n_agents = len(self.agent_id[i])
            for j in range(n_agents):
                agent_data = {
                    "agent_id": self.agent_id[i][j],
                    "predictions": []
                }
                
                # Process each rollout
                n_rollouts = self.pred_traj[i].shape[1]
                for k in range(n_rollouts):
                    rollout_data = {
                        "trajectory": self.pred_traj[i][j, k].tolist(),
                        "z": self.pred_z[i][j, k].tolist(),
                        "head": self.pred_head[i][j, k].tolist()
                    }
                    agent_data["predictions"].append(rollout_data)
                
                # Add ground truth if available
                if hasattr(self, 'gt_traj') and len(self.gt_traj) > i:
                    agent_data["ground_truth"] = {
                        "trajectory": self.gt_traj[i][j].tolist(),
                        "valid": self.gt_valid[i][j].tolist() if hasattr(self, 'gt_valid') and len(self.gt_valid) > i else None
                    }
                
                scenario_data["agents"].append(agent_data)
            
            trajectories.append(scenario_data)
        
        # Save to file
        if self.save_format == "json":
            output_file = self.save_dir / f"trajectories_{self.file_index:05d}.json"
            with open(output_file, 'w') as f:
                json.dump(trajectories, f, indent=2)
            log.info(f"Saved trajectories with WOSAC metrics to {output_file}")
        
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