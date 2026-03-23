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

"""
Beam search implementation for SMART model trajectory generation
"""

from typing import Dict, List, Tuple

import torch
from omegaconf import DictConfig

from src.smart.model.smart import SMART
from src.smart.utils import sample_next_token_traj, transform_to_global


class BeamSearch:
    """Beam search implementation for SMART model"""
    
    def __init__(self, model: SMART, config: DictConfig):
        """
        Initialize beam search
        
        Args:
            model: SMART model instance
            config: Beam search configuration
        """
        self.model = model
        self.config = config
        self.beam_width = config.beam_width
        self.max_length = config.max_length
    
    def search(self, tokenized_map: Dict[str, torch.Tensor], tokenized_agent: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Perform beam search to find optimal trajectory
        
        Args:
            tokenized_map: Tokenized map data
            tokenized_agent: Tokenized agent data
        
        Returns:
            Best trajectory prediction
        """
        # Initialize beam with initial state
        initial_state = {
            'pos_a': tokenized_agent["gt_pos"][:, :self.model.num_historical_steps//5].clone(),
            'head_a': tokenized_agent["gt_heading"][:, :self.model.num_historical_steps//5].clone(),
            'pred_valid': tokenized_agent["valid_mask"].clone(),
            'pred_idx': tokenized_agent["gt_idx"].clone(),
            'score': torch.zeros(tokenized_agent["gt_pos"].shape[0], device=tokenized_agent["gt_pos"].device)
        }
        
        beam = [initial_state]
        
        # Perform beam search
        for step in range(self.max_length):
            new_beam = []
            
            for state in beam:
                # Get current state
                pos_a = state['pos_a']
                head_a = state['head_a']
                t_now = pos_a.shape[1] - 1
                
                # Create temporary tokenized agent with current state
                temp_tokenized_agent = tokenized_agent.copy()
                temp_tokenized_agent['gt_pos'] = pos_a
                temp_tokenized_agent['gt_heading'] = head_a
                
                # Get model prediction
                map_feature = self.model.encoder.map_encoder(tokenized_map)
                pred_dict = self.model.encoder.agent_encoder.inference(
                    temp_tokenized_agent, map_feature, self.model.validation_rollout_sampling
                )
                
                # Get next token logits
                next_token_logits = pred_dict['next_token_logits'][:, 0]  # [n_agent, n_token]
                
                # Get top-k tokens and their scores
                topk_scores, topk_indices = torch.topk(next_token_logits, k=self.beam_width, dim=1)
                
                # Generate child states
                for i in range(self.beam_width):
                    action = topk_indices[:, i]
                    score = topk_scores[:, i]
                    
                    # Generate child state
                    child_state = self._generate_child_state(state, action, score, tokenized_agent)
                    new_beam.append(child_state)
            
            # Sort by score and keep top beam_width states
            new_beam.sort(key=lambda x: x['score'].mean().item(), reverse=True)
            beam = new_beam[:self.beam_width]
        
        # Get best trajectory
        best_state = max(beam, key=lambda x: x['score'].mean().item())
        return {
            'pred_pos': best_state['pos_a'],
            'pred_head': best_state['head_a'],
            'pred_traj_10hz': self._convert_to_10hz(best_state['pos_a'], best_state['head_a']),
            'pred_head_10hz': self._convert_to_10hz_heading(best_state['head_a']),
        }
    
    def _generate_child_state(self, parent_state: Dict[str, torch.Tensor], action: torch.Tensor, score: torch.Tensor, tokenized_agent: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Generate child state from parent state and action
        
        Args:
            parent_state: Parent state
            action: Action (token index)
            score: Score for this action
            tokenized_agent: Tokenized agent data
        
        Returns:
            Child state
        """
        pos_a = parent_state['pos_a']
        head_a = parent_state['head_a']
        pred_valid = parent_state['pred_valid'].clone()
        pred_idx = parent_state['pred_idx'].clone()
        parent_score = parent_state['score']
        
        t_now = pos_a.shape[1] - 1
        n_step = t_now + 1
        
        # Get token trajectory
        _, next_token_traj_all = sample_next_token_traj(
            token_traj=tokenized_agent["token_traj"],
            token_traj_all=tokenized_agent["token_traj_all"],
            sampling_scheme=self.model.validation_rollout_sampling,
            next_token_logits=None,
            pos_now=pos_a[:, t_now],
            head_now=head_a[:, t_now],
            pos_next_gt=tokenized_agent["gt_pos_raw"][:, n_step],
            head_next_gt=tokenized_agent["gt_head_raw"][:, n_step],
            valid_next_gt=tokenized_agent["gt_valid_raw"][:, n_step],
            token_agent_shape=tokenized_agent["token_agent_shape"],
            sampled_idx=action
        )
        
        # Transform to global coordinates
        token_traj_global = transform_to_global(
            pos_local=next_token_traj_all.flatten(1, 2),
            head_local=None,
            pos_now=pos_a[:, t_now],
            head_now=head_a[:, t_now]
        )[0].view(*next_token_traj_all.shape)
        
        # Get next state
        pos_a_next = token_traj_global[:, -1].mean(dim=1)
        diff_xy_next = token_traj_global[:, -1, 0] - token_traj_global[:, -1, 3]
        head_a_next = torch.arctan2(diff_xy_next[:, 1], diff_xy_next[:, 0])
        
        # Update state
        pos_a = torch.cat([pos_a, pos_a_next.unsqueeze(1)], dim=1)
        head_a = torch.cat([head_a, head_a_next.unsqueeze(1)], dim=1)
        pred_valid[:, n_step] = pred_valid[:, t_now]
        pred_idx[:, n_step] = action
        
        # Update score
        new_score = parent_score + score
        
        return {
            'pos_a': pos_a,
            'head_a': head_a,
            'pred_valid': pred_valid,
            'pred_idx': pred_idx,
            'score': new_score
        }
    
    def _convert_to_10hz(self, pos_a: torch.Tensor, head_a: torch.Tensor) -> torch.Tensor:
        """
        Convert 2Hz trajectory to 10Hz
        
        Args:
            pos_a: 2Hz position trajectory [n_agent, n_step_2hz, 2]
            head_a: 2Hz heading trajectory [n_agent, n_step_2hz]
        
        Returns:
            10Hz position trajectory [n_agent, n_step_10hz, 2]
        """
        n_agent, n_step_2hz, _ = pos_a.shape
        n_step_10hz = (n_step_2hz - 1) * 5
        
        pred_traj_10hz = torch.zeros([n_agent, n_step_10hz, 2], dtype=pos_a.dtype, device=pos_a.device)
        
        # Interpolate between 2Hz steps
        for i in range(n_step_2hz - 1):
            start_pos = pos_a[:, i]
            end_pos = pos_a[:, i+1]
            
            # Linear interpolation
            for j in range(5):
                alpha = j / 5.0
                pred_traj_10hz[:, i*5 + j] = start_pos * (1 - alpha) + end_pos * alpha
        
        return pred_traj_10hz
    
    def _convert_to_10hz_heading(self, head_a: torch.Tensor) -> torch.Tensor:
        """
        Convert 2Hz heading to 10Hz
        
        Args:
            head_a: 2Hz heading trajectory [n_agent, n_step_2hz]
        
        Returns:
            10Hz heading trajectory [n_agent, n_step_10hz]
        """
        n_agent, n_step_2hz = head_a.shape
        n_step_10hz = (n_step_2hz - 1) * 5
        
        pred_head_10hz = torch.zeros([n_agent, n_step_10hz], dtype=head_a.dtype, device=head_a.device)
        
        # Interpolate between 2Hz steps
        for i in range(n_step_2hz - 1):
            start_head = head_a[:, i]
            end_head = head_a[:, i+1]
            
            # Linear interpolation
            for j in range(5):
                alpha = j / 5.0
                pred_head_10hz[:, i*5 + j] = start_head * (1 - alpha) + end_head * alpha
        
        return pred_head_10hz


def run_beam_search(model: SMART, tokenized_map: Dict[str, torch.Tensor], tokenized_agent: Dict[str, torch.Tensor], config: DictConfig) -> Dict[str, torch.Tensor]:
    """
    Run beam search to generate trajectory
    
    Args:
        model: SMART model instance
        tokenized_map: Tokenized map data
        tokenized_agent: Tokenized agent data
        config: Beam search configuration
    
    Returns:
        Trajectory prediction
    """
    beam_search = BeamSearch(model, config)
    return beam_search.search(tokenized_map, tokenized_agent)