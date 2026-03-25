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
    
    def __init__(
        self,
        model: SMART,
        config: DictConfig
    ):
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
        self.max_search_iterations = config.max_search_iterations if hasattr(config, 'max_search_iterations') else config.max_length
    
    def search(
        self,
        tokenized_map: Dict[str, torch.Tensor],
        tokenized_agent: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Perform beam search for each agent independently
        
        Args:
            tokenized_map: Tokenized map data
            tokenized_agent: Tokenized agent data
        
        Returns:
            Best trajectory predictions for all agents
        """
        n_agent = tokenized_agent["gt_pos"].shape[0]
        agent_results = []
        
        # Get ground truth trajectory
        gt_pos_raw = tokenized_agent["gt_pos_raw"]
        gt_head_raw = tokenized_agent["gt_head_raw"]
        gt_valid_raw = tokenized_agent["gt_valid_raw"]
        
        # Get first step's ground truth (only use for first step)
        first_step = self.model.num_historical_steps//5
        first_step_gt_pos = None
        first_step_gt_head = None
        first_step_gt_valid = None
        if first_step < gt_pos_raw.shape[1]:
            first_step_gt_pos = gt_pos_raw[:, first_step].unsqueeze(1)
            first_step_gt_head = gt_head_raw[:, first_step].unsqueeze(1)
            first_step_gt_valid = gt_valid_raw[:, first_step]
        
        # Get map feature once for all agents
        map_feature = self.model.encoder.map_encoder(tokenized_map)
        
        # Perform beam search for each agent independently
        for agent_idx in range(n_agent):
            # Extract data for single agent
            single_agent_data = {
                k: v[agent_idx:agent_idx+1] if v.ndim >= 2 else v
                for k, v in tokenized_agent.items()
            }
            
            # Get ground truth for this agent
            agent_first_step_gt_pos = first_step_gt_pos[agent_idx:agent_idx+1] if first_step_gt_pos is not None else None
            agent_first_step_gt_head = first_step_gt_head[agent_idx:agent_idx+1] if first_step_gt_head is not None else None
            agent_first_step_gt_valid = first_step_gt_valid[agent_idx:agent_idx+1] if first_step_gt_valid is not None else None
            
            # Create data for surrounding agents using ground truth
            # For each step, we'll update surrounding agents to their ground truth positions
            
            # Run beam search for this agent
            agent_result = self._search_single_agent(
                tokenized_map, 
                single_agent_data, 
                map_feature, 
                agent_first_step_gt_pos, 
                agent_first_step_gt_head, 
                agent_first_step_gt_valid,
                full_tokenized_agent=tokenized_agent
            )
            agent_results.append(agent_result)
        
        # Combine results from all agents
        # Get agent IDs from tokenized_agent
        agent_ids = tokenized_agent.get('id', None)
        if agent_ids is None:
            # If agent_ids is not available, create sequential IDs
            n_agent = len(agent_results)
            agent_ids = torch.arange(n_agent, device=agent_results[0]['pred_pos'].device).unsqueeze(1)
        else:
            # Use existing agent IDs
            agent_ids = agent_ids.unsqueeze(1)
        
        return {
            'pred_pos': torch.cat([r['pred_pos'] for r in agent_results], dim=0),
            'pred_head': torch.cat([r['pred_head'] for r in agent_results], dim=0),
            'pred_traj_10hz': torch.cat([r['pred_traj_10hz'] for r in agent_results], dim=0),
            'pred_head_10hz': torch.cat([r['pred_head_10hz'] for r in agent_results], dim=0),
            'agent_ids': agent_ids,
        }
    
    def _search_single_agent(
        self,
        tokenized_map: Dict[str, torch.Tensor],
        tokenized_agent: Dict[str, torch.Tensor],
        map_feature: torch.Tensor,
        first_step_gt_pos: torch.Tensor,
        first_step_gt_head: torch.Tensor,
        first_step_gt_valid: torch.Tensor,
        full_tokenized_agent: Dict[str, torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Perform beam search for a single agent
        
        Args:
            tokenized_map: Tokenized map data
            tokenized_agent: Tokenized agent data (single agent)
            map_feature: Pre-computed map feature
            first_step_gt_pos: Ground truth position for first step
            first_step_gt_head: Ground truth heading for first step
            first_step_gt_valid: Ground truth validity for first step
            full_tokenized_agent: Full tokenized agent data including all agents
        
        Returns:
            Best trajectory prediction for the agent
        """
        # Initialize beam with initial state for single agent
        initial_state = {
            'pos_a': tokenized_agent["gt_pos"][:, :self.model.num_historical_steps//5].clone(),
            'head_a': tokenized_agent["gt_heading"][:, :self.model.num_historical_steps//5].clone(),
            'pred_valid': tokenized_agent["valid_mask"].clone(),
            'pred_idx': tokenized_agent["gt_idx"].clone(),
            'score': torch.zeros(tokenized_agent["gt_pos"].shape[0], device=tokenized_agent["gt_pos"].device),
            'feat_a': None,  # Will be set in first inference
            'feat_a_t_dict': None  # Will be set in first inference
        }
        
        beam = [initial_state]
        
        # Perform beam search
        for step in range(self.max_length):
            new_beam = []
            
            # Calculate current step in terms of ground truth
            current_gt_step = self.model.num_historical_steps//5 + step
            
            # Only use ground truth for the first step
            if step == 0:
                current_gt_pos = first_step_gt_pos
                current_gt_head = first_step_gt_head
                current_gt_valid = first_step_gt_valid
            else:
                # For subsequent steps, don't use ground truth for the agent being searched
                current_gt_pos = None
                current_gt_head = None
                current_gt_valid = None
            
            for state in beam:
                # Get current state
                pos_a = state['pos_a']
                head_a = state['head_a']
                t_now = pos_a.shape[1] - 1
                
                # Create temporary tokenized agent with current state
                temp_tokenized_agent = tokenized_agent.copy()
                temp_tokenized_agent['gt_pos'] = pos_a
                temp_tokenized_agent['gt_heading'] = head_a
                
                # Update surrounding agents' states with ground truth
                if full_tokenized_agent is not None:
                    # For surrounding agents, use ground truth at current step
                    # Note: This assumes that full_tokenized_agent contains all agents' data
                    # In practice, you would need to identify which agents are surrounding agents
                    # and update their states accordingly
                    pass
                
                # Check if this is the initial step for this state
                is_initial_step = (pos_a.shape[1] == self.model.num_historical_steps//5)
                
                # Get previous features if available
                prev_feat_a = state.get('feat_a', None)
                prev_feat_a_t_dict = state.get('feat_a_t_dict', None)
                
                # Run single step inference
                pred_dict = self.model.encoder.agent_encoder.inference_single_step(
                    temp_tokenized_agent, 
                    map_feature, 
                    prev_feat_a=prev_feat_a,
                    prev_feat_a_t_dict=prev_feat_a_t_dict,
                    is_initial_step=is_initial_step
                )
                
                # Get next token logits for this agent
                next_token_logits = pred_dict['next_token_logits']  # [1, n_token]
                
                # Get top-k tokens and their scores
                topk_scores, topk_indices = torch.topk(next_token_logits, k=self.beam_width, dim=1)
                
                # Generate child states
                for i in range(self.beam_width):
                    action = topk_indices[:, i]
                    score = topk_scores[:, i]
                    
                    # Generate child state using ground truth only for first step
                    child_state = self._generate_child_state_with_gt(
                        state,
                        action,
                        score,
                        tokenized_agent,
                        current_gt_pos,
                        current_gt_head,
                        current_gt_valid,
                        self.model.num_historical_steps//5 if step == 0 else None,
                        pred_dict=pred_dict
                    )
                    new_beam.append(child_state)
            
            # Sort by score and keep top beam_width states
            new_beam.sort(key=lambda x: x['score'].item(), reverse=True)
            beam = new_beam[:self.beam_width]
        
        # Get best trajectory for this agent
        best_state = max(beam, key=lambda x: x['score'].item())
        return {
            'pred_pos': best_state['pos_a'],
            'pred_head': best_state['head_a'],
            'pred_traj_10hz': self._convert_to_10hz(best_state['pos_a'], best_state['head_a']),
            'pred_head_10hz': self._convert_to_10hz_heading(best_state['head_a']),
        }
    
    def _generate_child_state_with_gt(
        self,
        parent_state: Dict[str, torch.Tensor],
        action: torch.Tensor,
        score: torch.Tensor,
        tokenized_agent: Dict[str, torch.Tensor],
        gt_pos: torch.Tensor,
        gt_head: torch.Tensor,
        gt_valid: torch.Tensor,
        step: int,
        pred_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Generate child state from parent state and action, using ground truth only for first step
        
        Args:
            parent_state: Parent state
            action: Action (token index)
            score: Score for this action
            tokenized_agent: Tokenized agent data
            gt_pos: Ground truth position for next step (only used for first step)
            gt_head: Ground truth heading for next step (only used for first step)
            gt_valid: Ground truth validity for next step (only used for first step)
            step: Current step index (None for non-first steps)
        
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
        
        # Use ground truth for next state only if it's the first step and ground truth is available
        if step is not None and gt_pos is not None and gt_head is not None:
            # Update state with ground truth (only for first step)
            pos_a = torch.cat([pos_a, gt_pos], dim=1)
            head_a = torch.cat([head_a, gt_head], dim=1)
            if gt_valid is not None:
                pred_valid[:, n_step] = gt_valid
            else:
                pred_valid[:, n_step] = pred_valid[:, t_now]
        else:
            # # For subsequent steps, always use model prediction
            # # Get token trajectory
            # _, next_token_traj_all = sample_next_token_traj(
            #     token_traj=tokenized_agent["token_traj"],
            #     token_traj_all=tokenized_agent["token_traj_all"],
            #     sampling_scheme=self.model.validation_rollout_sampling,
            #     next_token_logits=None,
            #     pos_now=pos_a[:, t_now],
            #     head_now=head_a[:, t_now],
            #     pos_next_gt=None,  # Don't use ground truth for subsequent steps
            #     head_next_gt=None,  # Don't use ground truth for subsequent steps
            #     valid_next_gt=None,  # Don't use ground truth for subsequent steps
            #     token_agent_shape=tokenized_agent["token_agent_shape"],
            #     sampled_idx=action
            # )

            next_token_traj_all = tokenized_agent["token_traj_all"][torch.arange(tokenized_agent["token_traj_all"].shape[0]), action]

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
        
        # Update prediction index and score
        pred_idx[:, n_step] = action
        new_score = parent_score + score
        
        return {
            'pos_a': pos_a,
            'head_a': head_a,
            'pred_valid': pred_valid,
            'pred_idx': pred_idx,
            'score': new_score,
            'feat_a': pred_dict.get('feat_a', None),
            'feat_a_t_dict': pred_dict.get('feat_a_t_dict', None)
        }
    
    def _convert_to_10hz(
        self,
        pos_a: torch.Tensor,
        head_a: torch.Tensor
    ) -> torch.Tensor:
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
        
        # Handle case when there are no steps to interpolate
        if n_step_10hz == 0:
            return torch.empty([n_agent, 0, 2], dtype=pos_a.dtype, device=pos_a.device)
        
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
    
    def _convert_to_10hz_heading(
        self,
        head_a: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert 2Hz heading to 10Hz
        
        Args:
            head_a: 2Hz heading trajectory [n_agent, n_step_2hz]
        
        Returns:
            10Hz heading trajectory [n_agent, n_step_10hz]
        """
        n_agent, n_step_2hz = head_a.shape
        n_step_10hz = (n_step_2hz - 1) * 5
        
        # Handle case when there are no steps to interpolate
        if n_step_10hz == 0:
            return torch.empty([n_agent, 0], dtype=head_a.dtype, device=head_a.device)
        
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


    def closed_loop_search(
        self,
        tokenized_map: Dict[str, torch.Tensor],
        tokenized_agent: Dict[str, torch.Tensor]
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Perform closed-loop beam search to find optimal trajectory
        
        Args:
            tokenized_map: Tokenized map data
            tokenized_agent: Tokenized agent data
        
        Returns:
            List of best trajectory predictions for each closed-loop step
        """
        # Initialize search history
        search_history = []
        
        # Copy input data
        current_tokenized_agent = {k: v.clone() for k, v in tokenized_agent.items()}
        
        # Get ground truth trajectory length and historical steps
        gt_length = tokenized_agent["gt_pos_raw"].shape[1]
        historical_steps = self.model.num_historical_steps // 5
        
        # Perform closed-loop search
        for search_iter in range(self.max_search_iterations):
            # Run one beam search
            search_result = self.search(tokenized_map, current_tokenized_agent)
            
            # Record search result
            search_history.append(search_result)
            
            # Check if reached ground truth length
            current_step = historical_steps + search_iter
            if current_step + 1 >= gt_length:
                break
            
            # Update agent state based on ground truth
            # Refer to agent_decoder.py state update logic
            
            # Get current ground truth step
            current_gt_step = current_step
            next_gt_step = current_gt_step + 1
            
            # Update position and heading data using ground truth
            # Keep historical ground truth data and append new ground truth data
            new_gt_pos = torch.cat([
                tokenized_agent["gt_pos"][:, :historical_steps],
                tokenized_agent["gt_pos_raw"][:, historical_steps:next_gt_step+1]
            ], dim=1)
            
            new_gt_head = torch.cat([
                tokenized_agent["gt_heading"][:, :historical_steps],
                tokenized_agent["gt_head_raw"][:, historical_steps:next_gt_step+1]
            ], dim=1)
            
            # Update valid mask
            new_valid_mask = torch.cat([
                tokenized_agent["valid_mask"][:, :historical_steps],
                tokenized_agent["gt_valid_raw"][:, historical_steps:next_gt_step+1]
            ], dim=1)
            
            # Update prediction index
            # Note: Use ground truth index as we update state based on ground truth
            new_gt_idx = torch.cat([
                tokenized_agent["gt_idx"][:, :historical_steps],
                tokenized_agent["gt_idx"][:, historical_steps:next_gt_step+1]
            ], dim=1)
            
            # Update input data
            current_tokenized_agent['gt_pos'] = new_gt_pos
            current_tokenized_agent['gt_heading'] = new_gt_head
            current_tokenized_agent['valid_mask'] = new_valid_mask
            current_tokenized_agent['gt_idx'] = new_gt_idx
            
            # Ensure other necessary fields are updated
            # Fields like token_traj, token_traj_all are already present in tokenized_agent
            # No need to recalculate as they are fixed trajectory libraries
        
        return search_history
    
    def _select_best_result(
        self,
        search_history: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Select the best result from search history
        
        Args:
            search_history: List of search results
        
        Returns:
            Best trajectory prediction
        """
        # Selection strategy can be implemented based on specific needs
        # For example, select the last search result, or based on some scoring mechanism
        return search_history[-1]

def run_beam_search(
    model: SMART,
    tokenized_map: Dict[str, torch.Tensor],
    tokenized_agent: Dict[str, torch.Tensor],
    config: DictConfig
) -> List[Dict[str, torch.Tensor]]:
    """
    Run beam search to generate trajectory
    
    Args:
        model: SMART model instance
        tokenized_map: Tokenized map data
        tokenized_agent: Tokenized agent data
        config: Beam search configuration
    
    Returns:
        List of trajectory predictions for each closed-loop step
    """
    beam_search = BeamSearch(model, config)
    return beam_search.closed_loop_search(tokenized_map, tokenized_agent)