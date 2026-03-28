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
        gt_pos = tokenized_agent["gt_pos"]  # [n_agent, n_step, 2]
        gt_heading = tokenized_agent["gt_heading"]  # [n_agent, n_step]
        valid_mask = tokenized_agent["valid_mask"]  # [n_agent, n_step]
        
        # Get first step's ground truth (only use for first step)
        first_step = self.model.num_historical_steps//5
        first_step_gt_pos = None
        first_step_gt_heading = None
        first_step_valid_mask = None
        if first_step < gt_pos.shape[1]:
            first_step_gt_pos = gt_pos[:, first_step].unsqueeze(1)
            first_step_gt_heading = gt_heading[:, first_step].unsqueeze(1)
            first_step_valid_mask = valid_mask[:, first_step]
        
        # Get map feature once for all agents
        map_feature = self.model.encoder.map_encoder(tokenized_map)
        
        # Perform beam search for each agent independently
        for agent_idx in range(n_agent):
            print(f"Running beam search for agent {agent_idx+1}/{n_agent}...")
            # Extract data for single agent
            single_agent_data = {
                k: v[agent_idx:agent_idx+1] if isinstance(v, torch.Tensor) and (v.ndim >= 2 or k == 'id') else v
                for k, v in tokenized_agent.items()
            }  # All tensors become [1, ...] for single agent
            
            # Get ground truth for this agent
            agent_first_step_gt_pos = first_step_gt_pos[agent_idx:agent_idx+1] if first_step_gt_pos is not None else None
            agent_first_step_gt_heading = first_step_gt_heading[agent_idx:agent_idx+1] if first_step_gt_heading is not None else None
            agent_first_step_valid_mask = first_step_valid_mask[agent_idx:agent_idx+1] if first_step_valid_mask is not None else None
            
            # Run beam search for this agent
            agent_result = self._search_single_agent(
                tokenized_map, 
                single_agent_data, 
                map_feature, 
                agent_first_step_gt_pos, 
                agent_first_step_gt_heading, 
                agent_first_step_valid_mask,
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
            'pred_pos': torch.cat([r['pred_pos'] for r in agent_results], dim=0),  # [n_agent, n_step, 2]
            'pred_head': torch.cat([r['pred_head'] for r in agent_results], dim=0),  # [n_agent, n_step]
            'pred_traj_10hz': torch.cat([r['pred_traj_10hz'] for r in agent_results], dim=0),  # [n_agent, n_step_10hz, 2]
            'pred_head_10hz': torch.cat([r['pred_head_10hz'] for r in agent_results], dim=0),  # [n_agent, n_step_10hz]
            'agent_ids': agent_ids,  # [n_agent, 1]
        }
    
    def _search_single_agent(
        self,
        tokenized_map: Dict[str, torch.Tensor],
        tokenized_agent: Dict[str, torch.Tensor],
        map_feature: torch.Tensor,
        first_step_gt_pos: torch.Tensor,
        first_step_gt_heading: torch.Tensor,
        first_step_valid_mask: torch.Tensor,
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
        # Get actual historical steps from input data
        historical_steps = tokenized_agent["pos"].shape[1]
        
        # Initialize beam with initial state for single agent
        initial_state = {
            'pos_a': tokenized_agent["pos"].clone(),  # [1, n_historical_step, 2]
            'head_a': tokenized_agent["heading"].clone(),  # [1, n_historical_step]
            'pred_valid': tokenized_agent["valid_mask"].clone(),  # [1, n_historical_step]
            'pred_idx': tokenized_agent["gt_idx"].clone(),  # [1, n_historical_step]
            'score': torch.zeros(tokenized_agent["pos"].shape[0], device=tokenized_agent["pos"].device),  # [1]
            'feat_a': None,  # Will be set in first inference
            'feat_a_t_dict': None  # Will be set in first inference
        }
        
        # Calculate current step in terms of ground truth for first step
        current_gt_step = historical_steps
        current_gt_pos = first_step_gt_pos
        current_gt_heading = first_step_gt_heading
        current_valid_mask = first_step_valid_mask
        
        # Get current state
        pos_a = initial_state['pos_a']
        head_a = initial_state['head_a']
        t_now = pos_a.shape[1] - 1
        
        # Create temporary tokenized agent with current state
        temp_tokenized_agent = tokenized_agent.copy()
        temp_tokenized_agent['pos'] = pos_a
        temp_tokenized_agent['heading'] = head_a
        
        # Update surrounding agents' states with ground truth
        if full_tokenized_agent is not None:
            # For surrounding agents, use ground truth at current step
            # Create a copy of full_tokenized_agent with ground truth data
            temp_full_tokenized_agent = {
                k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in full_tokenized_agent.items()
                }
            
            # For all agents except the current one, use ground truth at current_gt_step
            # First, get the current agent's ID (if available)
            current_agent_id = None
            if 'id' in tokenized_agent:
                current_agent_id = tokenized_agent['id'].item()
            
            # Get number of agents
            n_agent = full_tokenized_agent['gt_pos'].shape[0]
            
            # Update surrounding agents' positions and headings to ground truth
            # at the current step
            if temp_full_tokenized_agent['pos'].shape[1] <= t_now + 1:
                # If not enough steps, append zeros for all agents
                device = temp_full_tokenized_agent['pos'].device
                
                # Create zero tensors for the new step
                new_pos = torch.zeros([n_agent, 1, 2], device=device)
                new_heading = torch.zeros([n_agent, 1], device=device)
                
                # Append the new step
                temp_full_tokenized_agent['pos'] = torch.cat([
                    temp_full_tokenized_agent['pos'],
                    new_pos
                ], dim=1)
                temp_full_tokenized_agent['heading'] = torch.cat([
                    temp_full_tokenized_agent['heading'],
                    new_heading
                ], dim=1)
            
            # Update all agents with ground truth
            if current_gt_step < full_tokenized_agent['gt_pos'].shape[1]:
                # Get ground truth for all agents at current step
                gt_pos_all = full_tokenized_agent['gt_pos'][:, current_gt_step:current_gt_step+1]
                gt_heading_all = full_tokenized_agent['gt_heading'][:, current_gt_step:current_gt_step+1]
                
                # Apply ground truth to all agents
                for agent_idx in range(n_agent):
                    temp_full_tokenized_agent['gt_pos'][agent_idx:agent_idx+1, t_now+1:t_now+2] = gt_pos_all[agent_idx:agent_idx+1]
                    temp_full_tokenized_agent['gt_heading'][agent_idx:agent_idx+1, t_now+1:t_now+2] = gt_heading_all[agent_idx:agent_idx+1]
                    
        # Get ground truth action token indice for the first step
        gt_action = None
        if 'gt_idx' in tokenized_agent and tokenized_agent['gt_idx'].shape[1] > historical_steps:
            # Get the current agent's action token indice directly from tokenized_agent
            gt_action = tokenized_agent['gt_idx'][:, historical_steps]  # [1]
        
        # Update initial state with ground truth for first step
        # Use _generate_child_state_with_gt to update the state with ground truth
        initial_state = self._generate_child_state_with_gt(
            initial_state,
            gt_action,  # Use ground truth action token indice
            torch.zeros_like(initial_state['score']),  # No score change for ground truth
            temp_tokenized_agent, 
            current_gt_pos,
            current_gt_heading,
            current_valid_mask,
            step=0,  # Step 0 for ground truth
            pred_dict=None  # No pred_dict needed for ground truth
        )
        beam = [initial_state]

        # Perform beam search for remaining steps (reduced by 1)
        for step in range(1, self.max_length):
            print(f"Beam search step {step+1}")
            new_beam = []
            
            # Calculate current step in terms of ground truth
            current_gt_step = historical_steps + step
            
            # Check if this is the initial step for this state
            is_initial_step = step == 1  # Step 1 is the first step after the initial state which already used ground truth
            
            # For subsequent steps, don't use ground truth for the agent being searched
            for state in beam:
                # Get current state
                pos_a = state['pos_a']
                head_a = state['head_a']
                t_now = pos_a.shape[1] - 1
                
                # Create temporary tokenized agent with current state
                temp_tokenized_agent = tokenized_agent.copy()
                temp_tokenized_agent['pos'] = pos_a
                temp_tokenized_agent['heading'] = head_a
                
                # Get previous features if available
                prev_feat_a = state.get('feat_a', None)
                prev_feat_a_t_dict = state.get('feat_a_t_dict', None)
                
                # Run single step inference before updating with ground truth
                pred_dict = self.model.encoder.agent_encoder.inference_single_step(
                    temp_full_tokenized_agent, 
                    map_feature, 
                    prev_feat_a=prev_feat_a,
                    prev_feat_a_t_dict=prev_feat_a_t_dict,
                    is_initial_step=is_initial_step
                )
                
                # Update surrounding agents' states with ground truth
                if full_tokenized_agent is not None:
                    # For surrounding agents, use ground truth at current step
                    # Create a copy of full_tokenized_agent with ground truth data
                    temp_full_tokenized_agent = {
                        k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in full_tokenized_agent.items()
                        }
                    
                    # Update surrounding agents' positions and headings to ground truth
                    # at the current step
                    if temp_full_tokenized_agent['gt_pos'].shape[1] <= t_now + 1:
                        # If not enough steps, append zeros for all agents
                        n_agent = temp_full_tokenized_agent['gt_pos'].shape[0]
                        device = temp_full_tokenized_agent['gt_pos'].device
                        
                        # Create zero tensors for the new step
                        new_pos = torch.zeros([n_agent, 1, 2], device=device)
                        new_head = torch.zeros([n_agent, 1], device=device)
                        
                        # Append the new step
                        temp_full_tokenized_agent['pos'] = torch.cat([
                            temp_full_tokenized_agent['pos'],
                            new_pos
                        ], dim=1)
                        temp_full_tokenized_agent['heading'] = torch.cat([
                            temp_full_tokenized_agent['heading'],
                            new_head
                        ], dim=1)
                    
                    # Create a mask to identify which agents to update with ground truth
                    n_agent = full_tokenized_agent['gt_pos'].shape[0]
                    update_mask = torch.ones(n_agent, dtype=torch.bool, device=temp_full_tokenized_agent['gt_pos'].device)
                    
                    # Todo: use matrix computation to get update_mask instead of looping through agents
                    # Mark current agent as not to be updated
                    if current_agent_id is not None and 'id' in full_tokenized_agent:
                        for idx in range(n_agent):
                            if full_tokenized_agent['id'][idx].item() == current_agent_id:
                                update_mask[idx] = False
                                break
                    else:
                        # If no agent ID, assume first agent is current
                        update_mask[0] = False
                    
                    # Todo: use matrix computation to update all agents once instead of looping through agents
                    # Update agents with ground truth where mask is True
                    if current_gt_step < full_tokenized_agent['gt_pos'].shape[1]:
                        # Get ground truth for all agents at current step
                        gt_pos_all = full_tokenized_agent['gt_pos'][:, current_gt_step:current_gt_step+1]
                        gt_head_all = full_tokenized_agent['gt_heading'][:, current_gt_step:current_gt_step+1]
                        
                        # Apply ground truth only to agents where update_mask is True
                        for agent_idx in range(n_agent):
                            if update_mask[agent_idx]:
                                temp_full_tokenized_agent['pos'][agent_idx:agent_idx+1, t_now+1:t_now+2] = gt_pos_all[agent_idx:agent_idx+1]
                                temp_full_tokenized_agent['heading'][agent_idx:agent_idx+1, t_now+1:t_now+2] = gt_head_all[agent_idx:agent_idx+1]
                    
                    # Use the updated full tokenized agent data
                    temp_tokenized_agent = temp_full_tokenized_agent
                    # Update current agent's state using the beam state
                    # Use the inverse of update_mask to identify current agent
                    current_agent_mask = ~update_mask
                    current_agent_mask = current_agent_mask.unsqueeze(1).unsqueeze(2)  # [n_agent, 1, 1]

                    # Update position and heading for current agent only
                    temp_tokenized_agent['pos'][:, t_now+1:t_now+2] = torch.where(
                        current_agent_mask.expand(-1, 1, 2),
                        pos_a[:, -1:].expand(n_agent, 1, 2),
                        temp_tokenized_agent['pos'][:, t_now+1:t_now+2]
                    )
                    temp_tokenized_agent['heading'][:, t_now+1:t_now+2] = torch.where(
                        current_agent_mask.expand(-1, 1),
                        head_a[:, -1:].expand(n_agent, 1),
                        temp_tokenized_agent['heading'][:, t_now+1:t_now+2]
                    )

                    # # Use the updated full tokenized agent data
                    # temp_tokenized_agent = temp_full_tokenized_agent
                    # # But keep the current agent's state as the beam state
                    # # Ensure we have enough steps for the current agent
                    # if temp_tokenized_agent['pos'].shape[1] <= t_now + 1:
                    #     # If not enough steps, append the current agent's state
                    #     temp_tokenized_agent['pos'] = torch.cat([
                    #         temp_tokenized_agent['pos'],
                    #         pos_a[:, -1:]
                    #     ], dim=1)
                    #     temp_tokenized_agent['heading'] = torch.cat([
                    #         temp_tokenized_agent['heading'],
                    #         head_a[:, -1:]
                    #     ], dim=1)
                    #     # Also append to idx to maintain consistent time steps
                    #     if 'idx' in temp_tokenized_agent:
                    #         # Repeat the last token index for the new step
                    #         temp_tokenized_agent['idx'] = torch.cat([
                    #             temp_tokenized_agent['idx'],
                    #             temp_tokenized_agent['idx'][:, -1:]
                    #         ], dim=1)
                    # else:
                    #     # Otherwise, update the specific step
                    #     temp_tokenized_agent['pos'][0:1, t_now+1:t_now+2] = pos_a[:, -1:]
                    #     temp_tokenized_agent['heading'][0:1, t_now+1:t_now+2] = head_a[:, -1:]
                
                # Extract only the current agent's logits
                if 'next_token_logits' in pred_dict:
                    # If we're using full_tokenized_agent, extract only the current agent's logits
                    if full_tokenized_agent is not None:
                        current_agent_idx = 0
                        if current_agent_id is not None and 'id' in full_tokenized_agent:
                            for idx in range(full_tokenized_agent['id'].shape[0]):
                                if full_tokenized_agent['id'][idx].item() == current_agent_id:
                                    current_agent_idx = idx
                                    break
                        pred_dict['next_token_logits'] = pred_dict['next_token_logits'][current_agent_idx:current_agent_idx+1]
                    
                    # Get next token logits for this agent
                    next_token_logits = pred_dict['next_token_logits']  # [1, n_token]
                    
                    # Get top-k tokens and their scores
                    topk_scores, topk_indices = torch.topk(
                        next_token_logits, k=self.beam_width, dim=1
                        )  # topk_scores: [1, beam_width], topk_indices: [1, beam_width]
                    
                    # Generate child states
                    for i in range(self.beam_width):
                        action = topk_indices[:, i]
                        score = topk_scores[:, i]
                        
                        # Generate child state without ground truth for subsequent steps
                        child_state = self._generate_child_state_with_gt(
                            state,
                            action,
                            score,
                            temp_tokenized_agent,  # Use the updated temp_tokenized_agent instead of original
                            None,  # current_gt_pos
                            None,  # current_gt_heading
                            None,  # current_valid_mask
                            None,  # Not first step, no historical_steps
                            pred_dict=pred_dict
                        )
                        new_beam.append(child_state)
            
            # Sort by score and keep top beam_width states
            new_beam.sort(key=lambda x: x['score'].item(), reverse=True)
            beam = new_beam[:self.beam_width]
        
        # Get best trajectory for this agent
        best_state = max(beam, key=lambda x: x['score'].item())
        return {
            'pred_pos': best_state['pos_a'],  # [1, n_step, 2]
            'pred_head': best_state['head_a'],  # [1, n_step]
            'pred_traj_10hz': self._convert_to_10hz(best_state['pos_a'], best_state['head_a']),  # [1, n_step_10hz, 2]
            'pred_head_10hz': self._convert_to_10hz_heading(best_state['head_a']),  # [1, n_step_10hz]
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
            # For subsequent steps, always use model prediction
            # Get token trajectory directly from token_traj_all using the action as index
            range_a = torch.arange(tokenized_agent["token_traj_all"].shape[0])
            next_token_traj_all = tokenized_agent["token_traj_all"][range_a, action]

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
        
        # Handle case when pred_dict is None (e.g., when using ground truth)
        feat_a = None
        feat_a_t_dict = None
        if pred_dict is not None:
            feat_a = pred_dict.get('feat_a', None)
            feat_a_t_dict = pred_dict.get('feat_a_t_dict', None)
        
        return {
            'pos_a': pos_a,
            'head_a': head_a,
            'pred_valid': pred_valid,
            'pred_idx': pred_idx,
            'score': new_score,
            'feat_a': feat_a,
            'feat_a_t_dict': feat_a_t_dict
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
        # pos_a: [n_agent, n_step_2hz, 2]
        # head_a: [n_agent, n_step_2hz]
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
        # head_a: [n_agent, n_step_2hz]
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
        
        # Get ground truth trajectory length and historical steps
        gt_length = tokenized_agent["gt_pos_raw"].shape[1]
        
        # Calculate historical steps in 2Hz (following agent_decoder.py logic)
        # step_current_10hz = self.model.num_historical_steps - 1  # 10
        # step_current_2hz = step_current_10hz // self.model.shift  # 2
        # Using direct calculation for consistency
        historical_steps = (self.model.num_historical_steps - 1) // getattr(self.model, 'shift', 5)
        
        # Initialize current state using ground truth data with only historical steps
        current_state = {
            'pos': tokenized_agent["gt_pos"][:, :historical_steps].clone(),  # [n_agent, n_historical_step, 2]
            'heading': tokenized_agent["gt_heading"][:, :historical_steps].clone(),  # [n_agent, n_historical_step]
            'valid_mask': tokenized_agent["valid_mask"].clone(),  # [n_agent, n_historical_step]
            'idx': tokenized_agent["gt_idx"][:, :historical_steps].clone()  # [n_agent, n_historical_step]
        }
        
        # Perform closed-loop search
        for search_iter in range(self.max_search_iterations):
            print(f"Closed-loop search iteration {search_iter+1}/{self.max_search_iterations}...")
            # Prepare current iteration input data
            current_tokenized_agent = {
                k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in tokenized_agent.items()
            }
            current_tokenized_agent['pos'] = current_state['pos']
            current_tokenized_agent['heading'] = current_state['heading']
            current_tokenized_agent['valid_mask'] = current_state['valid_mask']
            current_tokenized_agent['idx'] = current_state['idx']
            
            # Run one beam search
            search_result = self.search(tokenized_map, current_tokenized_agent)
            
            # Record search result
            search_history.append(search_result)
            
            # Check if reached ground truth length
            current_step = historical_steps + search_iter
            if current_step + 1 >= gt_length:
                break
            
            # Get current ground truth step
            current_gt_step = current_step
            next_gt_step = current_gt_step + 1
            
            # Update position and heading data using search result for the current agent
            # and ground truth for historical data
            # Keep historical ground truth data and append new search result data
            # Note: We're using search result for the current step, but ground truth for other agents
            new_pos = torch.cat([
                current_state['pos'],
                search_result['pred_pos'][:, -1:]
            ], dim=1)
            new_heading = torch.cat([
                current_state['heading'],
                search_result['pred_head'][:, -1:]
            ], dim=1)
            new_valid_mask = torch.cat([
                current_state['valid_mask'],
                current_state['valid_mask'][:, -1:]
            ], dim=1)
            new_idx = torch.cat([
                current_state['idx'],
                current_state['idx'][:, -1:]
            ], dim=1)
            
            # Update current state
            current_state = {
                'pos': new_pos,
                'heading': new_heading,
                'valid_mask': new_valid_mask,
                'idx': new_idx
            }
        
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