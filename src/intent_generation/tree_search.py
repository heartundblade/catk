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
Tree search core implementation for SMART model trajectory generation
"""

from typing import Dict, List, Optional, Tuple

import torch
from omegaconf import DictConfig

from src.smart.model.smart import SMART
from src.smart.utils import sample_next_token_traj, transform_to_global
from src.intent_generation.beam_search import run_beam_search


class Node:
    """Tree node for trajectory search"""
    
    def __init__(self, state: Dict[str, torch.Tensor], parent=None, action=None, depth=0):
        self.state = state  # Current state (position, heading, etc.)
        self.parent = parent  # Parent node
        self.action = action  # Action taken to reach this node
        self.children = []  # Child nodes
        self.depth = depth  # Node depth
        self.value = 0.0  # Node value
        self.visits = 0  # Number of visits
        self.policy = None  # Policy distribution


class TreeSearch:
    """Tree search implementation for SMART model"""
    
    def __init__(self, model: SMART, config: DictConfig):
        """
        Initialize tree search
        
        Args:
            model: SMART model instance
            config: Tree search configuration
        """
        self.model = model
        self.config = config
        self.max_depth = config.max_depth
        self.max_children = config.max_children
        self.exploration_weight = config.exploration_weight
    
    def search(self, tokenized_map: Dict[str, torch.Tensor], tokenized_agent: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Perform tree search to find optimal trajectory
        
        Args:
            tokenized_map: Tokenized map data
            tokenized_agent: Tokenized agent data
        
        Returns:
            Best trajectory prediction
        """
        # Initialize root node with initial state
        root_state = {
            'pos_a': tokenized_agent["gt_pos"][:, :self.model.num_historical_steps//5].clone(),
            'head_a': tokenized_agent["gt_heading"][:, :self.model.num_historical_steps//5].clone(),
            'pred_valid': tokenized_agent["valid_mask"].clone(),
            'pred_idx': tokenized_agent["gt_idx"].clone(),
        }
        
        root = Node(root_state)
        
        # Perform search iterations
        for _ in range(self.config.num_iterations):
            node = self.select(root)
            value = self.evaluate(node, tokenized_map, tokenized_agent)
            self.backpropagate(node, value)
        
        # Get best trajectory
        best_trajectory = self.get_best_trajectory(root)
        return best_trajectory
    
    def select(self, node: Node) -> Node:
        """
        Select node for expansion using UCT
        
        Args:
            node: Current node
        
        Returns:
            Node to expand
        """
        while node.children and node.depth < self.max_depth:
            node = max(node.children, key=self.uct_score)
        return node
    
    def uct_score(self, node: Node) -> float:
        """
        Calculate UCT score
        
        Args:
            node: Node to evaluate
        
        Returns:
            UCT score
        """
        if node.visits == 0:
            return float('inf')
        
        exploitation = node.value / node.visits
        exploration = self.exploration_weight * torch.sqrt(
            torch.log(node.parent.visits) / node.visits
        ).item()
        
        return exploitation + exploration
    
    def evaluate(self, node: Node, tokenized_map: Dict[str, torch.Tensor], tokenized_agent: Dict[str, torch.Tensor]) -> float:
        """
        Evaluate node by expanding and simulating
        
        Args:
            node: Node to evaluate
            tokenized_map: Tokenized map data
            tokenized_agent: Tokenized agent data
        
        Returns:
            Node value
        """
        if node.depth >= self.max_depth:
            return self._compute_terminal_value(node, tokenized_map, tokenized_agent)
        
        # Expand node
        self.expand(node, tokenized_map, tokenized_agent)
        
        # Simulate from each child
        total_value = 0.0
        for child in node.children:
            value = self.simulate(child, tokenized_map, tokenized_agent)
            child.value = value
            child.visits = 1
            total_value += value
        
        return total_value / len(node.children)
    
    def expand(self, node: Node, tokenized_map: Dict[str, torch.Tensor], tokenized_agent: Dict[str, torch.Tensor]):
        """
        Expand node by generating child states
        
        Args:
            node: Node to expand
            tokenized_map: Tokenized map data
            tokenized_agent: Tokenized agent data
        """
        # Get current state
        pos_a = node.state['pos_a']
        head_a = node.state['head_a']
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
        
        # Sample top-k tokens
        topk_logits, topk_indices = torch.topk(next_token_logits, k=self.max_children, dim=1)
        
        # Create child nodes
        for i in range(self.max_children):
            action = topk_indices[:, i]
            child_state = self._generate_child_state(node.state, action, tokenized_agent)
            child = Node(child_state, parent=node, action=action, depth=node.depth + 1)
            child.policy = topk_logits[:, i]
            node.children.append(child)
    
    def _generate_child_state(self, parent_state: Dict[str, torch.Tensor], action: torch.Tensor, tokenized_agent: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Generate child state from parent state and action
        
        Args:
            parent_state: Parent state
            action: Action (token index)
            tokenized_agent: Tokenized agent data
        
        Returns:
            Child state
        """
        pos_a = parent_state['pos_a']
        head_a = parent_state['head_a']
        pred_valid = parent_state['pred_valid'].clone()
        pred_idx = parent_state['pred_idx'].clone()
        
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
        
        return {
            'pos_a': pos_a,
            'head_a': head_a,
            'pred_valid': pred_valid,
            'pred_idx': pred_idx,
        }
    
    def simulate(self, node: Node, tokenized_map: Dict[str, torch.Tensor], tokenized_agent: Dict[str, torch.Tensor]) -> float:
        """
        Simulate trajectory from node
        
        Args:
            node: Node to simulate from
            tokenized_map: Tokenized map data
            tokenized_agent: Tokenized agent data
        
        Returns:
            Simulation value
        """
        current_state = node.state.copy()
        depth = node.depth
        total_reward = 0.0
        
        while depth < self.max_depth:
            # Create temporary tokenized agent
            temp_tokenized_agent = tokenized_agent.copy()
            temp_tokenized_agent['gt_pos'] = current_state['pos_a']
            temp_tokenized_agent['gt_heading'] = current_state['head_a']
            
            # Get model prediction
            map_feature = self.model.encoder.map_encoder(tokenized_map)
            pred_dict = self.model.encoder.agent_encoder.inference(
                temp_tokenized_agent, map_feature, self.model.validation_rollout_sampling
            )
            
            # Sample action
            next_token_logits = pred_dict['next_token_logits'][:, 0]
            action = torch.argmax(next_token_logits, dim=1)
            
            # Generate next state
            current_state = self._generate_child_state(current_state, action, tokenized_agent)
            
            # Compute reward
            reward = self._compute_reward(current_state, tokenized_agent)
            total_reward += reward
            
            depth += 1
        
        # Add terminal reward
        total_reward += self._compute_terminal_value(node, tokenized_map, tokenized_agent)
        
        return total_reward
    
    def backpropagate(self, node: Node, value: float):
        """
        Backpropagate value up the tree
        
        Args:
            node: Node to start backpropagation from
            value: Value to backpropagate
        """
        while node:
            node.visits += 1
            node.value += value
            node = node.parent
    
    def get_best_trajectory(self, root: Node) -> Dict[str, torch.Tensor]:
        """
        Get best trajectory from root
        
        Args:
            root: Root node
        
        Returns:
            Best trajectory prediction
        """
        current = root
        trajectory = []
        
        while current.children:
            current = max(current.children, key=lambda x: x.value / x.visits if x.visits > 0 else 0)
            trajectory.append(current.action)
        
        # Reconstruct trajectory
        pos_a = root.state['pos_a']
        head_a = root.state['head_a']
        
        return {
            'pred_pos': pos_a,
            'pred_head': head_a,
            'pred_traj_10hz': self._convert_to_10hz(pos_a, head_a),
            'pred_head_10hz': self._convert_to_10hz_heading(head_a),
        }
    
    def _compute_reward(self, state: Dict[str, torch.Tensor], tokenized_agent: Dict[str, torch.Tensor]) -> float:
        """
        Compute reward for state
        
        Args:
            state: Current state
            tokenized_agent: Tokenized agent data
        
        Returns:
            Reward value
        """
        # Simple reward function: distance to ground truth
        pos_a = state['pos_a']
        t_now = pos_a.shape[1] - 1
        
        if t_now < tokenized_agent['gt_pos_raw'].shape[1]:
            gt_pos = tokenized_agent['gt_pos_raw'][:, t_now]
            pred_pos = pos_a[:, -1]
            distance = torch.norm(pred_pos - gt_pos, dim=1).mean().item()
            return -distance  # Negative because we want to minimize distance
        
        return 0.0
    
    def _compute_terminal_value(self, node: Node, tokenized_map: Dict[str, torch.Tensor], tokenized_agent: Dict[str, torch.Tensor]) -> float:
        """
        Compute terminal value for node
        
        Args:
            node: Node to evaluate
            tokenized_map: Tokenized map data
            tokenized_agent: Tokenized agent data
        
        Returns:
            Terminal value
        """
        # Use model's built-in evaluation
        temp_tokenized_agent = tokenized_agent.copy()
        temp_tokenized_agent['gt_pos'] = node.state['pos_a']
        temp_tokenized_agent['gt_heading'] = node.state['head_a']
        
        map_feature = self.model.encoder.map_encoder(tokenized_map)
        pred_dict = self.model.encoder.agent_encoder.inference(
            temp_tokenized_agent, map_feature, self.model.validation_rollout_sampling
        )
        
        # Return negative ADE as value
        if 'pred_traj_10hz' in pred_dict:
            pred_traj = pred_dict['pred_traj_10hz']
            gt_traj = tokenized_agent['gt_pos_raw'][:, self.model.num_historical_steps:, :2]
            
            # Calculate ADE
            ade = torch.norm(pred_traj - gt_traj, dim=2).mean().item()
            return -ade
        
        return 0.0
    
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


def run_tree_search(model: SMART, tokenized_map: Dict[str, torch.Tensor], tokenized_agent: Dict[str, torch.Tensor], config: DictConfig) -> Dict[str, torch.Tensor]:
    """
    Run tree search to generate trajectory
    
    Args:
        model: SMART model instance
        tokenized_map: Tokenized map data
        tokenized_agent: Tokenized agent data
        config: Tree search configuration
    
    Returns:
        Trajectory prediction
    """
    tree_search = TreeSearch(model, config)
    result = tree_search.search(tokenized_map, tokenized_agent)
    
    return result