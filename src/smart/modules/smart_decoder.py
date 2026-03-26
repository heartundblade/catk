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

from typing import Dict, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

from .agent_decoder import SMARTAgentDecoder
from .map_decoder import SMARTMapDecoder


class SMARTDecoder(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        num_historical_steps: int,
        num_future_steps: int,
        pl2pl_radius: float,
        time_span: Optional[int],
        pl2a_radius: float,
        a2a_radius: float,
        num_freq_bands: int,
        num_map_layers: int,
        num_agent_layers: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
        hist_drop_prob: float,
        n_token_agent: int,
    ) -> None:
        super(SMARTDecoder, self).__init__()
        self.map_encoder = SMARTMapDecoder(
            hidden_dim=hidden_dim,
            pl2pl_radius=pl2pl_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_map_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )
        self.agent_encoder = SMARTAgentDecoder(
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            num_future_steps=num_future_steps,
            time_span=time_span,
            pl2a_radius=pl2a_radius,
            a2a_radius=a2a_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_agent_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            hist_drop_prob=hist_drop_prob,
            n_token_agent=n_token_agent,
        )

    def forward(
        self, tokenized_map: Dict[str, Tensor], tokenized_agent: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        map_feature = self.map_encoder(tokenized_map)
        pred_dict = self.agent_encoder(tokenized_agent, map_feature)
        return pred_dict

    def inference(
        self,
        tokenized_map: Dict[str, Tensor],
        tokenized_agent: Dict[str, Tensor],
        sampling_scheme: DictConfig,
    ) -> Dict[str, Tensor]:
        map_feature = self.map_encoder(tokenized_map)
        pred_dict = self.agent_encoder.inference(
            tokenized_agent, map_feature, sampling_scheme
        )
        return pred_dict

    def inference_val(
        self,
        map_feature: Dict[str, Tensor],
        tokenized_agent: Dict[str, Tensor],
        sampling_scheme: DictConfig,
    ) -> Dict[str, Tensor]:
        pred_dict = self.agent_encoder.inference(
            tokenized_agent, map_feature, sampling_scheme
        )
        return pred_dict
    
    def inference_single_step(
        self,
        tokenized_map: Dict[str, Tensor],
        tokenized_agent: Dict[str, Tensor],
        prev_feat_a: torch.Tensor = None,
        prev_feat_a_t_dict: Dict = None,
        is_initial_step: bool = True,
    ) -> Dict[str, Tensor]:
        """
        Single step inference for beam search
        
        Args:
            tokenized_map: Tokenized map data
            tokenized_agent: Tokenized agent data
            prev_feat_a: Previous agent features (for non-initial steps)
            prev_feat_a_t_dict: Previous feature dictionary (for non-initial steps)
            is_initial_step: Whether this is the initial step
        
        Returns:
            Dictionary with next token logits and intermediate features
        """
        map_feature = self.map_encoder(tokenized_map)
        pred_dict = self.agent_encoder.inference_single_step(
            tokenized_agent, 
            map_feature, 
            prev_feat_a=prev_feat_a,
            prev_feat_a_t_dict=prev_feat_a_t_dict,
            is_initial_step=is_initial_step
        )
        return pred_dict