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

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch_cluster import radius, radius_graph
from torch_geometric.utils import dense_to_sparse, subgraph

from src.smart.modules.agent_decoder import SMARTAgentDecoder
from src.smart.utils import (
    angle_between_2d_vectors,
    sample_next_token_traj,
    sample_next_token_traj_parallel,
    transform_to_global,
    transform_to_global_parallel,
    wrap_angle,
)


class ParSMARTAgentDecoder(SMARTAgentDecoder):
    """
    Batch version of SMARTAgentDecoder that can process multiple batches in parallel
    """

    def __init__(
        self,
        hidden_dim: int,
        num_historical_steps: int,
        num_future_steps: int,
        time_span: Optional[int],
        pl2a_radius: float,
        a2a_radius: float,
        num_freq_bands: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
        hist_drop_prob: float,
        n_token_agent: int,
        n_parallel: int,
    ) -> None:
        super(ParSMARTAgentDecoder, self).__init__(
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            num_future_steps=num_future_steps,
            time_span=time_span,
            pl2a_radius=pl2a_radius,
            a2a_radius=a2a_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            hist_drop_prob=hist_drop_prob,
            n_token_agent=n_token_agent,
        )
        self.n_parallel = n_parallel
    
    def batch_agent_token_embedding(
            self,
            agent_token_index,  # [n_parallel, n_agent, n_step]
            trajectory_token_veh,  # [n_parallel, n_token_veh, 8]
            trajectory_token_ped,  # [n_parallel, n_token_ped, 8]
            trajectory_token_cyc,  # [n_parallel, n_token_cyc, 8]
            pos_a,  # [n_parallel, n_agent, n_step, 2]
            head_vector_a,  # [n_parallel, n_agent, n_step, 2]
            agent_type,  # [n_parallel, n_agent]
            agent_shape,  # [n_parallel, n_agent]
            inference=False):
        '''
        Batch version of agent token embedding for beam search inference
        '''
        n_agent, n_step, traj_dim = pos_a.shape
        _device = pos_a.device

        veh_mask = agent_type == 0
        ped_mask = agent_type == 1
        cyc_mask = agent_type == 2
        #  [n_token, hidden_dim]
        agent_token_emb_veh = self.token_emb_veh(trajectory_token_veh)
        agent_token_emb_ped = self.token_emb_ped(trajectory_token_ped)
        agent_token_emb_cyc = self.token_emb_cyc(trajectory_token_cyc)
        agent_token_emb = torch.zeros(
            (n_agent, n_step, self.hidden_dim), device=_device, dtype=pos_a.dtype
        )
        agent_token_emb[veh_mask] = agent_token_emb_veh[agent_token_index[veh_mask]]
        agent_token_emb[ped_mask] = agent_token_emb_ped[agent_token_index[ped_mask]]
        agent_token_emb[cyc_mask] = agent_token_emb_cyc[agent_token_index[cyc_mask]]

        motion_vector_a = torch.cat(
            [
                pos_a.new_zeros(agent_token_index.shape[0], 1, traj_dim),
                pos_a[:, 1:] - pos_a[:, :-1],
            ],
            dim=1,
        )  # [n_agent, n_step, 2]
        feature_a = torch.stack(
            [
                torch.norm(motion_vector_a[:, :, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_a, nbr_vector=motion_vector_a[:, :, :2]
                ),
            ],
            dim=-1,
        )  # [n_agent, n_step, 2]
        categorical_embs = [
            self.type_a_emb(agent_type.long()),
            self.shape_emb(agent_shape),
        ]  # List of len=2, shape [n_agent, hidden_dim]

        x_a = self.x_a_emb(
            continuous_inputs=feature_a.view(-1, feature_a.size(-1)),
            categorical_embs=[
                v.repeat_interleave(repeats=n_step, dim=0) for v in categorical_embs
            ],
        )  # [n_agent*n_step, hidden_dim]
        x_a = x_a.view(-1, n_step, self.hidden_dim)  # [n_agent, n_step, hidden_dim]

        feat_a = torch.cat((agent_token_emb, x_a), dim=-1)
        feat_a = self.fusion_emb(feat_a)

        if inference:
            return (
                feat_a,  # [n_agent, n_step, hidden_dim]
                agent_token_emb,  # [n_agent, n_step, hidden_dim]
                agent_token_emb_veh,  # [n_agent, hidden_dim]
                agent_token_emb_ped,  # [n_agent, hidden_dim]
                agent_token_emb_cyc,  # [n_agent, hidden_dim]
                veh_mask,  # [n_agent]
                ped_mask,  # [n_agent]
                cyc_mask,  # [n_agent]
                categorical_embs,  # List of len=2, shape [n_agent, hidden_dim]
            )
        else:
            return feat_a  # [n_agent, n_step, hidden_dim]
    
    def batch_build_temporal_edge(
        self,
        pos_a_,  # [n_parallel, n_agent, n_step, 2]
        head_a_,  # [n_parallel, n_agent, n_step]
        head_vector_a_,  # [n_parallel, n_agent, n_step, 2]
        mask_,  # [n_parallel, n_agent, n_step]
        inference_mask_=None,  # [n_parallel, n_agent, n_step]
    ):
        """
        Batch version of temporal edge construction
        """
        pos_t_ = pos_a_.flatten(1, 2)
        head_t_ = head_a_.flatten(1, 2)
        head_vector_t_ = head_vector_a_.flatten(1, 2)

        if self.hist_drop_prob > 0 and self.training:
            _mask_keep_ = torch.bernoulli(
                torch.ones_like(mask_) * (1 - self.hist_drop_prob)
            ).bool()
            mask_ = mask_ & _mask_keep_
        
        if inference_mask_ is not None:
            mask_t_ = mask_.unsqueeze(3) & inference_mask_.unsqueeze(2)
        else:
            mask_t_ = mask_.unsqueeze(3) & mask_.unsqueeze(2)
        
        n_parallel = mask_t_.shape[0]
        all_edge_index_t = []
        all_r_t = []

        for i in range(n_parallel):
            mask_t = mask_t_[i]
            pos_t = pos_t_[i]
            head_t = head_t_[i]
            head_vector_t = head_vector_t_[i]
            
            edge_index_t = dense_to_sparse(mask_t)[0]  # [2, num_edge]
            # print(edge_index_t.shape, mask_t.shape, i)
            edge_index_t = edge_index_t[:, edge_index_t[1] > edge_index_t[0]]
            edge_index_t = edge_index_t[
                :, edge_index_t[1] - edge_index_t[0] <= self.time_span / self.shift
            ]

            rel_pos_t = pos_t[edge_index_t[0]] - pos_t[edge_index_t[1]]
            rel_pos_t = rel_pos_t[:, :2]
            rel_head_t = wrap_angle(
                head_t[edge_index_t[0]] - head_t[edge_index_t[1]]
                )
            r_t = torch.stack(
                [
                    torch.norm(rel_pos_t, p=2, dim=-1),
                    angle_between_2d_vectors(
                        ctr_vector=head_vector_t[edge_index_t[1]], nbr_vector=rel_pos_t
                    ),
                    rel_head_t,
                    edge_index_t[0] - edge_index_t[1],
                ],
                dim=-1,
            )
            all_edge_index_t.append(edge_index_t)
            all_r_t.append(r_t)
        
        edge_index_t_ = torch.cat(all_edge_index_t, dim=1)
        r_t_ = torch.cat(all_r_t, dim=0)

        r_t_ = self.r_t_emb(continuous_inputs=r_t_, categorical_embs=None)

        # edge_index_t = dense_to_sparse(mask_t)[0]  # [num_edge, 2]
        # edge_index_t = edge_index_t[:, edge_index_t[:, 1] > edge_index_t[:, 0]]
        # edge_index_t = edge_index_t[
        #     :, edge_index_t[:, 1] - edge_index_t[:, 0] <= self.time_span / self.shift
        # ]

        # rel_pos_t = pos_t[edge_index_t[:, 0]] - pos_t[edge_index_t[:, 1]]
        # rel_pos_t = rel_pos_t[:, :, :2]
        # rel_head_t = wrap_angle(
        #     head_t[edge_index_t[:, 0]] - head_t[edge_index_t[:, 1]]
        #     )
        # r_t = torch.stack(
        #     [
        #         torch.norm(rel_pos_t, p=2, dim=-1),
        #         angle_between_2d_vectors(
        #             ctr_vector=head_vector_t[edge_index_t[:, 1]], nbr_vector=rel_pos_t
        #         ),
        #         rel_head_t,
        #         edge_index_t[:, 0] - edge_index_t[:, 1],
        #     ],
        #     dim=-1,
        # )
        # r_t = self.r_t_emb(continuous_inputs=r_t, categorical_embs=None)
        return edge_index_t_, r_t_
    
    def batch_build_interaction_edge(
        self,
        pos_a_,  # [n_parallel, n_agent, n_step, 2]
        head_a_,  # [n_parallel, n_agent, n_step]
        head_vector_a_,  # [n_parallel, n_agent, n_step, 2]
        batch_s_,  # [n_parallel, n_agent*n_step]
        mask_,  # [n_parallel, n_agent, n_step]
    ):
        """
        Batch version of interaction edge construction
        """
        n_parallel = pos_a_.shape[0]
        mask_ = mask_.transpose(1, 2).reshape(n_parallel, -1)  # [n_parallel, n_agent*n_step]
        pos_s_ = pos_a_.transpose(1, 2).flatten(1, 2)  # [n_parallel, n_agent*n_step, 2]
        head_s_ = head_a_.transpose(1, 2).reshape(n_parallel, -1)  # [n_parallel, n_agent*n_step]
        head_vector_s_ = head_vector_a_.transpose(1, 2).reshape(n_parallel, -1, 2)  # [n_parallel, n_agent*n_step, 2]

        all_edge_index_a2a = []
        all_r_a2a = []
        for i in range(n_parallel):
            edge_index_a2a = radius_graph(
                x=pos_s_[i, :, :2],
                r=self.a2a_radius,
                batch=batch_s_[i],
                loop = False,
                max_num_neighbors=300,
            )
            edge_index_a2a = subgraph(subset=mask_[i], edge_index=edge_index_a2a)[0]
            rel_pos_a2a = pos_s_[i, edge_index_a2a[0]] - pos_s_[i, edge_index_a2a[1]]
            rel_head_a2a = wrap_angle(
                head_s_[i, edge_index_a2a[0]] - head_s_[i, edge_index_a2a[1]]
            )
            r_a2a = torch.stack(
                [
                    torch.norm(rel_pos_a2a[:, :2], p=2, dim=-1),
                    angle_between_2d_vectors(
                        ctr_vector=head_vector_s_[i, edge_index_a2a[1]], 
                        nbr_vector=rel_pos_a2a[:, :2]
                    ),
                    rel_head_a2a,
                ],
                dim=-1,
            )
            all_edge_index_a2a.append(edge_index_a2a)
            all_r_a2a.append(r_a2a)
        
        edge_index_a2a_ = torch.cat(all_edge_index_a2a, dim=1)
        r_a2a_ = torch.cat(all_r_a2a, dim=0)

        r_a2a_ = self.r_a2a_emb(continuous_inputs=r_a2a_, categorical_embs=None)

        # edge_index_a2a_ = radius_graph(
        #     x=pos_s_[:, :, :2],
        #     r=self.a2a_radius,
        #     batch=batch_s_,
        #     loop = False,
        #     max_num_neighbors=300,
        # )
        # edge_index_a2a_ = subgraph(subset=mask_, edge_index=edge_index_a2a_)[0]
        # rel_pos_a2a_ = pos_s_[:, edge_index_a2a_[0]] - pos_s_[:, edge_index_a2a_[1]]
        # rel_head_a2a_ = wrap_angle(
        #     head_s_[:, edge_index_a2a_[0]] - head_s_[:, edge_index_a2a_[1]]
        # )
        # r_a2a_ = torch.stack(
        #     [
        #         torch.norm(rel_pos_a2a_, p=2, dim=-1),
        #         angle_between_2d_vectors(
        #             ctr_vector=head_vector_s_[edge_index_a2a_[1]], nbr_vector=rel_pos_a2a_
        #         ),
        #         rel_head_a2a_,
        #     ],
        #     dim=-1,
        # )
        # r_a2a_ = self.r_a2a_emb(continuous_inputs=r_a2a_, categorical_embs=None)
        return edge_index_a2a_, r_a2a_
    
    def batch_build_map2agent_edge(
        self,
        pos_pl: torch.Tensor,  # [n_pl, 2]
        orient_pl,  # [n_pl]
        pos_a_,  # [n_parallel, n_agent, n_step, 2]
        head_a_,  # [n_parallel, n_agent, n_step]
        head_vector_a_,  # [n_parallel, n_agent, n_step, 2]
        mask_,  # [n_parallel, n_agent, n_step]
        batch_s_,  # [n_parallel, n_agent*n_step]
        batch_pl_,  # [n_parallel, n_pl*n_step]
    ):
        """
        Batch version of map-to-agent edge construction
        """
        n_step = pos_a_.shape[2]
        n_parallel = pos_a_.shape[0]
        mask_pl2a_ = mask_.transpose(1, 2).reshape(n_parallel, -1)  # [n_parallel, n_agent*n_step]
        pos_s_ = pos_a_.transpose(1, 2).flatten(1, 2)  # [n_parallel, n_agent*n_step, 2]
        head_s_ = head_a_.transpose(1, 2).reshape(n_parallel, -1)  # [n_parallel, n_agent*n_step]
        head_vector_s_ = head_vector_a_.transpose(1, 2).reshape(n_parallel, -1, 2)  # [n_parallel, n_agent*n_step, 2]
        pos_pl_ = pos_pl.unsqueeze(0).repeat(n_parallel, n_step, 1)  # [n_parallel, n_pl*n_step, 2]
        orient_pl_ = orient_pl.unsqueeze(0).repeat(n_parallel, n_step)  # [n_parallel, n_pl*n_step]
        
        all_edge_index_pl2a = []
        all_r_pl2a = []
        for i in range(n_parallel):
            edge_index_pl2a = radius(
                x=pos_s_[i, :, :2],
                y=pos_pl_[i, :, :2],
                r=self.pl2a_radius,
                batch_x=batch_s_[i],
                batch_y=batch_pl_[i],
                max_num_neighbors=300,
            )
            edge_index_pl2a = edge_index_pl2a[:, mask_pl2a_[i, edge_index_pl2a[1]]]
            rel_pos_pl2a = pos_pl_[i, edge_index_pl2a[0]] - pos_s_[i, edge_index_pl2a[1]]
            rel_orient_pl2a = wrap_angle(
                orient_pl_[i, edge_index_pl2a[0]] - head_s_[i, edge_index_pl2a[1]]
            )
            r_pl2a = torch.stack(
                [
                    torch.norm(rel_pos_pl2a[:, :2], p=2, dim=-1),
                    angle_between_2d_vectors(
                        ctr_vector=head_vector_s_[i, edge_index_pl2a[1]], 
                        nbr_vector=rel_pos_pl2a[:, :2]
                    ),
                    rel_orient_pl2a,
                ],
                dim=-1,
            )
            all_edge_index_pl2a.append(edge_index_pl2a)
            all_r_pl2a.append(r_pl2a)

        edge_index_pl2a_ = torch.cat(all_edge_index_pl2a, dim=1)
        r_pl2a_ = torch.cat(all_r_pl2a, dim=0)

        r_pl2a_ = self.r_pt2a_emb(continuous_inputs=r_pl2a_, categorical_embs=None)

        # edge_index_pl2a_ = radius(
        #     x=pos_s_[:, :, :2],
        #     y=pos_pl_[:, :, :2],
        #     r=self.pl2a_radius,
        #     batch_x=batch_s_,
        #     batch_y=batch_pl_,
        #     max_num_neighbors=300,
        # )
        # edge_index_pl2a_ = edge_index_pl2a_[:, mask_pl2a_[:, edge_index_pl2a_[:, 1]]]
        # rel_pos_pl2a_ = pos_pl_[:, edge_index_pl2a_[:, 0]] - pos_s_[:, edge_index_pl2a_[:, 1]]
        # rel_orient_pl2a_ = wrap_angle(
        #     orient_pl_[:, edge_index_pl2a_[:, 0]] - head_s_[:, edge_index_pl2a_[:, 1]]
        # )
        # r_pl2a_ = torch.stack(
        #     [
        #         torch.norm(rel_pos_pl2a_[:, :, :2], p=2, dim=-1),
        #         angle_between_2d_vectors(
        #             ctr_vector=head_vector_s_[edge_index_pl2a_[:, 1]], nbr_vector=rel_pos_pl2a_
        #         ),
        #         rel_orient_pl2a_,
        #     ],
        #     dim=-1,
        # )
        # r_pl2a_ = self.r_pt2a_emb(continuous_inputs=r_pl2a_, categorical_embs=None)
        
        return edge_index_pl2a_, r_pl2a_
    
    def inference(
        self,
        tokenized_agent: Dict[str, torch.Tensor],
        map_feature: Dict[str, torch.Tensor],
        sampling_scheme: DictConfig,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Run parallel inference for sampling multiple scenario rollouts

        Args:
            tokenized_agent: tokenized agent data
            map_feature: map features
            sampling_scheme: Sampling scheme configuration
        Returns:
            List of output dictionaries for each scenario rollout
        """
        n_agent = tokenized_agent["valid_mask"].shape[0]
        n_step_future_10hz = self.num_future_steps  # 80
        n_step_future_2hz = n_step_future_10hz // self.shift  # 16
        step_current_10hz = self.num_historical_steps - 1  # 10
        step_current_2hz = step_current_10hz // self.shift  # 2

        pos_a = tokenized_agent["gt_pos"][:, :step_current_2hz].clone()  # [n_agent, n_step, 2]
        head_a = tokenized_agent["gt_heading"][:, :step_current_2hz].clone()  # [n_agent, n_step]
        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)  # [n_agent, n_step, 2]
        pred_idx = tokenized_agent["gt_idx"].clone()

        pos_a_ = pos_a.unsqueeze(0).repeat(
            self.n_parallel, 1, 1, 1
            )  # [n_parallel, n_agent, n_step, 2]
        head_a_ = head_a.unsqueeze(0).repeat(
            self.n_parallel, 1, 1
            )  # [n_parallel, n_agent, n_step]
        head_vector_a_ = head_vector_a.unsqueeze(0).repeat(
            self.n_parallel, 1, 1, 1
            )  # [n_parallel, n_agent, n_step, 2]
        pred_idx_ = pred_idx.unsqueeze(0).repeat(
            self.n_parallel, 1, 1
            )  # [n_parallel, n_agent, n_step]
        
        device = tokenized_agent["valid_mask"].device
        (
            feat_a,  # [n_agent, step_current_2hz, hidden_dim]
            agent_token_emb,  # [n_agent, step_current_2hz, hidden_dim]
            agent_token_emb_veh,  # [n_agent, hidden_dim]
            agent_token_emb_ped,  # [n_agent, hidden_dim]
            agent_token_emb_cyc,  # [n_agent, hidden_dim]
            veh_mask,  # [n_agent]
            ped_mask,  # [n_agent]
            cyc_mask,  # [n_agent]
            categorical_embs,  # List of len=2, shape [n_agent, hidden_dim]
        ) = self.agent_token_embedding(
            agent_token_index=tokenized_agent["gt_idx"][:, :step_current_2hz],
            trajectory_token_veh=tokenized_agent["trajectory_token_veh"],
            trajectory_token_ped=tokenized_agent["trajectory_token_ped"],
            trajectory_token_cyc=tokenized_agent["trajectory_token_cyc"],
            pos_a=pos_a,
            head_vector_a=head_vector_a,
            agent_type=tokenized_agent["type"],
            agent_shape=tokenized_agent["shape"],
            inference=True,
        )

        agent_token_emb_ = agent_token_emb.unsqueeze(0).repeat(
            self.n_parallel, 1, 1, 1
            )  # [n_parallel, n_agent, step_current_2hz, hidden_dim]
        agent_token_emb_veh_ = agent_token_emb_veh.unsqueeze(0).repeat(
            self.n_parallel, 1, 1
            )  # [n_parallel, n_agent, hidden_dim]
        agent_token_emb_ped_ = agent_token_emb_ped.unsqueeze(0).repeat(
            self.n_parallel, 1, 1
            )  # [n_parallel, n_agent, hidden_dim]
        agent_token_emb_cyc_ = agent_token_emb_cyc.unsqueeze(0).repeat(
            self.n_parallel, 1, 1
            )  # [n_parallel, n_agent, hidden_dim]
        veh_mask_ = veh_mask.unsqueeze(0).repeat(
            self.n_parallel, 1
            )  # [n_parallel, n_agent]
        ped_mask_ = ped_mask.unsqueeze(0).repeat(
            self.n_parallel, 1
            )  # [n_parallel, n_agent]
        cyc_mask_ = cyc_mask.unsqueeze(0).repeat(
            self.n_parallel, 1
            )  # [n_parallel, n_agent]
        categorical_embs_ = [emb.unsqueeze(0).repeat(
            self.n_parallel, 1, 1
            ) for emb in categorical_embs]  # [n_parallel, n_agent, hidden_dim]

        feat_a_ = feat_a.unsqueeze(0).repeat(
            self.n_parallel, 1, 1, 1
            )  # [n_parallel, n_agent, step_current_2hz, hidden_dim]

        if not self.training:
            pred_traj_10hz_ = torch.zeros(
                [self.n_parallel, n_agent, n_step_future_10hz, 2], 
                dtype=pos_a.dtype, device=pos_a.device
            )
            pred_head_10hz_ = torch.zeros(
                [self.n_parallel, n_agent, n_step_future_10hz], 
                dtype=pos_a.dtype, device=pos_a.device
            )
        
        pred_valid = tokenized_agent["valid_mask"].clone()
        pred_valid_ = pred_valid.unsqueeze(0).repeat(self.n_parallel, 1, 1)  # [n_parallel, n_agent, n_step]
        next_token_logits_list = []
        next_token_action_list = []
        feat_a_t_dict_ = {}

        for t in range(n_step_future_2hz):  # 0 -> 15
            t_now = step_current_2hz - 1 + t  # 1 -> 16
            n_step = t_now + 1  # 2 -> 17

            if t == 0:
                hist_step = step_current_2hz
                batch_s = torch.cat(
                    [
                        tokenized_agent["batch"] + tokenized_agent["num_graphs"] * t
                        for t in range(hist_step)
                    ], dim=0
                )  # [n_agent*hist_step]
                batch_pl = torch.cat(
                    [
                        map_feature["batch"] + tokenized_agent["num_graphs"] * t
                        for t in range(hist_step)
                    ], dim=0
                )  # [n_pl*hist_step]
                
                batch_s_ = batch_s.unsqueeze(0).repeat(self.n_parallel, 1)  # [n_parallel, n_agent*hist_step]
                batch_pl_ = batch_pl.unsqueeze(0).repeat(self.n_parallel, 1)  # [n_parallel, n_pl*hist_step]
                inference_mask_ = pred_valid_[:, :, :n_step]  # [n_parallel, n_agent, n_step]

                edge_index_t_, r_t_ = self.batch_build_temporal_edge(
                    pos_a_=pos_a_,
                    head_a_=head_a_,
                    head_vector_a_=head_vector_a_,
                    mask_=pred_valid_[:, :, :n_step],
                )
            else:
                hist_step = 1
                batch_s_ = tokenized_agent["batch"].unsqueeze(0).repeat(self.n_parallel, 1)  # [n_parallel, n_agent*hist_step]
                batch_pl_ = map_feature["batch"].unsqueeze(0).repeat(self.n_parallel, 1)  # [n_parallel, n_pl*hist_step]
                inference_mask_ = pred_valid_[:, :, :n_step].clone()  # [n_parallel, n_agent, n_step]
                inference_mask_[:, :, :-1] = False  # Only keep the last step for inference
                
                edge_index_t_, r_t_ = self.batch_build_temporal_edge(
                    pos_a_=pos_a_,
                    head_a_=head_a_,
                    head_vector_a_=head_vector_a_,
                    mask_=pred_valid_[:, :, :n_step],
                    inference_mask_=inference_mask_,
                )
                edge_index_t_[:, 1] = (edge_index_t_[:, 1] + 1) // n_step - 1

            edge_index_pl2a_, r_pl2a_ = self.batch_build_map2agent_edge(
                pos_pl=map_feature["position"],  # [n_pl, 2]
                orient_pl=map_feature["orientation"],  # [n_pl]
                pos_a_=pos_a_[:, :, -hist_step:],  # [n_parallel, n_agent, hist_step, 2]
                head_a_=head_a_[:, :, -hist_step:],  # [n_parallel, n_agent, hist_step]
                head_vector_a_=head_vector_a_[:, :, -hist_step:],  # [n_parallel, n_agent, hist_step, 2]
                mask_=inference_mask_[:, :, -hist_step:],  # [n_parallel, n_agent, hist_step]
                batch_s_=batch_s_,  # [n_parallel, n_agent*hist_step]
                batch_pl_=batch_pl_,  # [n_parallel, n_pl*hist_step]
            )
            edge_index_a2a_, r_a2a_ = self.batch_build_interaction_edge(
                pos_a_=pos_a_[:, :, -hist_step:],  # [n_parallel, n_agent, hist_step, 2]
                head_a_=head_a_[:, :, -hist_step:],  # [n_parallel, n_agent, hist_step]
                head_vector_a_=head_vector_a_[:, :, -hist_step:],  # [n_parallel, n_agent, hist_step, 2]
                batch_s_=batch_s_,  # [n_parallel, n_agent*hist_step]
                mask_=inference_mask_[:, :, -hist_step:],  # [n_parallel, n_agent, hist_step]
            )

            # attention layers
            for i in range(self.num_layers):
                # [n_parallel, n_agent, hist_step, hidden_dim]
                _feat_temporal_ = feat_a_ if i == 0 else feat_a_t_dict_[i]

                if t == 0:
                    # Process all historical steps together
                    _feat_temporal_ = self.t_attn_layers[i](
                        _feat_temporal_.flatten(1, 2), r_t_, edge_index_t_
                    ).view(self.n_parallel, n_agent, n_step, -1)
                    _feat_temporal_ = _feat_temporal_.transpose(1, 2).flatten(1, 2)  # [n_parallel, n_agent*n_step, hidden_dim]
                    
                    # Expand map features
                    _feat_map_ = (
                        map_feature["pt_token"]
                        .unsqueeze(0)
                        .repeat(hist_step, 1, 1)
                        .flatten(0, 1)
                    ).unsqueeze(0).repeat(self.n_parallel, 1, 1)  # [n_parallel, n_pl*hist_step, hidden_dim]
                    
                    # Process map2agent attention
                    _feat_temporal_ = self.pt2a_attn_layers[i](
                        (_feat_map_, _feat_temporal_), r_pl2a_, edge_index_pl2a_
                    )
                    
                    # Process agent2agent attention
                    _feat_temporal_ = self.a2a_attn_layers[i](
                        _feat_temporal_, r_a2a_, edge_index_a2a_
                    )
                    
                    _feat_temporal_ = _feat_temporal_.view(self.n_parallel, n_step, n_agent, -1).transpose(1, 2)  # [n_parallel, n_agent, n_step, hidden_dim]
                    feat_a_now_ = _feat_temporal_[:, :, -1]  # [n_parallel, n_agent, hidden_dim]
                    
                    if i + 1 < self.num_layers:
                        feat_a_t_dict_[i + 1] = _feat_temporal_
                else:
                    feat_a_now_ = self.t_attn_layers[i](
                        (_feat_temporal_.flatten(0, 1), _feat_temporal_[:, :, -1]), r_t_, edge_index_t_
                        )
                    
                    feat_a_now_ = self.pt2a_attn_layers[i](
                        (map_feature["pt_token"], feat_a_now_), r_pl2a_, edge_index_pl2a_
                    )  # [n_parallel, n_agent, hidden_dim]
                    feat_a_now_ = self.a2a_attn_layers[i](
                        feat_a_now_, r_a2a_, edge_index_a2a_
                    )  # [n_parallel, n_agent, hidden_dim]

                    # [n_parallel, n_agent, n_step, hidden_dim]
                    if i + 1 < self.num_layers:
                        feat_a_t_dict_[i + 1] = torch.cat(
                            (feat_a_t_dict_[i + 1], feat_a_now_.unsqueeze(2)), 
                            dim=2
                        )
            # get outputs
            next_token_logits_ = self.token_predict_head(feat_a_now_)  # [n_parallel, n_agent, n_token]
            next_token_logits_list.append(next_token_logits_)

            # Run parallel version of sampling
            next_token_idx_, next_token_traj_all_ = sample_next_token_traj_parallel(
                    token_traj=tokenized_agent["token_traj"],
                    token_traj_all=tokenized_agent["token_traj_all"],
                    sampling_scheme=sampling_scheme,
                    # ! for most-likely sampling
                    next_token_logits=next_token_logits_,
                    # ! for nearest-pos sampling
                    pos_now=pos_a_[:, :, -1, t_now],  # [n_parallel, n_agent, 2]
                    head_now=head_a_[:, :, t_now],  # [n_parallel, n_agent]
                    pos_next_gt=tokenized_agent["gt_pos_raw"][:, n_step],  # [n_agent, 2]
                    head_next_gt=tokenized_agent["gt_head_raw"][:, n_step],  # [n_agent]
                    valid_next_gt=tokenized_agent["gt_valid_raw"][:, n_step],  # [n_agent]
                    token_agent_shape=tokenized_agent["token_agent_shape"],  # [n_token, 2]
                )  # next_token_idx: [n_parallel, n_agent], next_token_traj_all: [n_parallel, n_agent, 6, 4, 2]

            diff_xy_ = next_token_traj_all_[:, :, -1, 0] - next_token_traj_all_[:, :, -1, 3]
            next_token_action_list.append(
                    torch.cat(
                        [
                            next_token_traj_all_[:, :, -1].mean(1),  # [n_parallel, n_agent, 2]
                            torch.arctan2(diff_xy_[:, :, [1]], diff_xy_[:, :, [0]]),  # [n_parallel, n_agent, 1]
                        ],
                        dim=-1,
                    )  # [n_parallel, n_agent, 3]
                )
            
            # Run parallel version of transform_to_global
            token_traj_global_ = transform_to_global_parallel(
                    pos_local=next_token_traj_all_.flatten(1, 2),  # [n_parallel, n_agent, 6*4, 2]
                    head_local=None,
                    pos_now=pos_a_[:, :, t_now],  # [n_parallel, n_agent, 2]
                    head_now=head_a_[:, :, t_now],  # [n_parallel, n_agent]
                )[0].view(*next_token_traj_all_.shape)
            
            if not self.training:
                pred_traj_10hz_[:, :, t * 5 : (t + 1) * 5] = token_traj_global_[:, :, 1:].mean(
                        2
                    )
                diff_xy_ = token_traj_global_[:, 1:, 0] - token_traj_global_[:, 1:, 3]
                pred_head_10hz_[:, :, t * 5 : (t + 1) * 5] = torch.arctan2(
                    diff_xy_[:, :, 1], diff_xy_[:, :, 0]
                )

            # get pos_a_next and head_a_next, spawn unseen agents
            pos_a_next_ = token_traj_global_[:, -1].mean(dim=1)
            diff_xy_next_ = token_traj_global_[:, -1, 0] - token_traj_global_[:, -1, 3]
            head_a_next_ = torch.arctan2(diff_xy_next_[:, 1], diff_xy_next_[:, 0])
            pred_idx_[:, n_step] = next_token_idx_

            # update tensors for next step
            pred_valid_[:, :, n_step] = pred_valid_[:, :, t_now]
            pos_a_ = torch.cat((pos_a_, pos_a_next_.unsqueeze(2)), dim=2)
            head_a_ = torch.cat((head_a_, head_a_next_.unsqueeze(2)), dim=2)
            head_vector_a_next_ = torch.stack(
                [head_a_next_.cos(), head_a_next_.sin()], dim=-1
                )
            head_vector_a_ = torch.cat(
                (head_vector_a_, head_vector_a_next_.unsqueeze(2)), dim=2
                )
            
            # get agent_token_emb_next
            agent_token_emb_next_ = torch.zeros_like(agent_token_emb_[:, :, 0])
            agent_token_emb_next_[veh_mask_] = agent_token_emb_veh_[
                next_token_idx_[veh_mask_]
            ]
            agent_token_emb_next_[ped_mask_] = agent_token_emb_ped_[
                next_token_idx_[ped_mask_]
            ]
            agent_token_emb_next_[cyc_mask_] = agent_token_emb_cyc_[
                next_token_idx_[cyc_mask_]
            ]
            agent_token_emb_ = torch.cat(
                [agent_token_emb_, agent_token_emb_next_.unsqueeze(2)], dim=2
            )

            # get fear_a_next
            motion_vector_a_ = pos_a_[:, :, -1] - pos_a_[:, :, -2]
            x_a_ = torch.stack(
                [
                    torch.norm(motion_vector_a_, p=2, dim=-1),
                    angle_between_2d_vectors(
                        ctr_vector=head_vector_a_[:, :, -1], nbr_vector=motion_vector_a_
                        )
                ]
            )
            # [n_parallel, n_agent, hidden_dim]
            x_a_ = self.x_a_emb(continuous_inputs=x_a_, categorical_embs=categorical_embs_)
            # [n_parallel, n_agent, 1, 2*hidden_dim]
            feat_a_next_ = torch.cat(
                (agent_token_emb_next_, x_a_), dim=-1
            )
            feat_a_next_ = self.fusion_emb(feat_a_next_)
            feat_a = torch.cat([feat_a, feat_a_next_], dim=2)

        # Process each batch
        out_dict = {
            # action that goes from [(10->15), ..., (85->90)]
            "next_token_logits": torch.stack(next_token_logits_list, dim=1),
            "next_token_valid": pred_valid_[:, :, 1:-1],  # [n_parallel, n_agent, 16]
            # for step {5, 10, ..., 90} and act [(0->5), (5->10), ..., (85->90)]
            "pred_pos": pos_a_,  # [n_parallel, n_agent, 18, 2]
            "pred_head": head_a_,  # [n_parallel, n_agent, 18]
            "pred_valid": pred_valid_,  # [n_parallel, n_agent, 18]
            "pred_idx": pred_idx_,  # [n_parallel, n_agent, 18]
            # for step {5, 10, ..., 90}
            "gt_pos_raw": tokenized_agent["gt_pos_raw"],  # [n_agent, 18, 2]
            "gt_head_raw": tokenized_agent["gt_head_raw"],  # [n_agent, 18]
            "gt_valid_raw": tokenized_agent["gt_valid_raw"],  # [n_agent, 18]
            # or use the tokenized gt
            "gt_pos": tokenized_agent["gt_pos"],  # [n_agent, 18, 2]
            "gt_head": tokenized_agent["gt_heading"],  # [n_agent, 18]
            "gt_valid": tokenized_agent["valid_mask"],  # [n_agent, 18]
            # for shifting proxy targets by lr
            "next_token_action": torch.stack(next_token_action_list, dim=1),
        }
        if not self.training:  # 10hz predictions for wosac evaluation and submission
            out_dict["pred_traj_10hz"] = pred_traj_10hz_
            out_dict["pred_head_10hz"] = pred_head_10hz_
            pred_z = tokenized_agent["gt_z_raw"].unsqueeze(1)  # [n_agent, 1]
            out_dict["pred_z_10hz"] = pred_z.repeat(1, pred_traj_10hz_.shape[2])
        
        return out_dict

    def batch_inference_single_step(
        self,
        tokenized_agents: List[Dict[str, torch.Tensor]],
        map_feature: Dict[str, torch.Tensor],
        prev_feat_a: List[Optional[torch.Tensor]],
        prev_feat_a_t_dict: List[Optional[Dict]],
        is_initial_step: bool = False,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Run parallel single step inference on a batch of states

        Args:
            tokenized_agents: Tokenized agent data
            map_feature: Map feature
            states: List of previous states
            is_initial_step: Whether this is the initial step
        Returns:
            List of dictionaries with next token logits and intermediate features
        """
        n_agent = tokenized_agents[0]["valid_mask"].shape[0]
        n_states = len(tokenized_agents)
        step_current_10hz = self.num_historical_steps - 1  # 10
        step_current_2hz = step_current_10hz // self.shift  # 2
        n_step = tokenized_agents["pos"].shape[1]
        t_now = n_step - 1
        device = tokenized_agents["pos"].device

        # Get current state
        pos_a = tokenized_agents["pos"]  # [batch_size, n_agent, n_step, 2]
        head_a = tokenized_agents["heading"]  # [batch_size, n_agent, n_step] 
        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)

        if prev_feat_a is None or prev_feat_a_t_dict is None:
            (
                feat_a,
                agent_token_emb,
                agent_token_emb_veh,
                agent_token_emb_ped, 
                agent_token_emb_cyc,
                veh_mask, 
                ped_mask, 
                cyc_mask, 
                categorical_embs
            ) = self.batch_agent_token_embedding(
                agent_token_index=tokenized_agents["gt_idx"][:, :n_step],
                trajectory_token_veh=tokenized_agents["trajectory_token_veh"],
                trajectory_token_ped=tokenized_agents["trajectory_token_ped"],
                trajectory_token_cyc=tokenized_agents["trajectory_token_cyc"],
                pos_a=pos_a,
                head_vector_a=head_vector_a,
                agent_type=tokenized_agents["type"],
                agent_shape=tokenized_agents["shape"],
                inference=True,
            )
            feat_a_t_dict = {}
        else:
            feat_a = prev_feat_a
            feat_a_t_dict = prev_feat_a_t_dict
        
        pred_valid = tokenized_agents["valid_mask"].clone()
        
        # Build edges
        if is_initial_step:
            hist_step = step_current_2hz
            batch_s = torch.cat([
                tokenized_agents["batch"] + tokenized_agents["num_graphs"] * t
                for t in range(hist_step)
            ], dim=0)
            batch_pl = torch.cat([
                map_feature["batch"] + tokenized_agents["num_graphs"] * t
                for t in range(hist_step)
            ], dim=0)
            inference_mask = pred_valid[:, :n_step]
            edge_index_t, r_t = self.build_temporal_edge(
                pos_a=pos_a,
                head_a=head_a,
                head_vector_a=head_vector_a,
                mask=pred_valid[:, :n_step],
            )
        else:
            hist_step = 1
            batch_s = tokenized_agents["batch"]
            batch_pl = map_feature["batch"]
            inference_mask = pred_valid[:, :n_step].clone()
            inference_mask[:, :-1] = False
            edge_index_t, r_t = self.build_temporal_edge(
                pos_a=pos_a,
                head_a=head_a,
                head_vector_a=head_vector_a,
                mask=pred_valid[:, :n_step],
                inference_mask=inference_mask,
            )
            edge_index_t[1] = (edge_index_t[1] + 1) // n_step - 1
        
        # Build map2agent edge
        edge_index_pl2a, r_pl2a = self.build_map2agent_edge(
            pos_pl=map_feature["position"],
            orient_pl=map_feature["orientation"],
            pos_a=pos_a[:, -hist_step:],
            head_a=head_a[:, -hist_step:],
            head_vector_a=head_vector_a[:, -hist_step:],
            mask=inference_mask[:, -hist_step:],
            batch_s=batch_s,
            batch_pl=batch_pl,
        )
        
        # Build interaction edge
        edge_index_a2a, r_a2a = self.build_interaction_edge(
            pos_a=pos_a[:, -hist_step:],
            head_a=head_a[:, -hist_step:],
            head_vector_a=head_vector_a[:, -hist_step:],
            batch_s=batch_s,
            mask=inference_mask[:, -hist_step:],
        )
        
        # Process through attention layers
        for i in range(self.num_layers):
            _feat_temporal = feat_a if i == 0 else feat_a_t_dict[i]

            # Process all historical steps together
            _feat_temporal = self.t_attn_layers[i](
                _feat_temporal.flatten(0, 1), r_t, edge_index_t
            ).view(n_agent, n_step, -1)
            _feat_temporal = _feat_temporal.transpose(0, 1).flatten(0, 1)
            
            # Expand map features
            _feat_map = map_feature["pt_token"].unsqueeze(0).repeat(hist_step, 1, 1).flatten(0, 1)
            
            # Process map2agent attention
            _feat_temporal = self.pt2a_attn_layers[i](
                (_feat_map, _feat_temporal), r_pl2a, edge_index_pl2a
            )
            
            # Process agent2agent attention
            _feat_temporal = self.a2a_attn_layers[i](
                _feat_temporal, r_a2a, edge_index_a2a
            )
            
            _feat_temporal = _feat_temporal.view(n_step, n_agent, -1).transpose(0, 1)
            feat_a_now = _feat_temporal[:, -1]  # [n_agent, hidden_dim]
            
            if i + 1 < self.num_layers:
                feat_a_t_dict[i + 1] = _feat_temporal

        # Predict next token
        next_token_logits = self.token_predict_head(feat_a_now)

        return {
            'next_token_logits': next_token_logits,
            'feat_a': feat_a,
            'feat_a_t_dict': feat_a_t_dict,
            't_now': t_now,
            'n_step': n_step
        }
