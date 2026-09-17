import os
import pickle
from typing import Any, Callable, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.nn.functional import smooth_l1_loss, cross_entropy, mse_loss

import lightning.pytorch as pl
from lightning.pytorch.utilities import grad_norm

from src.dp.model.module.encoder_knn import Encoder
from src.dp.model.module.decoder import Decoder
from src.dp.model.module.goal_predictor import GoalPredictor
from src.dp.model.loss.loss import CrossEntropyLoss
from src.dp.utils.normalizer import ActionNormalizer  # StateNormalizer, ObservationNormalizer
from src.dp.utils.lr_schedule import CosineAnnealingWarmUpRestarts
from src.dp.utils.train_utils import (
    transform_coords_to_sdc_frame, 
    transform_coords_to_global_frame, 
    inverse_kinematics, 
    inverse_kinematics_bicycle,
    batch_transform_trajs_to_local_frame,
    batch_transform_trajs_to_global_frame,
    roll_out,
    roll_out_bicycle,
    wrap_angle,
)
from src.dp.model.module.rel_emb import RelationEncoder
from src.smart.metrics import (
    WOSACMetric,
    WOSACMetrics,
    WOSACSubmission
)
from src.utils.wosac_utils import get_scenario_rollouts_vbd, get_scenario_id_int_tensor

class Diffusion_Planner(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = config

        # self.state_normalizer = StateNormalizer.from_json(self.cfg)
        # self.observation_normalizer = ObservationNormalizer.from_json(self.cfg.normalization_file_path)
        self.action_normalizer = ActionNormalizer.from_json(self.cfg)

        self.rel_encoder = RelationEncoder(
            hidden_dim=config.hidden_dim,
            num_freq_bands=64
        )
        self.encoder = Diffusion_Planner_Encoder(self.cfg, self.rel_encoder)
        self.decoder = Diffusion_Planner_Decoder(self.cfg, self.rel_encoder)
        
        # # Goal predictor
        # self._train_predictor = config.get('train_predictor', True)
        # self.predictor = GoalPredictor(self.cfg) if self._train_predictor else None
        self._predicted_neighbor_num = config.get('predicted_neighbor_num', 31)
        
        # Validation settings
        self._future_len = config.get('future_len', 80)
        self._step_len = config.get('step_len', 10)
        self._val_open_loop = config.get('val_open_loop', True)
        self._val_closed_loop = config.get('val_closed_loop', False)
        self._n_rollout_closed_val = config.get('n_rollout_closed_val', 32)
        self.log_epoch = config.get('log_epoch', -1)

        self._action_len = config.get('action_len', 1)
        self._num_actions = self._future_len
        self._action_loss_decay_start_epoch = config.get('action_loss_decay_start_epoch', -1)
        self._action_loss_decay_end_epoch = config.get('action_loss_decay_end_epoch', -1)
        # Global ρ per non-holonomic type (pre-computed from training set).
        # rho_fixed = {1: rho_veh, 3: rho_cyc}. If None, falls back to per-batch median.
        self._rho_fixed = config.get('rho_fixed', None)
        
        # WOSAC metrics
        if config.get('fast_wosac_metric', False):
            self.wosac_metrics = WOSACMetric('2024')
        else:
            self.wosac_metrics = WOSACMetrics("val_closed")
        
        wosac_submission = config.get('wosac_submission', {})
        self.wosac_submission = WOSACSubmission(**wosac_submission)

    @property
    def sde(self):
        return self.decoder.decoder.sde
    
    def forward(self, inputs):

        encoder_outputs = self.encoder(inputs)
        decoder_outputs = self.decoder(encoder_outputs, inputs)

        return encoder_outputs, decoder_outputs
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.learning_rate)
        scheduler = CosineAnnealingWarmUpRestarts(optimizer, self.cfg.train_epochs, self.cfg.warm_up_epoch)
        # return [optimizer], [scheduler]
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
            }
        }
    
    def training_step(self, batch, batch_idx):
        inputs = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                inputs[key] = value.clone()
            else:
                inputs[key] = value

        inputs["agents_history"][..., :6] = transform_coords_to_sdc_frame(inputs["agents_history"][..., :6], inputs['sdc_coord'])
        inputs["agents_future"][..., :6] = transform_coords_to_sdc_frame(inputs["agents_future"][..., :6], inputs['sdc_coord'])
        inputs["lanes"][..., :4] = transform_coords_to_sdc_frame(inputs["lanes"][..., :4], inputs['sdc_coord'])
        inputs["roadlines"][..., :4] = transform_coords_to_sdc_frame(inputs["roadlines"][..., :4], inputs['sdc_coord'])
        inputs["static_maps"][..., :4] = transform_coords_to_sdc_frame(inputs["static_maps"][..., :4], inputs['sdc_coord'])

        # Transform traffic lights to SDC frame: pad xy to [B, TL, 1, 4], transform, keep state untouched
        # tl = inputs["traffic_light_points"]
        # valid_mask = torch.any(tl != 0, dim=-1)
        # tl_xy = tl[..., :2].unsqueeze(2)  # [B, TL, 1, 2]
        # tl_xy_padded = torch.nn.functional.pad(tl_xy, (0, 2))  # [B, TL, 1, 4]: x, y, 0, 0
        # tl_xy_transformed = transform_coords_to_sdc_frame(tl_xy_padded, inputs['sdc_coord'])
        # tl_result = torch.cat([tl_xy_transformed.squeeze(2)[..., :2], tl[..., 2:]], dim=-1)
        # tl_result[~valid_mask] = 0.0
        # inputs["traffic_light_points"] = tl_result

        # Transform lanes_stop_point to SDC frame
        lsp = inputs["lanes_stop_point"]
        lsp_valid = torch.any(lsp != 0, dim=-1)
        lsp_xy = lsp.unsqueeze(2)  # [B, N, 1, 2]
        lsp_xy_padded = torch.nn.functional.pad(lsp_xy, (0, 2))  # [B, N, 1, 4]
        lsp_trans = transform_coords_to_sdc_frame(lsp_xy_padded, inputs['sdc_coord'])
        lsp_trans = lsp_trans.squeeze(2)[..., :2]  # [B, N, 2]
        lsp_trans[~lsp_valid] = 0.0
        inputs["lanes_stop_point"] = lsp_trans

        # inputs['agents_future'][..., :4] = self.state_normalizer(inputs['agents_future'][..., :4])
        # self._log_output(
        #     None, 
        #     inputs, 
        #     batch_idx,
        #     trans2global=False,
        #     sdc_coord=batch['sdc_coord'],
        # )

        loss_dict = {}
        loss, loss_dict, _ = self.diffusion_loss_func(
            inputs=inputs,
            marginal_prob=self.sde.marginal_prob,
            # futures=batch["agents_future"],
            norm=self.action_normalizer,
            loss_dict=loss_dict,
            model_type="x_start",
        )

        for key, value in loss_dict.items():
            self.log(
                key, 
                value,
                on_step=True, on_epoch=False, sync_dist=True,
                prog_bar=True
            )
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step of the model.

        Args:
            batch: Input batch.
            batch_idx: Batch index.
        """
        loss_dict = {}
        if self._val_open_loop:
            inputs = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    inputs[key] = value.clone()
                else:
                    inputs[key] = value

            inputs["agents_history"][..., :6] = transform_coords_to_sdc_frame(inputs["agents_history"][..., :6], inputs['sdc_coord'])
            inputs["agents_future"][..., :6] = transform_coords_to_sdc_frame(inputs["agents_future"][..., :6], inputs['sdc_coord'])
            inputs["lanes"][..., :4] = transform_coords_to_sdc_frame(inputs["lanes"][..., :4], inputs['sdc_coord'])
            inputs["roadlines"][..., :4] = transform_coords_to_sdc_frame(inputs["roadlines"][..., :4], inputs['sdc_coord'])
            inputs["static_maps"][..., :4] = transform_coords_to_sdc_frame(inputs["static_maps"][..., :4], inputs['sdc_coord'])
            # tl = inputs["traffic_light_points"]
            # valid_mask = torch.any(tl != 0, dim=-1)
            # tl_xy = tl[..., :2].unsqueeze(2)  # [B, TL, 1, 2]
            # tl_xy_padded = torch.nn.functional.pad(tl_xy, (0, 2))  # [B, TL, 1, 4]: x, y, 0, 0
            # tl_xy_transformed = transform_coords_to_sdc_frame(tl_xy_padded, inputs['sdc_coord'])
            # tl_result = torch.cat([tl_xy_transformed.squeeze(2)[..., :2], tl[..., 2:]], dim=-1)
            # tl_result[~valid_mask] = 0.0
            # inputs["traffic_light_points"] = tl_result

            # Transform lanes_stop_point to SDC frame
            lsp = inputs["lanes_stop_point"]
            lsp_valid = torch.any(lsp != 0, dim=-1)
            lsp_xy = lsp.unsqueeze(2)
            lsp_xy_padded = torch.nn.functional.pad(lsp_xy, (0, 2))
            lsp_trans = transform_coords_to_sdc_frame(lsp_xy_padded, inputs['sdc_coord'])
            lsp_trans = lsp_trans.squeeze(2)[..., :2]
            lsp_trans[~lsp_valid] = 0.0
            inputs["lanes_stop_point"] = lsp_trans
            
            loss, loss_dict, decoder_output = self.diffusion_loss_func(
                inputs=inputs,
                marginal_prob=self.sde.marginal_prob,
                norm=self.action_normalizer,
                loss_dict=loss_dict,
                model_type="x_start",
            )
            
            self.log_dict(
                loss_dict, 
                on_step=False, on_epoch=True, sync_dist=True,
                prog_bar=True
            )

            # Record decoder output and map elements
            # self._log_output(
            #     decoder_output['predicted_trajectories'], 
            #     batch, 
            #     batch_idx,
            #     trans2global=True,
            #     sdc_coord=batch['sdc_coord'],
            # )

        if self._val_closed_loop:
            step_len = self._step_len
            future_len = self._future_len
            batch_size = batch['agents_history'].shape[0]
            
            inputs = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    inputs[key] = value.clone()
                else:
                    inputs[key] = value

            # Transform to SDC frame
            inputs["agents_history"][..., :6] = transform_coords_to_sdc_frame(inputs["agents_history"][..., :6], inputs['sdc_coord'])
            inputs["agents_future"][..., :6] = transform_coords_to_sdc_frame(inputs["agents_future"][..., :6], inputs['sdc_coord'])
            inputs["lanes"][..., :4] = transform_coords_to_sdc_frame(inputs["lanes"][..., :4], inputs['sdc_coord'])
            inputs["roadlines"][..., :4] = transform_coords_to_sdc_frame(inputs["roadlines"][..., :4], inputs['sdc_coord'])
            inputs["static_maps"][..., :4] = transform_coords_to_sdc_frame(inputs["static_maps"][..., :4], inputs['sdc_coord'])
            # tl = inputs["traffic_light_points"]
            # valid_mask = torch.any(tl != 0, dim=-1)
            # tl_xy = tl[..., :2].unsqueeze(2)
            # tl_xy_padded = torch.nn.functional.pad(tl_xy, (0, 2))
            # tl_xy_transformed = transform_coords_to_sdc_frame(tl_xy_padded, inputs['sdc_coord'])
            # tl_result = torch.cat([tl_xy_transformed.squeeze(2)[..., :2], tl[..., 2:]], dim=-1)
            # tl_result[~valid_mask] = 0.0
            # inputs["traffic_light_points"] = tl_result

            # Transform lanes_stop_point to SDC frame
            lsp = inputs["lanes_stop_point"]
            lsp_valid = torch.any(lsp != 0, dim=-1)
            lsp_xy = lsp.unsqueeze(2)
            lsp_xy_padded = torch.nn.functional.pad(lsp_xy, (0, 2))
            lsp_trans = transform_coords_to_sdc_frame(lsp_xy_padded, inputs['sdc_coord'])
            lsp_trans = lsp_trans.squeeze(2)[..., :2]
            lsp_trans[~lsp_valid] = 0.0
            inputs["lanes_stop_point"] = lsp_trans
            
            pred_traj = []
            # Compute r from agent type and global ρ (or default 0.25 if not configured)
            agents_type_cl = inputs["agents_type"]  # [B, P]
            length = batch['agents_history'][:, :self._predicted_neighbor_num+1, -1, 6]  # [B, P]
            is_veh_cl = agents_type_cl[:, :self._predicted_neighbor_num+1] == 1
            is_cyc_cl = agents_type_cl[:, :self._predicted_neighbor_num+1] == 3
            rho_veh = self._rho_fixed.get(1, 0.25) if self._rho_fixed else 0.25
            rho_cyc = self._rho_fixed.get(3, 0.25) if self._rho_fixed else 0.25
            r_cl = torch.where(is_veh_cl, rho_veh * length,
                              torch.where(is_cyc_cl, rho_cyc * length, torch.zeros_like(length)))  # [B, P]
            for r_idx in range(self._n_rollout_closed_val):
                print('closed-loop rollout', r_idx)
                trajs = []
                inputs_ = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                
                # Track SDC position for each step (in global frame)
                sdc_coord_global = batch['sdc_coord'].clone()  # [B, 3]
                prev_pred_global = batch['agents_history'][..., :self._predicted_neighbor_num+1, :, :6].clone()  # [B, P, T_hist, 6] in global frame
                
                agents_mask = batch['agents_interested'][:, :self._predicted_neighbor_num+1] > 0  # [B, P], 0=padded, >0=valid
                
                for t in range((future_len + step_len - 1) // step_len):
                    # Convert inputs to SDC frame for model input
                    if t == 0:
                        # First step: inputs are already in initial SDC frame
                        inputs_sdc = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs_.items()}
                        current_states_global = batch['agents_history'][..., :self._predicted_neighbor_num+1, -1, :6].clone()  # [B, P, 6]
                    else:
                        # Convert all map elements to current SDC frame from global
                        inputs_sdc = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs_.items()}
                        inputs_sdc['agents_history'][..., :self._predicted_neighbor_num+1, :, :6] = transform_coords_to_sdc_frame(
                            prev_pred_global,
                            sdc_coord_global
                        )
                        inputs_sdc['lanes'][..., :4] = transform_coords_to_sdc_frame(
                            batch['lanes'][..., :4],
                            sdc_coord_global
                        )
                        inputs_sdc['roadlines'][..., :4] = transform_coords_to_sdc_frame(
                            batch['roadlines'][..., :4],
                            sdc_coord_global
                        )
                        inputs_sdc['static_maps'][..., :4] = transform_coords_to_sdc_frame(
                            batch['static_maps'][..., :4],
                            sdc_coord_global
                        )
                        inputs_sdc['agents_future'][..., :6] = transform_coords_to_sdc_frame(
                            batch['agents_future'][..., :6],
                            sdc_coord_global
                        )
                        # tl = inputs["traffic_light_points"]
                        # valid_mask = torch.any(tl != 0, dim=-1)
                        # tl_xy = tl[..., :2].unsqueeze(2)
                        # tl_xy_padded = torch.nn.functional.pad(tl_xy, (0, 2))
                        # tl_xy_transformed = transform_coords_to_sdc_frame(tl_xy_padded, inputs['sdc_coord'])
                        # tl_result = torch.cat([tl_xy_transformed.squeeze(2)[..., :2], tl[..., 2:]], dim=-1)
                        # tl_result[~valid_mask] = 0.0
                        # inputs["traffic_light_points"] = tl_result

                        # Transform lanes_stop_point to SDC frame
                        lsp = batch["lanes_stop_point"]
                        lsp_valid = torch.any(lsp != 0, dim=-1)
                        lsp_xy = lsp.unsqueeze(2)
                        lsp_xy_padded = torch.nn.functional.pad(lsp_xy, (0, 2))
                        lsp_trans = transform_coords_to_sdc_frame(lsp_xy_padded, sdc_coord_global)
                        lsp_trans = lsp_trans.squeeze(2)[..., :2]
                        lsp_trans[~lsp_valid] = 0.0
                        inputs_sdc["lanes_stop_point"] = lsp_trans
                        
                        current_states_global = prev_pred_global[:, :self._predicted_neighbor_num+1, -1, :6].clone()  # [B, P, 6]

                    # inputs_normalized = self.observation_normalizer(inputs_sdc)
                    inputs_normalized = inputs_sdc

                    _, decoder_output = self.forward(inputs_normalized)
                    pred_actions = decoder_output["prediction"]  # [B, P, num_actions, 2]
                    
                    pred_actions = self.action_normalizer.inverse(pred_actions)
                    pred_trajs_global = roll_out_bicycle(
                        current_states_global,  # [B, P, 6]
                        pred_actions[:, :self._predicted_neighbor_num+1],
                        r=r_cl[:, :self._predicted_neighbor_num+1],
                        agent_type=agents_type_cl[:, :self._predicted_neighbor_num+1],
                        dt=0.1,
                        valid_mask=agents_mask,
                    )  # [B, P, S+1, 6]

                    if t == 0:
                        pred_trajs_local = roll_out_bicycle(
                            inputs_sdc['agents_history'][:, :, -1, :6],  # [B, P, 6]
                            pred_actions[:, :self._predicted_neighbor_num+1],
                            r=r_cl[:, :self._predicted_neighbor_num+1],
                            agent_type=agents_type_cl[:, :self._predicted_neighbor_num+1],
                            dt=0.1,
                            valid_mask=agents_mask,
                        )
                        pred_trajs_local_4d = torch.cat([
                            pred_trajs_local[..., :2],
                            torch.zeros_like(pred_trajs_local[..., 0:1]),
                            torch.atan2(pred_trajs_local[..., 3:4], pred_trajs_local[..., 2:3])
                        ], dim=-1)
                        self._log_output(
                            pred_trajs_local_4d,
                            inputs_sdc,
                            batch_idx,
                            trans2global=False,
                            sdc_coord=batch['sdc_coord'],
                        )

                    pred_trajs_step_global = pred_trajs_global[:, :, :step_len, :]  # [B, P, step_len, 6]
                    trajs.append(pred_trajs_step_global[..., :4])
                    
                    ego_pred_global = pred_trajs_step_global[:, 0, step_len-1, :]  # [B, 6]
                    sdc_coord_global[:, 0] = ego_pred_global[:, 0]  # x
                    sdc_coord_global[:, 1] = ego_pred_global[:, 1]  # y
                    sdc_coord_global[:, 2] = torch.atan2(ego_pred_global[:, 3], ego_pred_global[:, 2])  # theta from cos/sin
                    
                    # Update history in global frame for next step
                    hist_len = batch['agents_history'].shape[2]
                    prev_pred_global = torch.cat(
                        [
                            prev_pred_global[:, :, -(hist_len-step_len):, :], pred_trajs_step_global
                        ]
                        , dim=2
                    )  # [B, P, T_hist, 6]
                
                full_trajs = torch.cat(trajs, dim=-2)  # [B, P, future_len, 4] - in global frame
                pred_traj.append(full_trajs)
            
            pred_traj = torch.stack(pred_traj, dim=1)  # [B, n_rollout, P, future_len, 4] - [x, y, cos, sin]
            pred_z = batch['agents_history'][:, :self._predicted_neighbor_num+1, -1, 8]
            pred_z = pred_z.unsqueeze(-1).repeat(1, 1, future_len)
            pred_z = pred_z.unsqueeze(1).repeat(1, self._n_rollout_closed_val, 1, 1)
            pred_head = torch.atan2(pred_traj[..., 3], pred_traj[..., 2])

            simulated_states = torch.cat(
                [pred_traj[..., :2], pred_z[..., None], pred_head[..., None]], dim=-1
            )  # [B, n_rollout, P, future_len, 4] - [x, y, z, heading]

            # Handle remaining agents (linear extrapolation)
            all_agents_id_list = [x for x in batch["agents_id"]]
            simulated_states_list = [x for x in simulated_states]
            
            for i in range(batch_size):
                if 'agents_history_remaining' in batch and batch['agents_history_remaining'][i].shape[0] > 0:
                    num_agents_remaining = batch['agents_history_remaining'][i].shape[0]
                    cur_data = batch['agents_history_remaining'][i].clone()  # [A, T, 9]
                    
                    if 'agents_id_remaining' in batch:
                        all_agents_id_list[i] = torch.cat([all_agents_id_list[i], batch['agents_id_remaining'][i]], dim=0)
                    
                    simulated_states_remaining = torch.zeros(
                        (self._n_rollout_closed_val, num_agents_remaining, future_len, 4), 
                        device=self.device
                    )
                    
                    states_current = cur_data[:, -1, :]  # [A, 9]
                    pos = states_current[:, :2]  # [A, 2]
                    heading_cos = states_current[:, 2]  # [A]
                    heading_sin = states_current[:, 3]  # [A]
                    heading = torch.atan2(heading_sin, heading_cos)  # [A]
                    z = states_current[:, 8]  # [A]
                    vel = states_current[:, 4:6]  # [A, 2]
                    
                    time_steps = torch.arange(future_len, device=self.device).float() * 0.1  # [future_len]
                    
                    pos_expanded = pos.unsqueeze(1).unsqueeze(0)  # [1, A, 1, 2]
                    vel_expanded = vel.unsqueeze(1).unsqueeze(0)  # [1, A, 1, 2]
                    time_expanded = time_steps.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # [1, 1, future_len, 1]
                    
                    pred_pos = pos_expanded + vel_expanded * time_expanded  # [1, A, future_len, 2]
                    
                    heading_expanded = heading.unsqueeze(0).unsqueeze(-1).expand(
                        self._n_rollout_closed_val, num_agents_remaining, future_len
                    )
                    z_expanded = z.unsqueeze(0).unsqueeze(-1).expand(
                        self._n_rollout_closed_val, num_agents_remaining, future_len
                    )
                    
                    simulated_states_remaining[..., :2] = pred_pos.expand(
                        self._n_rollout_closed_val, num_agents_remaining, future_len, 2
                    )
                    simulated_states_remaining[..., 2] = z_expanded
                    simulated_states_remaining[..., 3] = heading_expanded
                    
                    simulated_states_list[i] = torch.cat(
                        [simulated_states_list[i], simulated_states_remaining], dim=1
                    )

            self._log_output(
                [s.cpu().detach().numpy() for s in simulated_states_list], 
                batch, 
                batch_idx, 
                log_dir='/home/zhanghailiang/Repos/catk/logs/debug',
                trans2global=False,
            )
            
            # Update WOSAC metrics
            if isinstance(self.wosac_metrics, WOSACMetric):
                self.wosac_metrics.update(
                    scenario_id=batch["scenario_id"],
                    gt_scenarios=batch["gt_scenario"],
                    agent_id=all_agents_id_list,
                    simulated_states=simulated_states_list,
                )
            else:
                scenario_rollouts = get_scenario_rollouts_vbd(
                    scenario_id=get_scenario_id_int_tensor(
                        batch["scenario_id"], self.device
                    ),
                    agent_id=all_agents_id_list,
                    simulated_states=simulated_states_list,
                )
                self.wosac_metrics.update(batch["tfrecord_path"], scenario_rollouts)
        
        return loss_dict
    
    def on_validation_epoch_end(self):
        if self._val_closed_loop:
            epoch_wosac_metrics = self.wosac_metrics.compute()
            if self.global_rank == 0:
                epoch_wosac_metrics["epoch"] = (
                    self.log_epoch if self.log_epoch >= 0 else self.current_epoch
                )
                self.logger.log_metrics(epoch_wosac_metrics)

            self.wosac_metrics.reset()

            if self.global_rank == 0:
                if self.wosac_submission.is_active:
                    self.wosac_submission.save_sub_file()

    def on_after_backward(self):
        total_norm = 0.0
        has_nan_grad = False

        # 按模块分组统计梯度范数
        module_grad_norms = {}  # {module_name: squared_norm_sum}
        for name, p in self.named_parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any():
                    print(f"[NaN GRAD] step {self.global_step}: {name}, nan_count={torch.isnan(p.grad).sum()}")
                    has_nan_grad = True
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2

                # 提取模块前缀: 取前两级路径 (e.g., "decoder.dit.chunk_blocks.0")
                parts = name.split('.')
                if len(parts) >= 3:
                    module_key = '.'.join(parts[:3])
                    # 对于 decoder.dit.xxx 的模块，进一步细化到 block 级别
                    if parts[0] == 'decoder' and parts[1] == 'dit' and parts[2] in ('chunk_blocks', 'refiner_blocks'):
                        if len(parts) >= 4:
                            module_key = '.'.join(parts[:4])
                else:
                    module_key = '.'.join(parts[:2]) if len(parts) >= 2 else parts[0]

                module_grad_norms[module_key] = module_grad_norms.get(module_key, 0.0) + param_norm.item() ** 2

        total_norm = total_norm ** 0.5
        if has_nan_grad:
            print(f"[NaN GRAD] step {self.global_step}: gradient_norm={total_norm}")

        self.log("grad/global_norm_raw", total_norm, on_step=True, on_epoch=True, prog_bar=False)

        # 按 epoch 累积各模块的梯度范数平方和（只记 epoch 级，避免日志量过大）
        if not hasattr(self, '_epoch_module_grad_norms'):
            self._epoch_module_grad_norms = {}

        for mod_name, sq_norm in module_grad_norms.items():
            self._epoch_module_grad_norms[mod_name] = \
                self._epoch_module_grad_norms.get(mod_name, 0.0) + sq_norm

        if not hasattr(self, '_epoch_grad_count'):
            self._epoch_grad_count = 0
        self._epoch_grad_count += 1

        self._pre_clip_norm = total_norm
    
    def on_before_optimizer_step(self, optimizer):
            # Check for NaN in parameters before optimizer step
            has_nan_weight = False
            for name, p in self.named_parameters():
                if torch.isnan(p).any():
                    print(f"[NaN WEIGHT] step {self.global_step}: {name}, nan_count={torch.isnan(p).sum()}")
                    has_nan_weight = True
            if has_nan_weight:
                print(f"[NaN WEIGHT] step {self.global_step}: optimizer step skipped due to NaN weights")

            norms = grad_norm(self, norm_type=2)
            global_norm = norms.get("grad_2.0_norm_total", 0.0)
            
            clip_threshold = 1.0
            is_clipped = 1.0 if global_norm >= (clip_threshold - 1e-4) else 0.0
            
            self.log("grad/clipping_ratio", is_clipped, on_step=False, on_epoch=True)
            self.log("grad/global_norm", global_norm, on_step=True, on_epoch=True, prog_bar=False)

            if hasattr(self, '_pre_clip_norm') and self._pre_clip_norm > 0:
                clip_factor = global_norm / self._pre_clip_norm
                self.log("grad/clip_factor", clip_factor, on_step=True, on_epoch=True, prog_bar=False)

    def on_train_epoch_end(self):
        if not hasattr(self, '_epoch_module_grad_norms') or self._epoch_grad_count == 0:
            return

        count = self._epoch_grad_count
        module_avg_norms = {k: (v / count) ** 0.5 for k, v in self._epoch_module_grad_norms.items()}

        top_n = 8
        sorted_modules = sorted(module_avg_norms.items(), key=lambda x: x[1], reverse=True)
        for rank, (mod_name, avg_norm) in enumerate(sorted_modules[:top_n]):
            short_name = mod_name.replace('decoder.dit.', 'd.').replace('encoder.', 'e.')
            self.log(f"grad/mod/{short_name}", avg_norm, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        # enc_norm = sum(
        #     v for k, v in self._epoch_module_grad_norms.items() if k.startswith('encoder')
        # ) ** 0.5 / (count ** 0.5)
        # dec_norm = sum(
        #     v for k, v in self._epoch_module_grad_norms.items() if k.startswith('decoder')
        # ) ** 0.5 / (count ** 0.5)
        # total = enc_norm + dec_norm
        # if total > 0:
        #     self.log("grad/enc_ratio", enc_norm / total, on_step=False, on_epoch=True, prog_bar=False)
        #     self.log("grad/dec_ratio", dec_norm / total, on_step=False, on_epoch=True, prog_bar=False)

        self._epoch_module_grad_norms.clear()
        self._epoch_grad_count = 0
    
    def diffusion_loss_func(
        self,
        inputs: Dict[str, torch.Tensor],
        marginal_prob: Callable[[torch.Tensor], torch.Tensor],

        # futures: Tuple[torch.Tensor, torch.Tensor],
        
        norm: ActionNormalizer,
        loss_dict: Dict[str, Any],

        model_type: str,
        eps: float = 1e-3,
    ):
        # inputs_norm = self.observation_normalizer(inputs)
        inputs_norm = inputs

        agents_future = inputs["agents_future"]
        agents_future_norm = inputs_norm["agents_future"]  # [B, P, T+1, 9]
        agents_future_valid = inputs_norm["agents_future_valid"]  # [B, P, T+1]
        agents_interested = inputs_norm["agents_interested"]  # [B, P]
        agents_type = inputs["agents_type"]  # [B, P], WOMD: 1=VEH, 2=PED, 3=CYC
        B, P, T, _ = agents_future.shape
        
        current_states = inputs["agents_history"][:, :, -1:, :6].clone()  # [B, P, 1, 6]

        # Get ground truth future states [B, P, T, 3] - x, y, yaw (from cos and sin)
        gt_future_pos = agents_future[..., 1:, :2].clone()  # [B, P, T, 2] - x, y
        gt_future_yaw = torch.atan2(agents_future[..., 1:, 3], agents_future[..., 1:, 2])  # [B, P, T]
        # future_mask = agents_future_valid[..., 1:].clone()  # [B, P, T]
        future_mask = agents_future_valid[..., 1:]*(agents_interested[..., None]>0)  # [B, P, T]

        # Calculate GT actions using bicycle inverse kinematics
        gt_actions, gt_actions_valid, r = inverse_kinematics_bicycle(
            agents_future,
            agents_future_valid,
            agent_type=agents_type,
            dt=0.1,
            rho_fixed=self._rho_fixed,
        )  # gt_actions: [B, P, T-1, 3], gt_actions_valid: [B, P, T-1], r: [B, P]

        if self.training:
            t = torch.rand(B, device=agents_future.device) * (1 - eps) + eps # [B,]
            z = torch.randn_like(gt_actions, device=agents_future.device) # [B, P, T-1, 3]

            # # for debug
            # t = torch.full_like(t, eps)
            # z = torch.zeros_like(z)

            gt_actions_norm = norm(gt_actions)
            
            mean, std = marginal_prob(gt_actions_norm, t)
            std = std.view(-1, *([1] * (len(gt_actions_norm.shape)-1)))

            xT = mean + std * z
            
            merged_inputs = {
                **inputs_norm,
                # **inputs,
                "sampled_actions": xT,
                "diffusion_time": t,
                # "current_states": current_states,  # Pass current states for potential use
            }

            encoder_outputs, decoder_output = self.forward(merged_inputs)
            pred_actions = decoder_output["score"]  # [B, P, T-1, 3]
            pred_actions_norm = decoder_output["score"].clone()

            pred_actions = norm.inverse(pred_actions)
            pred_trajectories = roll_out_bicycle(
                current_states.squeeze(2),  # [B, P, 6]
                pred_actions,
                r=r,
                agent_type=agents_type,
                dt=0.1,
            )  # [B, P, S+1, 6]
            
            decoder_output["predicted_trajectories"] = pred_trajectories

            state_loss = mse_loss(pred_trajectories[..., :2], gt_future_pos, reduction='none').sum(-1)  # [B, P, T]
            
            pred_yaw = torch.atan2(pred_trajectories[..., 3], pred_trajectories[..., 2])  # [B, P, T]
            yaw_error = pred_yaw - gt_future_yaw  # [B, P, T]
            yaw_error = torch.atan2(torch.sin(yaw_error), torch.cos(yaw_error))
            yaw_loss = yaw_error ** 2  # [B, P, T], MSE
            
            # weight = 1.0 / (std.view(-1) ** 2 + 1e-4)
            # weight = weight / weight.mean()
            # action_loss = action_loss * weight.view(B, 1, 1)

            a_par_loss = smooth_l1_loss(pred_actions_norm[..., 0], gt_actions_norm[..., 0], reduction='none') * gt_actions_valid
            a_lat_loss = smooth_l1_loss(pred_actions_norm[..., 1], gt_actions_norm[..., 1], reduction='none') * gt_actions_valid
            a_psi_loss = smooth_l1_loss(pred_actions_norm[..., 2], gt_actions_norm[..., 2], reduction='none') * gt_actions_valid
            valid_count = gt_actions_valid.sum()

            loss_dict["train/a_par_loss"] = a_par_loss.sum() / valid_count
            loss_dict["train/a_lat_loss"] = a_lat_loss.sum() / valid_count
            loss_dict["train/a_psi_loss"] = a_psi_loss.sum() / valid_count
            # loss_dict["train/action_loss"] = (a_par_loss.sum() + a_lat_loss.sum() + a_psi_loss.sum()) / valid_count

            masked_loss = (state_loss + yaw_loss) * future_mask[:, :self._predicted_neighbor_num+1, :]

            action_loss_weight = 1.0
            if self._action_loss_decay_start_epoch >= 0 and self._action_loss_decay_end_epoch > self._action_loss_decay_start_epoch:
                if self.current_epoch >= self._action_loss_decay_start_epoch:
                    progress = (self.current_epoch - self._action_loss_decay_start_epoch) / (self._action_loss_decay_end_epoch - self._action_loss_decay_start_epoch)
                    action_loss_weight = max(0.0, 1.0 - progress)

            # loss = 0.1 * masked_loss.sum() / future_mask[:, :self._predicted_neighbor_num+1, :].sum() + action_loss_weight * (accel_loss.sum() + yaw_rate_loss.sum()) / valid_count
            loss = (a_par_loss.sum() + a_lat_loss.sum() + a_psi_loss.sum()) / valid_count

            # Calculate ADE and FDE metrics
            denoise_mse = torch.norm(pred_trajectories[..., :2] - gt_future_pos, dim=-1)
            denoise_ADE = denoise_mse[future_mask.bool()].mean()
            denoise_FDE = denoise_mse[..., -1][future_mask[..., -1].bool()].mean()
            loss_dict["train/denoise_ADE"] = denoise_ADE
            loss_dict["train/denoise_FDE"] = denoise_FDE

            # Diffusion loss: directly supervise x_0 prediction in action space
            # diffusion_loss = mse_loss(pred_actions_norm, gt_actions_norm, reduction='none').sum(-1)  # [B, P, num_actions]
            # diffusion_loss = (diffusion_loss * gt_actions_valid).sum() / gt_actions_valid.sum()
            # loss_dict["train/diffusion_loss"] = diffusion_loss.item()
        else:
            _, decoder_output = self.forward(inputs_norm)
            # _, decoder_output = self.forward(inputs)
            pred_actions = decoder_output["prediction"]  # [B, P, T-1, 3]
            
            pred_actions = norm.inverse(pred_actions)
            # No mask since mask is applied in loss calculation
            pred_trajectories = roll_out_bicycle(
                current_states[:, :self._predicted_neighbor_num+1, -1, :],  # [B, P, 6]
                pred_actions[:, :self._predicted_neighbor_num+1],
                r=r[:, :self._predicted_neighbor_num+1],
                agent_type=agents_type[:, :self._predicted_neighbor_num+1],
                dt=0.1,
            )  # [B, P, S+1, 6]
            
            decoder_output["predicted_trajectories"] = pred_trajectories
            
            pred_traj_len = pred_trajectories.shape[-2]
            gt_future_pos_trunc = gt_future_pos[:, :self._predicted_neighbor_num+1, :pred_traj_len, :2]
            state_loss = smooth_l1_loss(pred_trajectories[..., :2], gt_future_pos_trunc, reduction='none').sum(-1)
            
            pred_yaw = torch.atan2(pred_trajectories[..., 3], pred_trajectories[..., 2])
            gt_future_yaw_trunc = gt_future_yaw[:, :self._predicted_neighbor_num+1, :pred_traj_len]
            yaw_error = pred_yaw - gt_future_yaw_trunc
            yaw_error = torch.atan2(torch.sin(yaw_error), torch.cos(yaw_error))
            yaw_loss = yaw_error ** 2
            
            # loss_dict["val/state_loss"] = (state_loss * future_mask[:, :self._predicted_neighbor_num+1, :]).sum() / future_mask[:, :self._predicted_neighbor_num+1, :].sum()
            # loss_dict["val/yaw_loss"] = (yaw_loss * future_mask[:, :self._predicted_neighbor_num+1, :]).sum() / future_mask[:, :self._predicted_neighbor_num+1, :].sum()
            
            gt_actions_trunc = gt_actions[:, :self._predicted_neighbor_num+1, :pred_traj_len, :]
            gt_valid_trunc = gt_actions_valid[:, :self._predicted_neighbor_num+1, :pred_traj_len]
            action_loss = mse_loss(pred_actions[:, :self._predicted_neighbor_num+1], gt_actions_trunc, reduction='none').sum(-1)
            action_loss = action_loss * gt_valid_trunc
            loss_dict["val/action_loss"] = action_loss.sum() / gt_valid_trunc.sum()

            # Calculate ADE and FDE metrics
            denoise_mse = torch.norm(pred_trajectories[..., :2] - gt_future_pos_trunc, dim=-1)
            future_mask_trunc = future_mask[:, :self._predicted_neighbor_num+1, :pred_traj_len]
            denoise_ADE = denoise_mse[future_mask_trunc.bool()].mean()
            denoise_FDE = denoise_mse[..., -1][future_mask_trunc[..., -1].bool()].mean()
            loss_dict["val/denoise_ADE"] = denoise_ADE
            loss_dict["val/denoise_FDE"] = denoise_FDE
            
            masked_loss = (state_loss + yaw_loss) * future_mask_trunc
            loss = masked_loss.sum() / future_mask_trunc.sum()
            # loss = action_loss.sum() / gt_actions_valid.sum()
        
        # neighbor_mask = future_mask[:, 1:self._predicted_neighbor_num+1, :]
        # neighbor_loss = loss[:, 1:self._predicted_neighbor_num+1, :]
        # masked_prediction_loss = neighbor_loss[neighbor_mask]

        # if masked_prediction_loss.numel() > 0:
        #     loss_dict["neighbor_prediction_loss"] = masked_prediction_loss.mean()
        # else:
        #     loss_dict["neighbor_prediction_loss"] = torch.tensor(0.0, device=masked_prediction_loss.device)

        # loss_dict["ego_planning_loss"] = loss[:, 0, :][future_mask[:, 0, :]].mean()

        if self.training:
            assert not torch.isnan(loss).sum(), f"loss cannot be nan"

        return loss, loss_dict, decoder_output
    
    def _log_output(
            self, 
            pred, 
            batch, 
            batch_idx,
            log_dir = '/home/zhanghailiang/Repos/catk/logs',
            trans2global = False,
            sdc_coord = None
        ):
        """
        Args:
            pred: predicted trajectories
            batch: data
            batch_idx: data index
        """
        if isinstance(pred, torch.Tensor):
            if trans2global:
                if sdc_coord is None:
                    sdc_coord = batch['sdc_coord']
                pred_global = transform_coords_to_global_frame(pred, sdc_coord)
            else:
                pred_global = pred
            log_data = {
                'decoder_prediction': pred_global.cpu().numpy(),
                'agents_history': batch['agents_history'].cpu().detach().numpy(),
                'agents_future': batch['agents_future'].cpu().detach().numpy(),
                'lanes': batch['lanes'].cpu().detach().numpy(),
                'lanes_stop_point': batch['lanes_stop_point'].cpu().detach().numpy(),
                'roadlines': batch['roadlines'].cpu().detach().numpy(),
                'static_maps': batch['static_maps'].cpu().detach().numpy(),
                'sdc_coord': batch['sdc_coord'].cpu().detach().numpy() if hasattr(batch['sdc_coord'], 'cpu') else batch['sdc_coord'],
                'agents_future_valid': batch['agents_future_valid'].cpu().detach().numpy(),
                'agents_interested': batch['agents_interested'].cpu().detach().numpy(),
            }
        elif isinstance(pred, list):
            log_data = {
                'decoder_prediction': pred,
                'agents_history': batch['agents_history'].cpu().detach().numpy(),
                'agents_future': batch['agents_future'].cpu().detach().numpy(),
                'lanes': batch['lanes'].cpu().detach().numpy(),
                'lanes_stop_point': batch['lanes_stop_point'].cpu().detach().numpy(),
                'roadlines': batch['roadlines'].cpu().detach().numpy(),
                'static_maps': batch['static_maps'].cpu().detach().numpy(),
                'sdc_coord': batch['sdc_coord'].cpu().detach().numpy() if hasattr(batch['sdc_coord'], 'cpu') else batch['sdc_coord'],
                'agents_future_valid': batch['agents_future_valid'].cpu().detach().numpy(),
                'agents_interested': batch['agents_interested'].cpu().detach().numpy(),
            }
        elif pred is None:
            log_data = {
                'decoder_prediction': pred,
                'agents_history': batch['agents_history'].cpu().detach().numpy(),
                'agents_future': batch['agents_future'].cpu().detach().numpy(),
                'lanes': batch['lanes'].cpu().detach().numpy(),
                'lanes_stop_point': batch['lanes_stop_point'].cpu().detach().numpy(),
                'roadlines': batch['roadlines'].cpu().detach().numpy(),
                'static_maps': batch['static_maps'].cpu().detach().numpy(),
                'sdc_coord': batch['sdc_coord'].cpu().detach().numpy() if hasattr(batch['sdc_coord'], 'cpu') else batch['sdc_coord'],
                'agents_future_valid': batch['agents_future_valid'].cpu().detach().numpy(),
                'agents_interested': batch['agents_interested'].cpu().detach().numpy(),
            }
        
        os.makedirs(log_dir, exist_ok=True)
        
        if batch_idx == 0: 
            log_file_path = os.path.join(log_dir, f'output_log_{batch_idx}.pkl')
            with open(log_file_path, 'wb') as f:
                pickle.dump(log_data, f)
    
class Diffusion_Planner_Encoder(nn.Module):
    def __init__(self, config, rel_encoder=None):
        super().__init__()

        self.encoder = Encoder(config, rel_encoder=rel_encoder)

    def forward(self, inputs):

        encoder_outputs = self.encoder(inputs)

        return encoder_outputs
    

class Diffusion_Planner_Decoder(nn.Module):
    def __init__(self, config, rel_encoder=None):
        super().__init__()

        self.decoder = Decoder(config, rel_encoder=rel_encoder)
        # self.decoder = QCDecoder(config, rel_encoder=rel_encoder)

    def forward(self, encoder_outputs, inputs):

        decoder_outputs = self.decoder(encoder_outputs, inputs)
        
        return decoder_outputs