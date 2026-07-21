import os
import pickle
from typing import Any, Callable, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.nn.functional import smooth_l1_loss, cross_entropy, mse_loss

import lightning.pytorch as pl
from lightning.pytorch.utilities import grad_norm

from src.dp.model.module.encoder import Encoder
from src.dp.model.module.decoder import Decoder
from src.dp.model.module.denoise_qc_decoder import QCDecoder
# from src.dp.model.module.denoise_decoder import DenoiseDecoder
from src.dp.model.module.goal_predictor import GoalPredictor
from src.dp.model.loss.loss import CrossEntropyLoss
from src.dp.utils.normalizer import StateNormalizer, ObservationNormalizer, ActionNormalizer
from src.dp.utils.lr_schedule import CosineAnnealingWarmUpRestarts
from src.dp.utils.train_utils import (
    transform_coords_to_sdc_frame, 
    transform_coords_to_global_frame, 
    inverse_kinematics, 
    batch_transform_trajs_to_local_frame,
    batch_transform_trajs_to_global_frame,
    roll_out
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

        self.state_normalizer = StateNormalizer.from_json(self.cfg)
        self.observation_normalizer = ObservationNormalizer.from_json(self.cfg.normalization_file_path)
        self.action_normalizer = ActionNormalizer.from_json(self.cfg)

        self.rel_encoder = RelationEncoder(
            hidden_dim=config.hidden_dim,
            num_freq_bands=64
        )
        self.encoder = Diffusion_Planner_Encoder(self.cfg, self.rel_encoder)
        self.decoder = Diffusion_Planner_Decoder(self.cfg, self.rel_encoder)
        
        # Goal predictor
        self._train_predictor = config.get('train_predictor', True)
        self.predictor = GoalPredictor(self.cfg) if self._train_predictor else None
        self._predicted_neighbor_num = config.get('predicted_neighbor_num', 31)
        
        # Validation settings
        self._future_len = config.get('future_len', 80)
        self._step_len = config.get('step_len', 10)
        self._val_open_loop = config.get('val_open_loop', True)
        self._val_closed_loop = config.get('val_closed_loop', False)
        self._n_rollout_closed_val = config.get('n_rollout_closed_val', 32)
        self.log_epoch = config.get('log_epoch', -1)

        self._action_len = config.get('action_len', 2)
        self._num_actions = self._future_len // self._action_len
        self._pi_loss_weight = config.get('pi_loss_weight', 1.0)
        self.pi_loss = CrossEntropyLoss(reduction='mean')
        
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
        tl = inputs["traffic_light_points"]
        valid_mask = torch.any(tl != 0, dim=-1)
        tl_xy = tl[..., :2].unsqueeze(2)  # [B, TL, 1, 2]
        tl_xy_padded = torch.nn.functional.pad(tl_xy, (0, 2))  # [B, TL, 1, 4]: x, y, 0, 0
        tl_xy_transformed = transform_coords_to_sdc_frame(tl_xy_padded, inputs['sdc_coord'])
        tl_result = torch.cat([tl_xy_transformed.squeeze(2)[..., :2], tl[..., 2:]], dim=-1)
        tl_result[~valid_mask] = 0.0
        inputs["traffic_light_points"] = tl_result

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
                on_step=False, on_epoch=True, sync_dist=True,
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
            tl = inputs["traffic_light_points"]
            valid_mask = torch.any(tl != 0, dim=-1)
            tl_xy = tl[..., :2].unsqueeze(2)  # [B, TL, 1, 2]
            tl_xy_padded = torch.nn.functional.pad(tl_xy, (0, 2))  # [B, TL, 1, 4]: x, y, 0, 0
            tl_xy_transformed = transform_coords_to_sdc_frame(tl_xy_padded, inputs['sdc_coord'])
            tl_result = torch.cat([tl_xy_transformed.squeeze(2)[..., :2], tl[..., 2:]], dim=-1)
            tl_result[~valid_mask] = 0.0
            inputs["traffic_light_points"] = tl_result
            
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
            tl = inputs["traffic_light_points"]
            valid_mask = torch.any(tl != 0, dim=-1)
            tl_xy = tl[..., :2].unsqueeze(2)
            tl_xy_padded = torch.nn.functional.pad(tl_xy, (0, 2))
            tl_xy_transformed = transform_coords_to_sdc_frame(tl_xy_padded, inputs['sdc_coord'])
            tl_result = torch.cat([tl_xy_transformed.squeeze(2)[..., :2], tl[..., 2:]], dim=-1)
            tl_result[~valid_mask] = 0.0
            inputs["traffic_light_points"] = tl_result
            
            pred_traj = []
            for r in range(self._n_rollout_closed_val):
                print('closed-loop rollout', r)
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
                        tl = inputs["traffic_light_points"]
                        valid_mask = torch.any(tl != 0, dim=-1)
                        tl_xy = tl[..., :2].unsqueeze(2)
                        tl_xy_padded = torch.nn.functional.pad(tl_xy, (0, 2))
                        tl_xy_transformed = transform_coords_to_sdc_frame(tl_xy_padded, inputs['sdc_coord'])
                        tl_result = torch.cat([tl_xy_transformed.squeeze(2)[..., :2], tl[..., 2:]], dim=-1)
                        tl_result[~valid_mask] = 0.0
                        inputs["traffic_light_points"] = tl_result
                        
                        current_states_global = prev_pred_global[:, :self._predicted_neighbor_num+1, -1, :6].clone()  # [B, P, 6]

                    inputs_normalized = self.observation_normalizer(inputs_sdc)
                    # inputs_normalized = inputs_sdc

                    _, decoder_output = self.forward(inputs_normalized)
                    pred_trajectories = decoder_output["prediction"]  # [B, P, future_len, 4]
                    
                    # Convert local trajectories to global frame using each agent's current state
                    pred_trajs_global = batch_transform_trajs_to_global_frame(
                        pred_trajectories,
                        current_states_global[..., :4]  # [B, P, 4] - [x, y, cos, sin]
                    )  # [B, P, future_len, 4]
                    
                    # Zero out invalid agents' trajectories
                    if agents_mask is not None:
                        pred_trajs_global = pred_trajs_global * agents_mask[..., None, None].float()
                    
                    if t == 2:
                        pred_trajs_sdc = transform_coords_to_sdc_frame(pred_trajs_global, sdc_coord_global)
                        pred_trajs_3d = torch.cat([
                            pred_trajs_sdc[..., :2],
                            torch.zeros_like(pred_trajs_sdc[..., 0:1]),
                            torch.atan2(pred_trajs_sdc[..., 3:4], pred_trajs_sdc[..., 2:3])
                        ], dim=-1)
                        self._log_output(
                            pred_trajs_3d,
                            inputs_sdc,
                            batch_idx,
                            trans2global=False,
                            sdc_coord=batch['sdc_coord'],
                        )

                    pred_trajs_step_global = pred_trajs_global[:, :, :step_len, :]  # [B, P, step_len, 4]
                    trajs.append(pred_trajs_step_global)
                    
                    # Update SDC position for next step from ego's last predicted position
                    ego_pred_global = pred_trajs_step_global[:, 0, step_len-1, :]  # [B, 4]
                    sdc_coord_global[:, 0] = ego_pred_global[:, 0]  # x
                    sdc_coord_global[:, 1] = ego_pred_global[:, 1]  # y
                    sdc_coord_global[:, 2] = torch.atan2(ego_pred_global[:, 3], ego_pred_global[:, 2])  # theta from cos/sin
                    
                    # Update history in global frame for next step
                    dt = 0.1
                    pos_global = torch.cat([current_states_global[:, :, None, :2], pred_trajs_global[..., :2]], dim=-2)  # [B, P, future_len, 2]
                    v = torch.diff(pos_global, dim=-2) / dt  # [B, P, future_len-1, 2]
                    v_step = v[:, :, :step_len, :]  # [B, P, step_len, 2]
                    pred_trajs_step_global_6d = torch.cat([pred_trajs_step_global, v_step], dim=-1)  # [B, P, step_len, 6]
                    
                    hist_len = batch['agents_history'].shape[2]
                    prev_pred_global = torch.cat(
                        [
                            prev_pred_global[:, :, -(hist_len-step_len):, :], pred_trajs_step_global_6d
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
        for name, p in self.named_parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any():
                    print(f"[NaN GRAD] step {self.global_step}: {name}, nan_count={torch.isnan(p.grad).sum()}")
                    has_nan_grad = True
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        if has_nan_grad:
            print(f"[NaN GRAD] step {self.global_step}: gradient_norm={total_norm}")
        
        self.log("grad/global_norm_raw", total_norm, on_step=True, on_epoch=True, prog_bar=False)
    
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
            # self.log("grad/global_norm", global_norm, on_step=False, on_epoch=True)
    
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
        inputs_norm = self.observation_normalizer(inputs)
        # inputs_norm = inputs

        agents_future = inputs["agents_future"]
        agents_future_norm = inputs_norm["agents_future"]  # [B, P, T+1, 9]
        agents_future_valid = inputs_norm["agents_future_valid"]  # [B, P, T+1]
        agents_interested = inputs_norm["agents_interested"]  # [B, P]
        B, P, T, _ = agents_future.shape
        
        current_states = inputs["agents_history"][:, :, -1:, :6].clone()  # [B, P, 1, 6]

        # Get ground truth future states [B, P, T, 3] - x, y, yaw (from cos and sin)
        gt_future_pos = agents_future[..., 1:, :2].clone()  # [B, P, T, 2] - x, y
        gt_future_yaw = torch.atan2(agents_future[..., 1:, 3], agents_future[..., 1:, 2])  # [B, P, T]
        
        future_mask = agents_future_valid[..., 1:]*(agents_interested[..., None]>0)  # [B, P, T]

        if self.training:
            t = torch.rand(B, device=agents_future.device) * (1 - eps) + eps # [B,]
            
            # Direct trajectory prediction: add noise to GT trajectory [x, y, cos, sin]
            gt_trajs_global = agents_future[..., :, :4]  # [B, P, T, 4] - x, y, cos, sin (global frame)
            gt_trajs = batch_transform_trajs_to_local_frame(
                gt_trajs_global, ref_idx=0
            )[:, :, 1:, :]  # [B, P, T, 4] - GT in local frame
            
            z = torch.randn_like(gt_trajs, device=agents_future.device) # [B, P, T, 4]
            
            mean, std = marginal_prob(self.state_normalizer(gt_trajs), t)
            std = std.view(-1, *([1] * (len(gt_trajs.shape)-1)))

            xT = mean + std * z
            
            merged_inputs = {
                **inputs_norm,
                # **inputs,
                "sampled_trajectories": xT,
                "diffusion_time": t,
                # "current_states": current_states,  # Pass current states for potential use
            }

            _, decoder_output = self.forward(merged_inputs)
            score = decoder_output["score"]  # [B, P, T, 4]
            
            gt_trajs_norm = self.state_normalizer(gt_trajs)  # [B, P, T, 4]
            
            # Direct single-modal regression loss
            state_loss = mse_loss(score, gt_trajs_norm, reduction='none').sum(-1)  # [B, P, T]
            loss = state_loss * future_mask

            pred_trajectories = self.state_normalizer.inverse(score)  # [B, P, T, 4]
            decoder_output["predicted_trajectories"] = pred_trajectories
            ref_state = current_states[..., :4]  # [B, P, 1, 4]
            
            loss_dict["train/state_loss"] = (state_loss * future_mask).sum() / future_mask.sum()
            
            denoise_mse = torch.norm(pred_trajectories[..., :2] - gt_trajs[..., :2], dim=-1)
            denoise_ADE = denoise_mse[future_mask.bool()].mean()
            denoise_FDE = denoise_mse[..., -1][future_mask[..., -1].bool()].mean()
            loss_dict["train/denoise_ADE"] = denoise_ADE
            loss_dict["train/denoise_FDE"] = denoise_FDE

            # Diffusion loss: directly supervise x_0 prediction in trajectory space
            # diffusion_loss = mse_loss(pred_trajectories, gt_trajs, reduction='none').sum(-1)  # [B, P, T]
            # diffusion_loss = (diffusion_loss * future_mask).sum() / future_mask.sum()
            # loss_dict["train/diffusion_loss"] = diffusion_loss.item()
        else:
            _, decoder_output = self.forward(inputs_norm)
            pred_trajectories = decoder_output["prediction"]  # [B, P, future_len, 4]
            
            P_val = self._predicted_neighbor_num + 1
            pred_trajectories = pred_trajectories[:, :P_val]  # [B, P', future_len, 4]

            gt_trajs_global = agents_future[:, :P_val, :, :4]
            gt_trajs = batch_transform_trajs_to_local_frame(
                gt_trajs_global, ref_idx=0
            )[:, :, 1:, :]  # [B, P', future_len, 4] - GT in local frame

            decoder_output["predicted_trajectories"] = pred_trajectories

            state_loss = mse_loss(pred_trajectories, gt_trajs, reduction='none').sum(-1)  # [B, P', future_len]

            loss_dict["val/state_loss"] = (state_loss * future_mask[:, :P_val, :]).sum() / future_mask[:, :P_val, :].sum()
            
            denoise_mse = torch.norm(pred_trajectories[..., :2] - gt_trajs[..., :2], dim=-1)
            denoise_ADE = denoise_mse[future_mask[:, :P_val, :].bool()].mean()
            denoise_FDE = denoise_mse[..., -1][future_mask[:, :P_val, -1].bool()].mean()
            loss_dict["val/denoise_ADE"] = denoise_ADE
            loss_dict["val/denoise_FDE"] = denoise_FDE
            
            loss = state_loss * future_mask[:, :P_val, :]
        
        # ############### Behavior Prior Prediction #################
        # if self.training and self._train_predictor and self.predictor is not None:
        #     # Get anchors from inputs
        #     anchors = inputs.get("anchors", None)
        #     if anchors is not None:
        #         goal_actions, goal_scores = self.predictor(encoder_outputs, anchors)
                
        #         # Roll out predicted actions to get trajectories
        #         goal_trajs = roll_out(
        #             current_states.squeeze(2)[:, :self.predictor._agents_len],  # [B, P, 6]
        #             goal_actions.flatten(2, 3),  # [B, P*Q, num_actions, 2]
        #             dt=0.1,
        #             action_len=self._action_len,
        #             global_frame=True,
        #             training=True
        #         )  # [B, P*Q, T, 6]
                
        #         # Reshape back to [B, P, Q, T, 6]
        #         goal_trajs = goal_trajs.view(B, self.predictor._agents_len, -1, self._future_len, 6)
                
        #         # Calculate goal loss (similar to VBD)
        #         # Get ground truth future
        #         gt_future = agents_future[:, :self.predictor._agents_len, 1:, :4]  # [B, P, T, 4]
        #         gt_future_valid = agents_future_valid[:, :self.predictor._agents_len, 1:]  # [B, P, T]
        #         agents_interest = agents_interested[:, :self.predictor._agents_len]  # [B, P]
                
        #         # Find closest anchor to ground truth end point
        #         goal_gt = agents_future[:, :self.predictor._agents_len, -1:, :2]  # [B, P, 1, 2]
        #         # anchors_global = transform_coords_to_global_frame(
        #         #     anchors[:, :self.predictor._agents_len], 
        #         #     inputs['sdc_coord']
        #         # )  # [B, P, Q, 2]
                
        #         # Find closest anchor
        #         dist_to_goal = torch.norm(anchors[:, :self.predictor._agents_len] - goal_gt, dim=-1)  # [B, P, Q]
        #         idx_anchor = torch.argmin(dist_to_goal, dim=-1)  # [B, P]
                
        #         # Find trajectory with min ADE
        #         trajs_pred = goal_trajs[..., :4]  # [B, P, Q, T, 4]
        #         dist = torch.norm(trajs_pred - gt_future[:, :, None, :, :], dim=-1)  # [B, P, Q, T]
        #         dist = dist * gt_future_valid[:, :, None, :]  # [B, P, Q, T]
        #         idx_min_ade = torch.argmin(dist.mean(-1), dim=-1)  # [B, P]
                
        #         # Select based on whether end point is valid
        #         idx = torch.where(
        #             agents_future_valid[:, :self.predictor._agents_len, -1], 
        #             idx_anchor, 
        #             idx_min_ade
        #         )  # [B, P]
                
        #         # Select the best trajectory for each agent
        #         batch_idx = torch.arange(B)[:, None].expand(B, self.predictor._agents_len).flatten()
        #         agent_idx = torch.arange(self.predictor._agents_len)[None, :].expand(B, self.predictor._agents_len).flatten()
        #         selected_trajs = goal_trajs[batch_idx, agent_idx, idx.flatten()]  # [B*P, T, 5]
                
        #         # Calculate trajectory loss
        #         traj_loss = smooth_l1_loss(selected_trajs[..., :4], gt_future.flatten(0, 1), reduction='none').sum(-1)  # [B*P, T]
        #         traj_mask = gt_future_valid.flatten(0, 1) * (agents_interest.flatten(0, 1) > 0).unsqueeze(-1)  # [B*P, T]
        #         traj_loss = traj_loss * traj_mask  # [B*P, T]
        #         goal_loss_mean = traj_loss.sum() / traj_mask.sum()
                
        #         # Calculate score loss (cross entropy)
        #         scores = goal_scores.flatten(0, 1)  # [B*P, Q]
        #         score_loss = cross_entropy(scores, idx.flatten(), reduction='none')  # [B*P]
        #         score_loss = score_loss * (agents_interest.flatten(0, 1) > 0)  # [B*P]
        #         score_loss_mean = score_loss.sum() / (agents_interest > 0).sum()
                
        #         # Combine losses
        #         pred_loss = goal_loss_mean + 0.05 * score_loss_mean
                
        #         # Add to total loss
        #         total_loss = loss.sum() / future_mask.sum() + pred_loss
                
        #         loss_dict["val/goal_loss"] = goal_loss_mean.item()
        #         loss_dict["val/score_loss"] = score_loss_mean.item()
        #         loss_dict["val/pred_loss"] = pred_loss.item()
        #     else:
        #         total_loss = loss.sum() / future_mask.sum()
        # else:
        #     total_loss = loss.sum() / future_mask[:, :self._predicted_neighbor_num+1, :].sum()

        if self.training:
            total_loss = loss.sum() / future_mask.sum()
            assert not torch.isnan(total_loss).any(), f"loss cannot be nan"
        else:
            total_loss = loss.sum() / future_mask[:, :P_val, :].sum()
        return total_loss, loss_dict, decoder_output
    
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
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
        self.apply(_basic_init)

        # Initialize embedding MLP:
        # nn.init.normal_(self.encoder.pos_emb.weight, std=0.02)
        nn.init.normal_(self.encoder.agents_encoder.type_emb.weight, std=0.02)
        nn.init.normal_(self.encoder.lane_encoder.speed_limit_emb.weight, std=0.02)
        nn.init.normal_(self.encoder.lane_encoder.traffic_emb.weight, std=0.02)
        nn.init.normal_(self.encoder.traffic_light_encoder.type_embed.weight, std=0.02)

    def forward(self, inputs):

        encoder_outputs = self.encoder(inputs)

        return encoder_outputs
    

class Diffusion_Planner_Decoder(nn.Module):
    def __init__(self, config, rel_encoder=None):
        super().__init__()

        self.decoder = Decoder(config, rel_encoder=rel_encoder)
        # self.decoder = QCDecoder(config, rel_encoder=rel_encoder)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.decoder.dit.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.decoder.dit.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.decoder.dit.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.decoder.dit.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.decoder.dit.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.decoder.dit.final_layer.proj[-1].weight, 0)
        nn.init.constant_(self.decoder.dit.final_layer.proj[-1].bias, 0)

    def forward(self, encoder_outputs, inputs):

        decoder_outputs = self.decoder(encoder_outputs, inputs)
        
        return decoder_outputs