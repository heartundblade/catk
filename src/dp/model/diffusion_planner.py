import os
import pickle
from typing import Any, Callable, Dict, List, Tuple

import torch
import torch.nn as nn

import lightning.pytorch as pl

from src.dp.model.module.encoder import Encoder
from src.dp.model.module.decoder import Decoder
from src.dp.utils.normalizer import StateNormalizer, ObservationNormalizer
from src.dp.utils.lr_schedule import CosineAnnealingWarmUpRestarts
from src.dp.utils.train_utils import transform_coords_to_sdc_frame, transform_coords_to_global_frame
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

        self.encoder = Diffusion_Planner_Encoder(self.cfg)
        self.decoder = Diffusion_Planner_Decoder(self.cfg)
        
        # Validation settings
        self._future_len = config.get('future_len', 80)
        self._step_len = config.get('step_len', 10)
        self._val_open_loop = config.get('val_open_loop', True)
        self._val_closed_loop = config.get('val_closed_loop', False)
        self._n_rollout_closed_val = config.get('n_rollout_closed_val', 32)
        self.log_epoch = config.get('log_epoch', -1)
        
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.learning_rate)
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

        inputs = self.observation_normalizer(inputs)

        # inputs['agents_future'][..., :4] = self.state_normalizer(inputs['agents_future'][..., :4])
        # self._log_output(
        #     None, 
        #     inputs, 
        #     batch_idx,
        #     trans2global=False,
        #     sdc_coord=batch['sdc_coord'],
        # )

        loss = {}
        dpm_loss, loss, _ = self.diffusion_loss_func(
            inputs=inputs,
            marginal_prob=self.sde.marginal_prob,
            # futures=batch["agents_future"],
            norm=self.state_normalizer,
            loss=loss,
            model_type="x_start",
        )

        # Log training losses for monitoring
        self.log("train/dpm_loss", dpm_loss, prog_bar=True)
        self.log("train/ego_planning_loss", loss["ego_planning_loss"], prog_bar=True)
        self.log("train/neighbor_prediction_loss", loss["neighbor_prediction_loss"])
        self.log("train/total_loss", loss["ego_planning_loss"] + loss["neighbor_prediction_loss"])
        
        return dpm_loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step of the model.

        Args:
            batch: Input batch.
            batch_idx: Batch index.
        """
        # Open-loop validation
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
            
            inputs = self.observation_normalizer(inputs)

            loss = {}
            dpm_loss, loss, decoder_output = self.diffusion_loss_func(
                inputs=inputs,
                marginal_prob=self.sde.marginal_prob,
                norm=self.state_normalizer,
                loss=loss,
                model_type="x_start",
            )
            
            # Log validation losses for monitoring
            self.log("val/dpm_loss", dpm_loss, prog_bar=True, batch_size=inputs["agents_history"].shape[0])
            self.log("val/ego_planning_loss", loss["ego_planning_loss"], prog_bar=True)
            self.log("val/neighbor_prediction_loss", loss["neighbor_prediction_loss"])
            self.log("val/total_loss", loss["ego_planning_loss"] + loss["neighbor_prediction_loss"])

            # Record decoder output and map elements
            # self._log_output(
            #     decoder_output['prediction'], 
            #     batch, 
            #     batch_idx,
            #     trans2global=True,
            #     sdc_coord=batch['sdc_coord'],
            # )

        # Closed-loop validation
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
            
            pred_traj = []
            for r in range(self._n_rollout_closed_val):
                print('closed-loop rollout', r)
                traj = []
                inputs_ = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                
                # Track SDC position for each step (in global frame)
                sdc_coord_global = batch['sdc_coord'].clone()  # [B, 3]
                
                for t in range((future_len + step_len - 1) // step_len):
                    # Convert all map elements to current SDC frame
                    # For t=0, inputs_ is already in initial SDC frame
                    # For t>0, inputs_ needs to be converted from previous SDC frame to current SDC frame
                    if t > 0:
                        # Convert to current SDC frame
                        inputs_['agents_history'][..., :6] = transform_coords_to_sdc_frame(
                            inputs_['agents_history'][..., :6],
                            prev_sdc_coord
                        )
                        inputs_['lanes'][..., :4] = transform_coords_to_sdc_frame(
                            inputs_['lanes'][..., :4],
                            prev_sdc_coord
                        )
                        inputs_['roadlines'][..., :4] = transform_coords_to_sdc_frame(
                            inputs_['roadlines'][..., :4],
                            prev_sdc_coord
                        )
                        inputs_['static_maps'][..., :4] = transform_coords_to_sdc_frame(
                            inputs_['static_maps'][..., :4],
                            prev_sdc_coord
                        )

                        inputs_['agents_future'][..., :6] = transform_coords_to_sdc_frame(
                            inputs_['agents_future'][..., :6],
                            prev_sdc_coord
                        )
                    
                    # prev_sdc_coord = sdc_coord_global.clone()
                    inputs_normalized = self.observation_normalizer(inputs_)
                    
                    # Get prediction from decoder (in SDC frame)
                    _, decoder_output = self.forward(inputs_normalized)
                    pred = decoder_output["prediction"]  # [B, P, T, 4]
                    
                    # if t==0:
                    #     self._log_output(
                    #         # inputs_["agents_history"][..., :6],
                    #         pred,
                    #         inputs_,
                    #         # batch,
                    #         batch_idx,
                    #         log_dir='/home/zhanghailiang/Repos/catk/logs/debug',
                    #         trans2global=False,
                    #         sdc_coord=None
                    #     )
                    
                    pred_trajs_step_sdc = pred[:, :, :step_len, :]  # [B, P, step_len, 4]
                    
                    # Convert prediction to global frame for storage and SDC update
                    # prev_coord_pt = pred_trajs_step_sdc[:, 0, -1, :].clone()
                    prev_coord_pt = pred_trajs_step_sdc[:, 0, -1, :].clone()  # [B, 4]
                    prev_sdc_coord = torch.zeros_like(prev_coord_pt[:, :3])  # [B, 3]
                    prev_sdc_coord[:, :2] = prev_coord_pt[:, :2]  # x, y
                    prev_sdc_coord[:, 2] = torch.atan2(prev_coord_pt[:, 3], prev_coord_pt[:, 2])
                    pred_trajs_step_global = transform_coords_to_global_frame(
                        pred_trajs_step_sdc,
                        sdc_coord_global
                    )  # [B, P, step_len, 4]
                    
                    traj.append(pred_trajs_step_global)
                    
                    # Update SDC position based on ego vehicle's prediction
                    # Assuming ego is the first agent (index 0)
                    ego_pred_global = pred_trajs_step_global[:, 0, self._step_len-1, :]  # [B, 4]
                    sdc_coord_global[:, 0] = ego_pred_global[:, 0]  # x
                    sdc_coord_global[:, 1] = ego_pred_global[:, 1]  # y
                    sdc_coord_global[:, 2] = torch.atan2(ego_pred_global[:, 3], ego_pred_global[:, 2])
                    
                    # Update inputs_ for next step (closed-loop)
                    # Shift history and append predicted traj (in SDC frame)
                    current_states = inputs_['agents_history'][:, :, -1:, :6]  # [B, P, 1, 6] - include vx, vy
                    
                    # Compute velocity from position differences
                    # pred_trajs_step_sdc: [B, P, step_len, 4] -> [x, y, cos_heading, sin_heading]
                    # velocity = (pos[t] - pos[t-1]) / dt
                    dt = 0.1  # assuming 10Hz
                    pos_history = torch.cat([current_states[:, :, :, :2], pred_trajs_step_sdc[:, :, :, :2]], dim=2)  # [B, P, step_len+1, 2]
                    vel = (pos_history[:, :, 1:, :] - pos_history[:, :, :-1, :]) / dt  # [B, P, step_len, 2]
                    pred_with_vel = torch.cat([pred_trajs_step_sdc, vel], dim=-1)  # [B, P, step_len, 6]
                    new_history = torch.cat([current_states, pred_with_vel], dim=2)  # [B, P, step_len+1, 6]
                    
                    inputs_['agents_history'][:, :, :, :6] = new_history
                    # Note: new_history is already in current SDC frame, no need to transform again
                
                full_traj = torch.cat(traj, dim=-2)  # [B, P, future_len, 4] - in global frame
                pred_traj.append(full_traj)
            
            pred_traj = torch.stack(pred_traj, dim=1)  # [B, n_rollout, P, future_len, 4]
            
            # Handle remaining agents (linear extrapolation)
            all_agents_id_list = [x for x in batch["agents_id"]]
            simulated_states_list = [x for x in pred_traj]
            
            for i in range(batch_size):
                if 'agents_history_remaining' in batch and batch['agents_history_remaining'][i].shape[0] > 0:
                    num_agents_remaining = batch['agents_history_remaining'][i].shape[0]
                    cur_data = batch['agents_history_remaining'][i].clone()  # [A, T, 9]
                    
                    # Concatenate agent ids
                    if 'agents_id_remaining' in batch:
                        all_agents_id_list[i] = torch.cat([all_agents_id_list[i], batch['agents_id_remaining'][i]], dim=0)
                    
                    # Linear extrapolation for remaining agents
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
                    
                    # Linear extrapolation: pos[t] = pos_0 + vel * t
                    time_steps = torch.arange(future_len, device=self.device).float() * 0.1  # [future_len]
                    
                    pos_expanded = pos.unsqueeze(1).unsqueeze(0)  # [1, A, 1, 2]
                    vel_expanded = vel.unsqueeze(1).unsqueeze(0)  # [1, A, 1, 2]
                    time_expanded = time_steps.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # [1, 1, future_len, 1]
                    
                    pred_pos = pos_expanded + vel_expanded * time_expanded  # [1, A, future_len, 2]
                    
                    # Heading and z remain constant
                    heading_expanded = heading.unsqueeze(0).unsqueeze(-1).expand(
                        self._n_rollout_closed_val, num_agents_remaining, future_len
                    )
                    z_expanded = z.unsqueeze(0).unsqueeze(-1).expand(
                        self._n_rollout_closed_val, num_agents_remaining, future_len
                    )
                    
                    # Format: [n_rollout, A, future_len, 4] -> [x, y, z, heading]
                    simulated_states_remaining[..., :2] = pred_pos.expand(
                        self._n_rollout_closed_val, num_agents_remaining, future_len, 2
                    )
                    simulated_states_remaining[..., 2] = z_expanded
                    simulated_states_remaining[..., 3] = heading_expanded
                    
                    # Concatenate with main predictions
                    simulated_states_list[i] = torch.cat(
                        [simulated_states_list[i], simulated_states_remaining], dim=1
                    )

            # self._log_output(
            #     [s.cpu().detach().numpy() for s in simulated_states_list], 
            #     batch, 
            #     batch_idx, 
            #     log_dir='/home/zhanghailiang/Repos/catk/logs/debug',
            #     trans2global=False,
            # )
            
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
        
        return loss
    
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
    
    def diffusion_loss_func(
        self,
        inputs: Dict[str, torch.Tensor],
        marginal_prob: Callable[[torch.Tensor], torch.Tensor],

        # futures: Tuple[torch.Tensor, torch.Tensor],
        
        norm: StateNormalizer,
        loss: Dict[str, Any],

        model_type: str,
        eps: float = 1e-3,
    ):
        agents_future = inputs["agents_future"]  # [B, P, T+1, 9]
        agents_future_valid = inputs["agents_future_valid"]  # [B, P, T+1]

        B, P, T, _ = agents_future.shape
        # ego_current, neighbors_current = inputs["ego_current_state"][:, :4], inputs["neighbor_agents_past"][:, :Pn, -1, :4]
        # neighbor_current_mask = torch.sum(torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0
        # neighbor_mask = torch.concat((neighbor_current_mask.unsqueeze(-1), neighbor_future_mask), dim=-1)

        # gt_future = torch.cat([ego_future[:, None, :, :], neighbors_future[..., :]], dim=1) # [B, P = 1 + 1 + neighbor, T, 4]
        current_states = inputs["agents_history"][:, :, -1:, :4].clone() # [B, P, 1, 4]
        gt_future = torch.cat([current_states, norm(agents_future[..., 1:, :4])], dim=2) # [B, P, T+1, 4]
        gt_future = gt_future * agents_future_valid.unsqueeze(-1)

        if self.training:
            t = torch.rand(B, device=agents_future.device) * (1 - eps) + eps # [B,]
            z = torch.randn_like(agents_future[..., 1:, :4], device=agents_future.device) # [B, P, T, 4]
            
            mean, std = marginal_prob(gt_future[:, :, 1:, :], t)
            std = std.view(-1, *([1] * (len(gt_future[:, :, 1:, :].shape)-1)))

            xT = mean + std * z
            xT = torch.cat([gt_future[:, :, :1, :], xT], dim=2)
            
            merged_inputs = {
                **inputs,
                "sampled_trajectories": xT,
                "diffusion_time": t,
            }

            _, decoder_output = self.forward(merged_inputs)
            # Training mode returns "score"
            pred = decoder_output["score"][:, :, 1:, :]  # [B, P, T, 4]

            self._log_output(
                pred, 
                inputs, 
                0,
                trans2global=False,
                # sdc_coord=batch['sdc_coord'],
            )

            # Compute dpm_loss based on model_type
            if model_type == "score":
                dpm_loss = torch.sum((pred * std + z)**2, dim=-1)
            elif model_type == "x_start":
                dpm_loss = torch.sum((pred - gt_future[:, :, 1:, :])**2, dim=-1)
        
        else:
            _, decoder_output = self.forward(inputs)
            pred = decoder_output["prediction"]  # [B, P, T, 4]

            agents_future = inputs["agents_future"][:, :, 1:, :4]  # [B, P, T, 4]
            dpm_loss = torch.sum((pred - agents_future)**2, dim=-1)
        
        neighbor_future_valid = agents_future_valid.clone()
        neighbor_future_valid[:, 0, :] = False
        masked_prediction_loss = dpm_loss[neighbor_future_valid[:, :, 1:]]

        if masked_prediction_loss.numel() > 0:
            loss["neighbor_prediction_loss"] = masked_prediction_loss.mean()
        else:
            loss["neighbor_prediction_loss"] = torch.tensor(0.0, device=masked_prediction_loss.device)

        loss["ego_planning_loss"] = dpm_loss[:, 0, :].mean()

        assert not torch.isnan(dpm_loss).sum(), f"loss cannot be nan, z={z}"

        return dpm_loss[agents_future_valid[:, :, 1:]].mean(), loss, decoder_output
    
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
        
        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)
        
        if batch_idx == 0: 
            log_file_path = os.path.join(log_dir, f'output_log_{batch_idx}.pkl')
            with open(log_file_path, 'wb') as f:
                pickle.dump(log_data, f)
    
class Diffusion_Planner_Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.encoder = Encoder(config)
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
        nn.init.normal_(self.encoder.pos_emb.weight, std=0.02)
        nn.init.normal_(self.encoder.agents_encoder.type_emb.weight, std=0.02)
        nn.init.normal_(self.encoder.lane_encoder.speed_limit_emb.weight, std=0.02)
        nn.init.normal_(self.encoder.lane_encoder.traffic_emb.weight, std=0.02)

    def forward(self, inputs):

        encoder_outputs = self.encoder(inputs)

        return encoder_outputs
    

class Diffusion_Planner_Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.decoder = Decoder(config)
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