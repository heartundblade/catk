"""Agent-type-aware flow-matching pre-training planner (Flow-ERD AFM stage)."""

from typing import Any, Dict, Tuple

import lightning.pytorch as pl
import torch
from torch.nn.functional import mse_loss

from src.dp.model.diffusion_planner import (
    Diffusion_Planner,
    Diffusion_Planner_Encoder,
)
# from src.dp.model.flow_matching_utils import affine_ot_path, masked_flow_mse  # moved to src.flow
# from src.dp.model.module.flow_matching_decoder import FlowMatchingDecoder  # moved to src.flow
from src.flow.flow_matching_utils import affine_ot_path, masked_flow_mse
from src.flow.flow_matching_decoder import FlowMatchingDecoder
from src.dp.model.module.rel_emb import RelationEncoder
from src.dp.utils.normalizer import ActionNormalizer
from src.dp.utils.train_utils import inverse_kinematics_bicycle, roll_out_bicycle
from src.smart.metrics import WOSACMetric, WOSACMetrics, WOSACSubmission


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_float = mask.to(device=value.device, dtype=value.dtype)
    count = mask_float.sum()
    result = (value * mask_float).sum() / count.clamp_min(1.0)
    return result * (count > 0).to(result.dtype)


class FlowMatchingPlanner(Diffusion_Planner):
    """Parallel AFM planner that leaves the diffusion baseline untouched."""

    def __init__(self, config):
        # Initialize Lightning directly: constructing Diffusion_Planner would also
        # allocate an incompatible diffusion decoder before replacing it.
        pl.LightningModule.__init__(self)
        self.save_hyperparameters()
        self.cfg = config

        self.action_normalizer = ActionNormalizer.from_json(config)
        self.rel_encoder = RelationEncoder(
            hidden_dim=config.hidden_dim,
            num_freq_bands=64,
        )
        self.encoder = Diffusion_Planner_Encoder(config, self.rel_encoder)
        self.decoder = FlowMatchingDecoder(config)

        self._predicted_neighbor_num = config.get("predicted_neighbor_num", 31)
        self._future_len = config.get("future_len", 80)
        self._step_len = config.get("step_len", 10)
        self._val_open_loop = config.get("val_open_loop", True)
        self._val_closed_loop = config.get("val_closed_loop", False)
        self._n_rollout_closed_val = config.get("n_rollout_closed_val", 32)
        self.log_epoch = config.get("log_epoch", -1)
        self._action_len = config.get("action_len", 1)
        self._num_actions = self._future_len
        self._rho_fixed = config.get("rho_fixed", None)

        if config.get("fast_wosac_metric", False):
            self.wosac_metrics = WOSACMetric("2024")
        else:
            self.wosac_metrics = WOSACMetrics("val_closed")
        self.wosac_submission = WOSACSubmission(
            **config.get("wosac_submission", {})
        )

    @property
    def flow_steps(self):
        return self.decoder.flow_steps

    @property
    def flow_solver(self):
        return self.decoder.flow_solver

    @property
    def flow_noise_scale(self):
        return self.decoder.flow_noise_scale

    def load_encoder_state_dict(self, state_dict, strict: bool = True):
        """Explicitly transfer only encoder weights from a compatible checkpoint."""
        return self.encoder.load_state_dict(state_dict, strict=strict)

    def _clone_batch(self, batch):
        return {
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

    def _action_targets(self, inputs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        predicted_agents = self._predicted_neighbor_num + 1
        actions, action_valid, radius = inverse_kinematics_bicycle(
            inputs["agents_future"],
            inputs["agents_future_valid"],
            agent_type=inputs["agents_type"],
            dt=0.1,
            rho_fixed=self._rho_fixed,
        )
        if actions.shape[1] < predicted_agents or actions.shape[2] < self._future_len:
            raise ValueError(
                "ground-truth actions do not cover the configured predicted agents and future_len"
            )

        actions = actions[:, :predicted_agents, :self._future_len]
        action_valid = action_valid[:, :predicted_agents, :self._future_len].bool()
        interested = inputs["agents_interested"][:, :predicted_agents] > 0
        valid_mask = action_valid & interested.unsqueeze(-1)
        return self.action_normalizer(actions), valid_mask, radius[:, :predicted_agents]

    def flow_loss_func(
        self,
        inputs: Dict[str, torch.Tensor],
        prefix: str,
    ):
        """Affine OT conditional flow-matching objective (paper Eqs. 7-9)."""
        clean_actions, valid_mask, radius = self._action_targets(inputs)
        batch_size = clean_actions.shape[0]

        initial_noise = torch.randn_like(clean_actions) * self.flow_noise_scale
        interested = inputs["agents_interested"][:, :clean_actions.shape[1]] > 0
        initial_noise = initial_noise * interested[:, :, None, None].to(initial_noise.dtype)
        flow_time = torch.rand(
            batch_size, device=clean_actions.device, dtype=clean_actions.dtype
        )
        flow_state, target_velocity = affine_ot_path(
            initial_noise, clean_actions, flow_time
        )

        merged_inputs = {
            **inputs,
            "flow_state": flow_state,
            "flow_time": flow_time,
        }
        _, decoder_output = self.forward(merged_inputs)
        predicted_velocity = decoder_output["velocity"]
        loss, component_loss = masked_flow_mse(
            predicted_velocity, target_velocity, valid_mask
        )

        loss_dict = {
            f"{prefix}/flow_loss": loss,
            f"{prefix}/flow_a_parallel_loss": component_loss[0],
            f"{prefix}/flow_a_lateral_loss": component_loss[1],
            f"{prefix}/flow_a_psi_loss": component_loss[2],
        }

        # Reconstruct the clean endpoint only for interpretable diagnostics. It is
        # deliberately not included in the optimized objective.
        remaining_time = (1.0 - flow_time).view(batch_size, 1, 1, 1)
        estimated_clean = flow_state + remaining_time * predicted_velocity
        diagnostics = self._trajectory_diagnostics(
            estimated_clean.detach(), inputs, valid_mask, radius, prefix
        )
        loss_dict.update(diagnostics)
        decoder_output["estimated_clean_actions"] = estimated_clean
        return loss, loss_dict, decoder_output

    def _trajectory_diagnostics(
        self,
        normalized_actions,
        inputs,
        action_valid,
        radius,
        prefix,
    ):
        predicted_agents = normalized_actions.shape[1]
        # This is the sole inverse-normalization before kinematic execution.
        physical_actions = self.action_normalizer.inverse(normalized_actions)
        current_states = inputs["agents_history"][:, :predicted_agents, -1, :6]
        trajectories = roll_out_bicycle(
            current_states,
            physical_actions,
            r=radius,
            agent_type=inputs["agents_type"][:, :predicted_agents],
            dt=0.1,
            valid_mask=inputs["agents_interested"][:, :predicted_agents] > 0,
        )

        horizon = trajectories.shape[2]
        gt_future = inputs["agents_future"][:, :predicted_agents, 1:horizon + 1]
        future_valid = inputs["agents_future_valid"][
            :, :predicted_agents, 1:horizon + 1
        ].bool()
        future_valid = future_valid & (
            inputs["agents_interested"][:, :predicted_agents, None] > 0
        )

        displacement = torch.linalg.vector_norm(
            trajectories[..., :2] - gt_future[..., :2], dim=-1
        )
        predicted_yaw = torch.atan2(trajectories[..., 3], trajectories[..., 2])
        gt_yaw = torch.atan2(gt_future[..., 3], gt_future[..., 2])
        yaw_error = torch.atan2(
            torch.sin(predicted_yaw - gt_yaw),
            torch.cos(predicted_yaw - gt_yaw),
        )

        clean_actions, _, _ = inverse_kinematics_bicycle(
            inputs["agents_future"],
            inputs["agents_future_valid"],
            agent_type=inputs["agents_type"],
            dt=0.1,
            rho_fixed=self._rho_fixed,
        )
        clean_actions = clean_actions[:, :predicted_agents, :horizon]
        action_error = mse_loss(
            physical_actions, clean_actions, reduction="none"
        ).sum(dim=-1)

        return {
            f"{prefix}/action_loss": _masked_mean(action_error, action_valid),
            f"{prefix}/denoise_ADE": _masked_mean(displacement, future_valid),
            f"{prefix}/denoise_FDE": _masked_mean(
                displacement[..., -1], future_valid[..., -1]
            ),
            f"{prefix}/yaw_MSE": _masked_mean(yaw_error.square(), future_valid),
        }

    def training_step(self, batch, batch_idx):
        inputs = self._clone_batch(batch)
        loss, loss_dict, _ = self.flow_loss_func(inputs, prefix="train")
        for key, value in loss_dict.items():
            self.log(
                key,
                value,
                on_step=True,
                on_epoch=False,
                sync_dist=True,
                prog_bar=(key == "train/flow_loss"),
            )
        if not torch.isfinite(loss):
            raise FloatingPointError("flow-matching loss is not finite")
        return loss

    def validation_step(self, batch, batch_idx):
        loss_dict: Dict[str, Any] = {}
        if self._val_open_loop:
            inputs = self._clone_batch(batch)
            _, objective_metrics, _ = self.flow_loss_func(inputs, prefix="val")
            loss_dict.update(objective_metrics)

            # ODE-sampled predictions drive the existing open-loop metrics.
            _, decoder_output = self.forward(inputs)
            _, valid_mask, radius = self._action_targets(inputs)
            sampled_metrics = self._trajectory_diagnostics(
                decoder_output["prediction"], inputs, valid_mask, radius, "val"
            )
            loss_dict.update(sampled_metrics)
            self.log_dict(
                loss_dict,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                prog_bar=True,
            )

        if self._val_closed_loop:
            # The baseline routine calls self.forward at each receding-horizon
            # step, so it naturally uses fresh flow noise while preserving global
            # history feedback, fixed map inputs, logging, and WOSAC formatting.
            open_loop_setting = self._val_open_loop
            self._val_open_loop = False
            try:
                Diffusion_Planner.validation_step(self, batch, batch_idx)
            finally:
                self._val_open_loop = open_loop_setting

        return loss_dict