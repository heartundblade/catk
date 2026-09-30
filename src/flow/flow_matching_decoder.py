"""Chunked DiT velocity field and fixed-step ODE sampler for AFM pre-training."""

from typing import Dict, Optional

import torch
import torch.nn as nn

from src.dp.model.diffusion_utils.sde import VPSDE_linear
# from src.dp.model.flow_matching_utils import integrate_flow_ode  # moved to src.flow
# from src.dp.model.module.decoder import DiT  # moved to src.flow (needs velocity support)
from src.flow.flow_matching_utils import integrate_flow_ode
from src.flow.decoder import DiT


class FlowMatchingDecoder(nn.Module):
    """Use the existing chunked DiT as a normalized-action velocity field."""

    def __init__(self, config):
        super().__init__()
        self._predicted_neighbor_num = config.get("predicted_neighbor_num", 31)
        self._future_len = config.get("future_len", 80)
        self._action_len = config.get("action_len", 1)
        self.flow_steps = int(config.get("flow_steps", 10))
        self.flow_solver = str(config.get("flow_solver", "heun")).lower()
        self.flow_noise_scale = float(config.get("flow_noise_scale", 1.0))

        if self.flow_steps <= 0:
            raise ValueError("flow_steps must be positive")
        if self.flow_solver not in ("euler", "heun"):
            raise ValueError("flow_solver must be either 'euler' or 'heun'")
        if self.flow_noise_scale < 0:
            raise ValueError("flow_noise_scale must be non-negative")

        self.velocity_net = DiT(
            # DiT only consults the SDE in score mode. Supplying the baseline SDE
            # keeps the shared module/checkpoint structure unchanged.
            sde=VPSDE_linear(),
            action_normalizer=None,
            depth=config.decoder_depth,
            output_dim=self._future_len * 3,
            hidden_dim=config.hidden_dim,
            heads=config.num_heads,
            dropout=config.decoder_drop_path_rate,
            future_len=self._future_len,
            model_type="velocity",
            agent_num=config.agent_num,
            num_chunks=config.num_chunks,
            action_len=self._action_len,
        )

    @property
    def dit(self):
        """Compatibility alias used by existing gradient diagnostics."""
        return self.velocity_net

    def _model_kwargs(self, encoder_outputs: Dict[str, torch.Tensor], inputs):
        predicted_agents = self._predicted_neighbor_num + 1
        current_states = inputs["agents_history"][:, :predicted_agents, -1, :6]
        return current_states, {
            "cross_c": encoder_outputs["encoding"],
            "current_states": current_states,
            "relation_encodings": None,
            "encoding_mask": encoder_outputs["encoding_mask"],
            "action_len": self._action_len,
        }

    def predict_velocity(self, encoder_outputs, inputs, flow_state, flow_time):
        current_states, model_kwargs = self._model_kwargs(encoder_outputs, inputs)
        batch_size, predicted_agents, horizon, action_dim = flow_state.shape
        if horizon != self._future_len or action_dim != 3:
            raise ValueError(
                f"flow_state must have shape [B,P,{self._future_len},3], got {flow_state.shape}"
            )
        if predicted_agents != current_states.shape[1]:
            raise ValueError("flow_state agent dimension does not match configured predicted agents")
        velocity = self.velocity_net(
            flow_state.reshape(batch_size, predicted_agents, -1),
            flow_time,
            **model_kwargs,
        )
        velocity = velocity.reshape(batch_size, predicted_agents, self._future_len, 3)
        valid_agents = ~encoder_outputs["encoding_mask"][:, :predicted_agents].bool()
        interested = inputs.get("agents_interested")
        if interested is not None:
            valid_agents = valid_agents & (interested[:, :predicted_agents] > 0)
        return velocity * valid_agents[:, :, None, None].to(velocity.dtype)

    @torch.no_grad()
    def sample(
        self,
        encoder_outputs,
        inputs,
        initial_noise: Optional[torch.Tensor] = None,
    ):
        current_states, _ = self._model_kwargs(encoder_outputs, inputs)
        batch_size, predicted_agents, _ = current_states.shape
        shape = (batch_size, predicted_agents, self._future_len, 3)

        if initial_noise is None:
            state = torch.randn(shape, device=current_states.device, dtype=current_states.dtype)
            state = state * self.flow_noise_scale
        else:
            if tuple(initial_noise.shape) != shape:
                raise ValueError(f"initial noise must have shape {shape}, got {initial_noise.shape}")
            state = initial_noise.to(device=current_states.device, dtype=current_states.dtype).clone()

        encoding_valid = ~encoder_outputs["encoding_mask"][:, :predicted_agents].bool()
        interested = inputs.get("agents_interested")
        if interested is not None:
            encoding_valid = encoding_valid & (interested[:, :predicted_agents] > 0)

        def velocity_fn(flow_state, flow_time):
            return self.predict_velocity(
                encoder_outputs, inputs, flow_state, flow_time
            )

        return integrate_flow_ode(
            velocity_fn,
            state,
            steps=self.flow_steps,
            solver=self.flow_solver,
            valid_agent_mask=encoding_valid,
        )

    def forward(self, encoder_outputs, inputs):
        if "flow_state" in inputs:
            return {
                "velocity": self.predict_velocity(
                    encoder_outputs,
                    inputs,
                    inputs["flow_state"],
                    inputs["flow_time"],
                )
            }

        return {
            "prediction": self.sample(
                encoder_outputs,
                inputs,
                initial_noise=inputs.get("flow_initial_noise"),
            )
        }