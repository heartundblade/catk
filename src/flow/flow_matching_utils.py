"""Small, model-independent utilities for affine optimal-transport flow matching."""

from typing import Callable, Tuple

import torch


def affine_ot_path(
    x0: torch.Tensor,
    x1: torch.Tensor,
    flow_time: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``x_lambda`` and its constant target velocity for the affine OT path."""
    if x0.shape != x1.shape:
        raise ValueError(f"x0 and x1 must have identical shapes, got {x0.shape} and {x1.shape}")
    if flow_time.ndim != 1 or flow_time.shape[0] != x0.shape[0]:
        raise ValueError("flow_time must have shape [B]")

    lam = flow_time.to(device=x0.device, dtype=x0.dtype)
    lam = lam.view(x0.shape[0], *([1] * (x0.ndim - 1)))
    target_velocity = x1 - x0
    return (1.0 - lam) * x0 + lam * x1, target_velocity


def masked_flow_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Masked component-wise MSE, with a finite zero loss for an empty mask."""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    if valid_mask.shape != prediction.shape[:-1]:
        raise ValueError(
            f"valid_mask must have shape {prediction.shape[:-1]}, got {valid_mask.shape}"
        )

    mask = valid_mask.to(device=prediction.device, dtype=prediction.dtype)
    squared_error = (prediction - target).square()
    valid_count = mask.sum()
    component_loss = (squared_error * mask.unsqueeze(-1)).sum(dim=(0, 1, 2))
    component_loss = component_loss / valid_count.clamp_min(1.0)
    component_loss = component_loss * (valid_count > 0).to(component_loss.dtype)
    return component_loss.sum(), component_loss


def integrate_flow_ode(
    velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    initial_state: torch.Tensor,
    steps: int,
    solver: str = "heun",
    valid_agent_mask: torch.Tensor = None,
) -> torch.Tensor:
    """Integrate ``dx/dlambda = velocity_fn(x, lambda)`` from zero to one."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    solver = solver.lower()
    if solver not in ("euler", "heun"):
        raise ValueError(f"Unsupported flow solver: {solver}")

    state = initial_state
    state_mask = None
    if valid_agent_mask is not None:
        if valid_agent_mask.shape != initial_state.shape[:2]:
            raise ValueError("valid_agent_mask must have shape [B, P]")
        state_mask = valid_agent_mask.to(initial_state.device, initial_state.dtype)
        state_mask = state_mask.view(initial_state.shape[0], initial_state.shape[1], 1, 1)
        state = state * state_mask

    dt = 1.0 / float(steps)
    batch_size = initial_state.shape[0]
    for step in range(steps):
        time = torch.full(
            (batch_size,), step * dt, device=state.device, dtype=state.dtype
        )
        velocity = velocity_fn(state, time)
        if velocity.shape != state.shape:
            raise ValueError("velocity_fn must return a tensor with the state shape")
        if state_mask is not None:
            velocity = velocity * state_mask

        if solver == "euler":
            state = state + dt * velocity
        else:
            predicted_state = state + dt * velocity
            next_time = torch.full(
                (batch_size,), (step + 1) * dt,
                device=state.device, dtype=state.dtype,
            )
            next_velocity = velocity_fn(predicted_state, next_time)
            if next_velocity.shape != state.shape:
                raise ValueError("velocity_fn must return a tensor with the state shape")
            if state_mask is not None:
                next_velocity = next_velocity * state_mask
            state = state + 0.5 * dt * (velocity + next_velocity)

        if state_mask is not None:
            state = state * state_mask

    return state
