import torch
# import random
import numpy as np
# import io
# import os
import json

def openjson(path):
    with open(path, 'r', encoding='utf-8') as f:
        value = f.read()
    dict = json.loads(value)
    return dict

def wrap_angle(angle):
    """
    Wrap the angle to [-pi, pi].

    Args:
        angle (torch.Tensor): Angle array.

    Returns:
        torch.Tensor: Wrapped angle.

    """
    return (angle + torch.pi) % (2 * torch.pi) - torch.pi

def transform_coords_to_sdc_frame(coords, sdc_states):
    """
    Batch transform coordinates to the SDC frame of reference.

    Args:
        coords (torch.Tensor): Coordinates array of shape [B, N, T, 4/6]. [x, y, cos(heading), sin(heading), (vx, vy)]
        sdc_states (torch.Tensor): SDC states array of shape [B, 3]. [x, y, theta]

    Returns:
        torch.Tensor: Transformed coordinates in the SDC frame. [B, N, T, 4/6]

    """
    valid_mask = torch.any(coords != 0, axis=-1)
    transformed = coords.clone()
    
    if not torch.any(valid_mask):
        return transformed
    
    cos_theta = torch.cos(sdc_states[..., 2])
    sin_theta = torch.sin(sdc_states[..., 2])
    
    rotation_matrix = torch.stack(
        [
            torch.stack([cos_theta, -sin_theta], dim=-1),
            torch.stack([sin_theta, cos_theta], dim=-1)
        ], dim=-2
    )
    rotation_matrix = rotation_matrix.to(coords.dtype)
    
    def rotate_vectors(vectors):
        """Rotate 2D vectors using rotation matrix with proper broadcasting."""
        return torch.matmul(vectors.unsqueeze(-2), rotation_matrix[:, None, None, :, :]).squeeze(-2)
    
    translated = coords[..., :2] - sdc_states[:, None, None, :2]
    
    rotated = rotate_vectors(translated)
    heading_rotated = rotate_vectors(coords[..., 2:4])
    
    if coords.shape[-1] == 4:
        valid_rotated = rotated[valid_mask]
        valid_heading = heading_rotated[valid_mask]
        transformed[..., 0:2][valid_mask] = valid_rotated
        transformed[..., 2:4][valid_mask] = valid_heading
    elif coords.shape[-1] == 6:
        vel_rotated = rotate_vectors(coords[..., 4:6])
        
        valid_rotated = rotated[valid_mask]
        valid_heading = heading_rotated[valid_mask]
        valid_vel = vel_rotated[valid_mask]
        transformed[..., 0:2][valid_mask] = valid_rotated
        transformed[..., 2:4][valid_mask] = valid_heading
        transformed[..., 4:6][valid_mask] = valid_vel
    else:
        raise ValueError(f"Invalid coordinate shape: {coords.shape}")
    
    return transformed

def transform_coords_to_global_frame(coords, sdc_states):
    """
    Batch transform coordinates from the SDC frame of reference to global frame.
    
    This is the inverse function of transform_coords_to_sdc_frame.

    Args:
        coords (torch.Tensor): Coordinates array of shape [B, N, T, 4/6]. 
            [x, y, cos(heading), sin(heading), (vx, vy)] in SDC frame
        sdc_states (torch.Tensor): SDC states array of shape [B, 3]. [x, y, theta] in global frame

    Returns:
        torch.Tensor: Transformed coordinates in the global frame. [B, N, T, 4/6]

    """
    valid_mask = torch.any(coords != 0, axis=-1)
    transformed = coords.clone()
    
    if not torch.any(valid_mask):
        return transformed
    
    cos_theta = torch.cos(sdc_states[..., 2])
    sin_theta = torch.sin(sdc_states[..., 2])
    
    inv_rotation_matrix = torch.stack(
        [
            torch.stack([cos_theta, sin_theta], dim=-1),
            torch.stack([-sin_theta, cos_theta], dim=-1)
        ], dim=-2
    )
    inv_rotation_matrix = inv_rotation_matrix.to(coords.dtype)
    
    def rotate_vectors(vectors):
        """Rotate 2D vectors using rotation matrix with proper broadcasting."""
        return torch.matmul(vectors.unsqueeze(-2), inv_rotation_matrix[:, None, None, :, :]).squeeze(-2)
    
    rotated = rotate_vectors(coords[..., :2])
    translated = rotated + sdc_states[:, None, None, :2]

    if coords.shape[-1] == 2:
        valid_translated = translated[valid_mask]
        transformed[..., 0:2][valid_mask] = valid_translated
    elif coords.shape[-1] == 4:
        heading_rotated = rotate_vectors(coords[..., 2:4])
        valid_translated = translated[valid_mask]
        valid_heading = heading_rotated[valid_mask]
        transformed[..., 0:2][valid_mask] = valid_translated
        transformed[..., 2:4][valid_mask] = valid_heading
    elif coords.shape[-1] == 6:
        heading_rotated = rotate_vectors(coords[..., 2:4])
        vel_rotated = rotate_vectors(coords[..., 4:6])
        
        valid_translated = translated[valid_mask]
        valid_heading = heading_rotated[valid_mask]
        valid_vel = vel_rotated[valid_mask]
        transformed[..., 0:2][valid_mask] = valid_translated
        transformed[..., 2:4][valid_mask] = valid_heading
        transformed[..., 4:6][valid_mask] = valid_vel
    else:
        raise ValueError(f"Invalid coordinate shape: {coords.shape}")
    
    return transformed

def batch_transform_maps_to_local_frame(maps):
    """
    Batch transform map elements to local frame. Uses first point as reference.

    Args:
        maps (torch.Tensor): Map tensor of shape [B, N, P, D] with D >= 4 (x, y, cos, sin, ...).

    Returns:
        torch.Tensor: Transformed maps in local frame.
    """
    x = maps[..., 0]
    y = maps[..., 1]
    cos = maps[..., 2]
    sin = maps[..., 3]

    ref_idx = maps.shape[2] // 2
    # ref_idx = 0
    ref_cos = cos[:, :, ref_idx]
    ref_sin = sin[:, :, ref_idx]
    ref_x = x[:, :, ref_idx]
    ref_y = y[:, :, ref_idx]

    dx = x - ref_x[:, :, None]
    dy = y - ref_y[:, :, None]

    local_x = dx * ref_cos[:, :, None] + dy * ref_sin[:, :, None]
    local_y = -dx * ref_sin[:, :, None] + dy * ref_cos[:, :, None]

    local_cos = cos * ref_cos[:, :, None] + sin * ref_sin[:, :, None]
    local_sin = -cos * ref_sin[:, :, None] + sin * ref_cos[:, :, None]

    local_maps = torch.stack([local_x, local_y, local_cos, local_sin], dim=-1)
    local_maps[(maps[..., :4] == 0).all(dim=-1)] = 0

    if maps.shape[-1] > 4:
        maps = torch.cat([local_maps, maps[..., 4:]], dim=-1)
    else:
        maps = local_maps

    return maps


def batch_transform_trajs_to_local_frame(trajs, ref_idx=-1):
    """
    Batch transform trajectories to the local frame of reference.
    Supports DP format: [x, y, cos, sin, vx, vy, ...] (D>=6) or [x, y, cos, sin] (D=4).

    Args:
        trajs (torch.Tensor): Trajectories tensor of shape [B, N, T, D].
        ref_idx (int): Reference index for the local frame. Default is -1.

    Returns:
        torch.Tensor: Transformed trajectories in the local frame.
    """
    D = trajs.shape[-1]
    x = trajs[..., 0]
    y = trajs[..., 1]
    cos = trajs[..., 2]
    sin = trajs[..., 3]

    ref_cos = cos[:, :, ref_idx]
    ref_sin = sin[:, :, ref_idx]
    ref_x = x[:, :, ref_idx]
    ref_y = y[:, :, ref_idx]

    dx = x - ref_x[:, :, None]
    dy = y - ref_y[:, :, None]

    local_x = dx * ref_cos[:, :, None] + dy * ref_sin[:, :, None]
    local_y = -dx * ref_sin[:, :, None] + dy * ref_cos[:, :, None]

    local_cos = cos * ref_cos[:, :, None] + sin * ref_sin[:, :, None]
    local_sin = -cos * ref_sin[:, :, None] + sin * ref_cos[:, :, None]

    if D == 4:
        local_trajs = torch.stack([local_x, local_y, local_cos, local_sin], dim=-1)
        local_trajs[(trajs[..., :4] == 0).all(dim=-1)] = 0
        return local_trajs

    v_x = trajs[..., 4]
    v_y = trajs[..., 5]

    local_v_x = v_x * ref_cos[:, :, None] + v_y * ref_sin[:, :, None]
    local_v_y = -v_x * ref_sin[:, :, None] + v_y * ref_cos[:, :, None]

    local_trajs = torch.stack([local_x, local_y, local_cos, local_sin, local_v_x, local_v_y], dim=-1)
    local_trajs[(trajs[..., :6] == 0).all(dim=-1)] = 0

    if D > 6:
        trajs = torch.cat([local_trajs, trajs[..., 6:]], dim=-1)
    else:
        trajs = local_trajs

    return trajs


def batch_transform_trajs_to_global_frame(trajs, ref_states):
    """
    Batch transform trajectories from local frame to global frame.
    Inverse of batch_transform_trajs_to_local_frame.

    Args:
        trajs (torch.Tensor): Local trajectories tensor of shape [B, N, T, 4].
            [x, y, cos(heading), sin(heading)] in local frame.
        ref_states (torch.Tensor): Reference states tensor of shape [B, N, 4].
            [x, y, cos(heading), sin(heading)] in global frame.

    Returns:
        torch.Tensor: Transformed trajectories in global frame. [B, N, T, 4]
    """
    ref_x = ref_states[..., 0]
    ref_y = ref_states[..., 1]
    ref_cos = ref_states[..., 2]
    ref_sin = ref_states[..., 3]

    local_x = trajs[..., 0]
    local_y = trajs[..., 1]
    local_cos = trajs[..., 2]
    local_sin = trajs[..., 3]

    global_x = ref_x[..., None] + local_x * ref_cos[..., None] - local_y * ref_sin[..., None]
    global_y = ref_y[..., None] + local_x * ref_sin[..., None] + local_y * ref_cos[..., None]
    global_cos = ref_cos[..., None] * local_cos - ref_sin[..., None] * local_sin
    global_sin = ref_sin[..., None] * local_cos + ref_cos[..., None] * local_sin

    global_trajs = torch.stack([global_x, global_y, global_cos, global_sin], dim=-1)
    return global_trajs


@torch.no_grad()
def batch_calculate_relations(
    agents_history: torch.Tensor,
    static_maps: torch.Tensor,
    lanes: torch.Tensor,
    roadlines: torch.Tensor,
    traffic_lights: torch.Tensor = None,
    encoding_mask: torch.Tensor = None,
    device: str = 'cpu'
):
    """
    Calculate relations between agents, lanes, roadlines, static maps, and traffic lights.

    Args:
        agents_history (torch.Tensor): Agent history tensor of shape [B, N, T, D]. [x, y, cos(theta), sin(theta), ...]
        lanes (torch.Tensor): Lanes tensor of shape [B, N, T, D]. [x, y, cos(theta), sin(theta), ...]
        roadlines (torch.Tensor): Roadlines tensor of shape [B, N, T, D]. [x, y, cos(theta), sin(theta), ...]
        static_maps (torch.Tensor): Static maps tensor of shape [B, N, T, D]. [x, y, cos(theta), sin(theta), ...]
        traffic_lights (torch.Tensor): Traffic lights tensor of shape [B, TL, 3]. [x, y, state]. Default is None.
        device (str): Device to use. Default is 'cpu'.

    Returns:
        torch.Tensor: Relations tensor of shape [B, N_total, N_total, 4]. [local_pos_x, local_pos_y, cos_theta_diff, sin_theta_diff]
    """
    batch_size = agents_history.shape[0]
    n_agents = agents_history.shape[1]
    n_lanes = lanes.shape[1]
    n_roadlines = roadlines.shape[1]
    n_static_maps = static_maps.shape[1]
    n_traffic_lights = traffic_lights.shape[1] if traffic_lights is not None else 0
    n = n_agents + n_lanes + n_roadlines + n_static_maps + n_traffic_lights

    # Store (x, y, cos, sin) directly — avoid atan2→cos/sin round-trip
    agents_elem = agents_history[:, :, -1, :4]

    lanes_elem = lanes[:, :, lanes.shape[2] // 2, :4]
    roadlines_elem = roadlines[:, :, roadlines.shape[2] // 2, :4]
    static_maps_elem = static_maps[:, :, static_maps.shape[2] // 2, :4]
    # lanes_elem = lanes[:, :, 0, :4]
    # roadlines_elem = roadlines[:, :, 0, :4]
    # static_maps_elem = static_maps[:, :, 0, :4]

    # Traffic lights: (x, y, cos=1, sin=0) — no orientation
    if traffic_lights is not None:
        tl_xy = traffic_lights[..., :2]
        tl_cos = torch.ones_like(tl_xy[..., 0])
        tl_sin = torch.zeros_like(tl_xy[..., 0])
        traffic_lights_elem = torch.stack([tl_xy[..., 0], tl_xy[..., 1], tl_cos, tl_sin], dim=-1)
    else:
        traffic_lights_elem = None

    # Concatenate all elements
    elem_list = [agents_elem, static_maps_elem, lanes_elem, roadlines_elem]
    if traffic_lights_elem is not None:
        elem_list.append(traffic_lights_elem)
    all_elements = torch.cat(elem_list, dim=1)

    # Compute pairwise differences using broadcasting
    pos_diff = all_elements[:, :, :2][:, :, None, :] - all_elements[:, :, :2][:, None, :, :]

    # Compute local position using stored cos/sin (no trig calls)
    cos_i = all_elements[:, :, 2][:, :, None]  # [B, N, 1]
    sin_i = all_elements[:, :, 3][:, :, None]  # [B, N, 1]
    local_pos_x = pos_diff[..., 0] * cos_i + pos_diff[..., 1] * sin_i
    local_pos_y = -pos_diff[..., 0] * sin_i + pos_diff[..., 1] * cos_i

    # Compute cos/sin of angle differences using trig identities (no atan2/wrap_angle/cos/sin)
    cos_j = all_elements[:, :, 2][:, None, :]  # [B, 1, N]
    sin_j = all_elements[:, :, 3][:, None, :]  # [B, 1, N]
    cos_theta_diff = cos_i * cos_j + sin_i * sin_j
    sin_theta_diff = sin_i * cos_j - cos_i * sin_j

    # Set theta_diff to zero for traffic light pairs (traffic lights have no orientation)
    start_idx = n_agents + n_lanes + n_roadlines + n_static_maps
    tl_mask = (torch.arange(n, device=device) >= start_idx).repeat(batch_size, 1)
    tl_pair_mask = tl_mask[:, :, None] | tl_mask[:, None, :]
    cos_theta_diff = torch.where(tl_pair_mask, 1.0, cos_theta_diff)
    sin_theta_diff = torch.where(tl_pair_mask, 0.0, sin_theta_diff)

    # Set the diagonal of the differences to a very small value
    diag_mask = torch.eye(n, dtype=bool, device=device)
    epsilon = 1e-5
    local_pos_x = torch.where(diag_mask, epsilon, local_pos_x)
    local_pos_y = torch.where(diag_mask, epsilon, local_pos_y)
    cos_theta_diff = torch.where(diag_mask, epsilon, cos_theta_diff)
    sin_theta_diff = torch.where(diag_mask, epsilon, sin_theta_diff)

    # Zero out relations for masked tokens (using real mask, not x==0 heuristic)
    if encoding_mask is not None:
        zero_mask = torch.logical_or(encoding_mask.unsqueeze(2), encoding_mask.unsqueeze(1))
    else:
        zero_mask = torch.logical_or(all_elements[:, :, 0][:, :, None] == 0, all_elements[:, :, 0][:, None, :] == 0)

    relations = torch.stack([local_pos_x, local_pos_y, cos_theta_diff, sin_theta_diff], dim=-1)

    # Apply zero mask
    relations = torch.where(zero_mask[..., None], 0.0, relations)

    return relations, all_elements

def inverse_kinematics(
    agents_future: torch.Tensor,
    agents_future_valid: torch.Tensor,
    dt: float = 0.1,
    action_len: int = 2,
):
    """
    Perform inverse kinematics to compute actions.

    Args:
        agents_future (torch.Tensor): Future agent positions tensor. 
            [B, A, T, 9] # x, y, cos(heading), sin(heading), velx, vely, length, width, z
        agents_future_valid (torch.Tensor): Future agent validity tensor. [B, A, T]
        dt (float): Time interval. Default is 0.1.
        action_len (int): Length of each action. Default is 2.

    Returns:
        torch.Tensor: Predicted actions.

    """
    # Inverse kinematics implementation goes here
    batch_size, num_agents, num_timesteps, _ = agents_future.shape
    assert (num_timesteps-1) % action_len == 0, "future_len must be divisible by action_len"
    num_actions = (num_timesteps-1) // action_len
    
    # Compute yaw from cos and sin (agents_future[..., 2] = cos_yaw, agents_future[..., 3] = sin_yaw)
    yaw = torch.atan2(agents_future[..., 3], agents_future[..., 2])
    speed = torch.norm(agents_future[..., 4:6], dim=-1)
    
    yaw_rate = wrap_angle(torch.diff(yaw, dim=-1)) / dt
    accel = torch.diff(speed, dim=-1) / dt
    action_valid = agents_future_valid[..., :1] & agents_future_valid[..., 1:]

    # Mask out annotation errors (outliers)
    action_valid = action_valid & (yaw_rate.abs() < 1.0)   # rad/s, ~29°/s
    action_valid = action_valid & (accel.abs() < 8.0)       # m/s², ~0.5g
    
    # filter out invalid actions
    yaw_rate = torch.where(action_valid, yaw_rate, 0.0)
    accel = torch.where(action_valid, accel, 0.0)
    
    # Reshape for mean pooling
    yaw_rate = yaw_rate.reshape(batch_size, num_agents, num_actions, -1)
    accel = accel.reshape(batch_size, num_agents, num_actions, -1)
    action_valid = action_valid.reshape(batch_size, num_agents, num_actions, -1)
    
    yaw_rate_sample = yaw_rate.sum(dim=-1) / torch.clamp(action_valid.sum(dim=-1), min=1.0)
    accel_sample = accel.sum(dim=-1) / torch.clamp(action_valid.sum(dim=-1), min=1.0)
    action = torch.stack([accel_sample, yaw_rate_sample], dim=-1)
    # Account for invalid actions of 2 step
    action_valid = action_valid.any(dim=-1)
    
    # Filter again
    action = torch.where(action_valid[..., None], action, 0.0)
    
    return action, action_valid


def roll_out(
        current_states: torch.Tensor,
        actions: torch.Tensor,
        dt: float = 0.1,
        action_len: int = 5,
        global_frame: bool = True,
        training: bool = False,
        valid_mask: torch.Tensor = None,
    ):
        """
        Forward pass of the dynamics model.

        Args:
            current_states (torch.Tensor): Current states tensor of shape [B, N, 6]. [x, y, cos_theta, sin_theta, v_x, v_y]
            actions (torch.Tensor): Inputs tensor of shape [B, N, num_actions, 2]. [Accel, yaw_rate]
            global_frame (bool): Flag indicating whether to use the global frame of reference. Default is True.
            valid_mask (torch.Tensor): Optional mask of shape [B, N] indicating valid agents. Invalid agents' trajectories will be zeroed out.

        Returns:
            torch.Tensor: Predicted trajectories of shape [B, N, T, 6].

        """
        x = current_states[..., 0]
        y = current_states[..., 1]
        theta = torch.atan2(current_states[..., 3], current_states[..., 2])
        v_x = current_states[..., 4]
        v_y = current_states[..., 5]
        v = torch.sqrt(v_x**2 + v_y**2)

        a = actions[..., 0].repeat_interleave(action_len, dim=-1) 
        v = v.unsqueeze(-1) + torch.cumsum(a * dt, dim=-1)
        # if training:
        #     v += torch.randn_like(v) * 0.1
        # v = torch.clamp(v, min=0)

        yaw_rate = actions[..., 1].repeat_interleave(action_len, dim=-1) 
        # if training:
        #     yaw_rate += torch.randn_like(yaw_rate) * 0.01

        # if global_frame:
        theta = theta.unsqueeze(-1) + torch.cumsum(yaw_rate * dt, dim=-1)
        # else:
        #     theta = torch.cumsum(yaw_rate * dt, dim=-1)

        # theta = torch.fmod(theta + torch.pi, 2*torch.pi) - torch.pi
        # theta = wrap_angle(theta)
        
        v_x = v * torch.cos(theta)
        v_y = v * torch.sin(theta)
        
        # if global_frame:
        x = x.unsqueeze(-1) + torch.cumsum(v_x * dt, dim=-1)
        y = y.unsqueeze(-1) + torch.cumsum(v_y * dt, dim=-1)
        # else:
        #     x = torch.cumsum(v_x * dt, dim=-1)
        #     y = torch.cumsum(v_y * dt, dim=-1)

        # Output format: [x, y, cos(theta), sin(theta), v_x, v_y]
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        trajectories = torch.stack([x, y, cos_theta, sin_theta, v_x, v_y], dim=-1)  # [B, N, T, 6]

        if valid_mask is not None:
            trajectories = trajectories * valid_mask[..., None, None].float()

        return trajectories

def inverse_kinematics_bicycle_old(
    agents_future: torch.Tensor,
    agents_future_valid: torch.Tensor,
    agent_type: torch.Tensor,
    dt: float = 0.1,
    action_len: int = 2,
):
    """
    Inverse kinematics using rear-axle bicycle model (no sideslip) for VEH/CYC,
    and point-mass model for PED.

    Agent types: 0=VEH, 1=CYC (nonholonomic, bicycle model), 2=PED (point-mass).

    For nonholonomic agents, center displacement is decomposed via chord formula:
        Δp_nh = Δp_prog + Δp_swing
        Δp_prog  = a∥ · sinc(aψ/2) · u∥(ψ̄)    # no-slip point forward (chord)
        Δp_swing = 2r · sin(aψ/2) · u⊥(ψ̄)    # center swing due to rotation

    The no-slip offset r is estimated per-agent from turning segments:
        r = (Δp_c·u⊥) / (2·sin(aψ/2)).  For PED agents, r = 0.

    Actions are inferred sequentially, ensuring consistency with roll_out_bicycle.

    Args:
        agents_future: [B, A, T, 9] (x, y, cos, sin, vx, vy, length, width, z)
        agents_future_valid: [B, A, T]
        agent_type: [B, A] int tensor (0=VEH, 1=CYC, 2=PED)
        dt: time step interval. Default is 0.1.
        action_len: number of timesteps per action. Default is 2.

    Returns:
        action: [B, A, num_actions, 2] (accel, yaw_rate)
        action_valid: [B, A, num_actions]
        r: [B, A] estimated no-slip-offset (0 for PED)
    """
    B, A, T, D = agents_future.shape
    assert (T - 1) % action_len == 0, "future_len must be divisible by action_len"
    num_actions = (T - 1) // action_len

    is_nonholonomic = (agent_type < 2)                                               # [B, A] bool: VEH=0, CYC=1

    cos_psi = agents_future[..., 2]                                                 # [B, A, T]
    sin_psi = agents_future[..., 3]
    vx = agents_future[..., 4]
    vy = agents_future[..., 5]
    x_c = agents_future[..., 0]
    y_c = agents_future[..., 1]
    psi = torch.atan2(sin_psi, cos_psi)                                             # [B, A, T]

    # ---- 1. Estimate r for nonholonomic agents; r=0 for PED ----
    dx_c = torch.diff(x_c, dim=-1)                                                  # [B, A, T-1]
    dy_c = torch.diff(y_c, dim=-1)
    a_psi = wrap_angle(torch.diff(psi, dim=-1))                                     # [B, A, T-1]
    psi_mean = psi[..., :-1] + a_psi / 2                                            # [B, A, T-1]

    u_perp_x = -torch.sin(psi_mean)                                                 # u⊥ = (-sin(ψ̄), cos(ψ̄))
    u_perp_y = torch.cos(psi_mean)
    swing_proj = dx_c * u_perp_x + dy_c * u_perp_y                                  # Δp_c · u⊥  [B, A, T-1]

    half_a_psi = a_psi / 2
    sin_half = torch.sin(half_a_psi)

    eps = 0.05
    r_obs = swing_proj / (2.0 * sin_half + 1e-8)                                    # [B, A, T-1]
    r_valid = agents_future_valid[..., :-1] & agents_future_valid[..., 1:] \
              & (half_a_psi.abs() > eps) & is_nonholonomic.unsqueeze(-1)            # only turning + nonholonomic

    r_obs = torch.where(r_valid, r_obs, torch.tensor(float('inf'), device=r_obs.device))
    r_obs_sorted = torch.sort(r_obs, dim=-1).values
    r_valid_count = r_valid.sum(dim=-1)                                             # [B, A]
    r_median_idx = (r_valid_count // 2).clamp(min=0, max=T - 2)
    r = torch.gather(r_obs_sorted, -1, r_median_idx.unsqueeze(-1)).squeeze(-1)     # [B, A]
    r = torch.clamp(r, 0.5, 6.0)

    r_no_obs = r_valid_count == 0
    r = torch.where(r_no_obs & is_nonholonomic, torch.tensor(2.0, device=r.device, dtype=r.dtype), r)
    r = torch.where(~is_nonholonomic, torch.tensor(0.0, device=r.device, dtype=r.dtype), r)

    # ---- 2. Convert GT center trajectory to no-slip point ----
    x_r = x_c - r.unsqueeze(-1) * cos_psi                                           # [B, A, T]
    y_r = y_c - r.unsqueeze(-1) * sin_psi
    # Nonholonomic: longitudinal speed; PED: point-mass speed (v along heading)
    v_r_nh = vx * cos_psi + vy * sin_psi
    v_r_ped = torch.sqrt(vx ** 2 + vy ** 2 + 1e-8)
    v_r = torch.where(is_nonholonomic.unsqueeze(-1), v_r_nh, v_r_ped)               # [B, A, T]

    # ---- 3. Sequential inverse kinematics ----
    actions_list = []
    action_valid_list = []

    x_r_cur = x_r[..., 0]                                                           # [B, A]
    y_r_cur = y_r[..., 0]
    psi_cur = psi[..., 0]
    v_r_cur = v_r[..., 0]

    for k in range(num_actions):
        t_start = k * action_len
        t_end = t_start + action_len
        T_seg = action_len * dt

        x_r_tgt = x_r[..., t_end]
        y_r_tgt = y_r[..., t_end]
        psi_tgt = psi[..., t_end]
        v_r_tgt = v_r[..., t_end]

        a_psi_seg = wrap_angle(psi_tgt - psi_cur)                                    # total yaw change
        omega = a_psi_seg / T_seg
        psi_mean_seg = (psi_cur + psi_tgt) / 2

        dx_r = x_r_tgt - x_r_cur
        dy_r = y_r_tgt - y_r_cur
        a_parallel = dx_r * torch.cos(psi_mean_seg) + dy_r * torch.sin(psi_mean_seg)  # Δp_r · u∥

        half = a_psi_seg / 2
        sinc = torch.where(half.abs() < 1e-6, torch.ones_like(half), torch.sin(half) / half)
        a_parallel = a_parallel / (sinc + 1e-8)                                      # arc length a∥

        accel = 2.0 * (a_parallel - v_r_cur * T_seg) / (T_seg ** 2 + 1e-8)

        valid = agents_future_valid[..., t_start] & agents_future_valid[..., t_end]
        valid = valid & (omega.abs() < 1.0) & (accel.abs() < 8.0)

        omega = torch.where(valid, omega, torch.tensor(0.0, device=omega.device))
        accel = torch.where(valid, accel, torch.tensor(0.0, device=accel.device))

        actions_list.append(torch.stack([accel, omega], dim=-1))                     # [B, A, 2]
        action_valid_list.append(valid)

        # Forward simulate no-slip point using chord formula (matches roll_out_bicycle)
        v_r_new = v_r_cur + accel * T_seg
        a_parallel_fwd = (v_r_cur + v_r_new) / 2.0 * T_seg
        sinc_fwd = torch.where(half.abs() < 1e-6, torch.ones_like(half), torch.sin(half) / half)

        x_r_cur = x_r_cur + a_parallel_fwd * sinc_fwd * torch.cos(psi_mean_seg)
        y_r_cur = y_r_cur + a_parallel_fwd * sinc_fwd * torch.sin(psi_mean_seg)
        psi_cur = psi_cur + a_psi_seg
        v_r_cur = v_r_new

    action = torch.stack(actions_list, dim=-2)                                       # [B, A, num_actions, 2]
    action_valid = torch.stack(action_valid_list, dim=-1)                            # [B, A, num_actions]

    return action, action_valid, r

def roll_out_bicycle_old(
    current_states: torch.Tensor,
    actions: torch.Tensor,
    r: torch.Tensor,
    agent_type: torch.Tensor,
    dt: float = 0.1,
    action_len: int = 5,
    valid_mask: torch.Tensor = None,
):
    """
    Forward pass: bicycle model for VEH/CYC, point-mass model for PED.

    Agent types: 0=VEH, 1=CYC (nonholonomic, bicycle model), 2=PED (point-mass).

    For nonholonomic agents, center displacement per step:
        Δp_nh = a∥·sinc(aψ/2)·u∥(ψ̄) + 2r·sin(aψ/2)·u⊥(ψ̄)

    For PED agents (r=0), the swing term vanishes and the chord formula
    reduces to the point-mass displacement.

    The integration scheme matches inverse_kinematics_bicycle exactly.

    Args:
        current_states: [B, N, 6] (x, y, cos_psi, sin_psi, vx, vy) at center
        actions: [B, N, num_actions, 2] (accel, yaw_rate)
        r: [B, N] no-slip-offset (0 for PED, >0 for VEH/CYC)
        agent_type: [B, N] int tensor (0=VEH, 1=CYC, 2=PED)
        dt: time step interval. Default is 0.1.
        action_len: number of timesteps per action. Default is 5.
        valid_mask: [B, N] optional validity mask

    Returns:
        trajectories: [B, N, T, 6] (x, y, cos_psi, sin_psi, vx, vy) at center
    """
    B, N, _ = current_states.shape
    num_actions = actions.shape[2]

    is_nonholonomic = (agent_type < 2)                                               # [B, N] bool: VEH=0, CYC=1

    x_c = current_states[..., 0]
    y_c = current_states[..., 1]
    cos_psi_0 = current_states[..., 2]
    sin_psi_0 = current_states[..., 3]
    vx_c = current_states[..., 4]
    vy_c = current_states[..., 5]

    psi_0 = torch.atan2(sin_psi_0, cos_psi_0)

    r_eff = torch.where(is_nonholonomic, r, torch.tensor(0.0, device=r.device, dtype=r.dtype))

    x_r = x_c - r_eff * cos_psi_0
    y_r = y_c - r_eff * sin_psi_0
    v_r_nh = vx_c * cos_psi_0 + vy_c * sin_psi_0
    v_r_ped = torch.sqrt(vx_c ** 2 + vy_c ** 2 + 1e-8)
    v_r = torch.where(is_nonholonomic, v_r_nh, v_r_ped)

    # Expand actions to per-step
    a = actions[..., 0].repeat_interleave(action_len, dim=-1)                        # [B, N, S]
    omega = actions[..., 1].repeat_interleave(action_len, dim=-1)

    num_steps = num_actions * action_len

    x_r_list = [x_r]
    y_r_list = [y_r]
    psi_list = [psi_0]
    v_r_list = [v_r]

    for t in range(num_steps):
        a_t = a[..., t]                                                              # [B, N]
        omega_t = omega[..., t]

        v_r_prev = v_r
        psi_prev = psi

        # No-slip point dynamics (constant a, ω over dt)
        v_r = v_r + a_t * dt
        psi = psi + omega_t * dt

        a_psi_step = omega_t * dt
        psi_mean = psi_prev + a_psi_step / 2
        a_parallel = (v_r_prev + v_r) / 2.0 * dt                                     # trapezoidal arc length

        half = a_psi_step / 2
        sinc = torch.where(half.abs() < 1e-6, torch.ones_like(half), torch.sin(half) / half)

        x_r = x_r + a_parallel * sinc * torch.cos(psi_mean)
        y_r = y_r + a_parallel * sinc * torch.sin(psi_mean)

        x_r_list.append(x_r)
        y_r_list.append(y_r)
        psi_list.append(psi)
        v_r_list.append(v_r)

    x_r_stack = torch.stack(x_r_list, dim=-1)                                        # [B, N, T]
    y_r_stack = torch.stack(y_r_list, dim=-1)
    psi_stack = torch.stack(psi_list, dim=-1)
    v_r_stack = torch.stack(v_r_list, dim=-1)

    cos_psi_stack = torch.cos(psi_stack)
    sin_psi_stack = torch.sin(psi_stack)

    # Convert back to center
    x_c_out = x_r_stack + r_eff.unsqueeze(-1) * cos_psi_stack
    y_c_out = y_r_stack + r_eff.unsqueeze(-1) * sin_psi_stack

    # Center velocity: v_c = v_r·u∥(ψ) + r·ω·u⊥(ψ)
    omega_steps = actions[..., 1].repeat_interleave(action_len, dim=-1)              # [B, N, S]
    omega_full = torch.cat([omega_steps[..., :1], omega_steps], dim=-1)              # [B, N, T]

    vx_c_out = v_r_stack * cos_psi_stack - r_eff.unsqueeze(-1) * omega_full * sin_psi_stack
    vy_c_out = v_r_stack * sin_psi_stack + r_eff.unsqueeze(-1) * omega_full * cos_psi_stack

    trajectories = torch.stack([x_c_out, y_c_out, cos_psi_stack, sin_psi_stack, vx_c_out, vy_c_out], dim=-1)

    if valid_mask is not None:
        trajectories = trajectories * valid_mask[..., None, None].float()

    return trajectories

def inverse_kinematics_bicycle(
    agents_future: torch.Tensor,
    agents_future_valid: torch.Tensor,
    agent_type: torch.Tensor,
    dt: float = 0.1,
    turn_eps: float = 1e-3,
    reexec_err_tol: float = 0.3,
    rho_fixed: dict = None,
):
    """
    Inverse kinematics matching the bicycle model in the paper (Eqs. 1-3, 10-14).

    One metric kinematic action a_t = (a_parallel, a_lateral, a_psi) is recovered
    for every simulation step:

      - PED (holonomic, WOMD type 2):       Δp = R(ψ)·(a∥, a⊥)ᵀ            (Eq. 1)
      - VEH/CYC (non-holonomic, WOMD 1/3):  ψ̄ = ψ + aψ/2,
            Δp = a∥·sinc(aψ/2)·u∥(ψ̄) + 2r·sin(aψ/2)·u⊥(ψ̄)               (Eq. 2)
        a_lateral stays in the shared action space but is NOT executed (Eq. 11).

    The no-slip offset uses r_i = ρ_c·ℓ_i, with one ρ_c per non-holonomic type.
    ρ̂_i is computed per turning interval from Eq. (12):
        ρ̂_i = Δp_iᵀ·u⊥(ψ̄_i) / (2·ℓ_i·sin(aψ,i/2))
    and aggregated with a robust statistic (median) per type (VEH, CYC).

    Actions are recovered sequentially from the previously executed pose s̃
    (Eq. 13) and re-executed with the same transition (Eq. 14). Targets whose
    re-execution error exceeds `reexec_err_tol` are filtered out (invalid).

    Args:
        agents_future: [B, A, T, 9] (x, y, cos, sin, vx, vy, length, width, z);
                       frame 0 is the current logged pose.
        agents_future_valid: [B, A, T]
        agent_type: [B, A] WOMD object type (1=VEH, 2=PED, 3=CYC; 0/UNSET is
                     treated as holonomic).
        dt: interval duration in seconds (kept for interface compatibility; the
            paper's actions are per-step displacements and do not depend on it).
        turn_eps: min |aψ| (rad) for an interval to inform ρ. Straight intervals
                  (aψ≈0) carry no information about r (Eq. 3).
        reexec_err_tol: max center re-execution error (m) to keep a target.

    Returns:
        action: [B, A, T-1, 3] per-step (a_parallel, a_lateral, a_psi)
        action_valid: [B, A, T-1]
        r: [B, A] no-slip offset in meters (0 for PED/UNSET)
    """
    B, A, T, _ = agents_future.shape
    assert T >= 2, "need at least the current pose plus one future pose"

    x_c = agents_future[..., 0]
    y_c = agents_future[..., 1]
    cos_psi = agents_future[..., 2]
    sin_psi = agents_future[..., 3]
    length = agents_future[..., 6]                                                   # box length ℓ
    psi = torch.atan2(sin_psi, cos_psi)

    is_veh = agent_type == 1                                                         # WOMD: VEH
    is_cyc = agent_type == 3                                                         # WOMD: CYC
    is_nonhol = is_veh | is_cyc                                                      # bicycle model

    # ---- 1. One ρ per non-holonomic type (Eq. 12 + robust median) ----
    rho_default = torch.tensor(0.25, device=x_c.device, dtype=x_c.dtype)

    if rho_fixed is not None:
        rho_by_type = {
            1: torch.tensor(rho_fixed.get(1, 0.25), device=x_c.device, dtype=x_c.dtype),
            3: torch.tensor(rho_fixed.get(3, 0.25), device=x_c.device, dtype=x_c.dtype),
        }
    else:
        dx_c = torch.diff(x_c, dim=-1)                                               # [B, A, T-1]
        dy_c = torch.diff(y_c, dim=-1)
        a_psi = wrap_angle(torch.diff(psi, dim=-1))
        psi_bar = psi[..., :-1] + a_psi / 2

        swing_proj = dx_c * (-torch.sin(psi_bar)) + dy_c * torch.cos(psi_bar)        # Δp_c·u⊥(ψ̄)
        sin_half = torch.sin(a_psi / 2)
        rho_obs = swing_proj / (2.0 * length[..., :-1] * sin_half + 1e-8)           # per-interval ρ̂

        interval_ok = agents_future_valid[..., :-1] & agents_future_valid[..., 1:]
        rho_valid = interval_ok & is_nonhol.unsqueeze(-1) & (a_psi.abs() > turn_eps)

        rho_obs = torch.where(rho_valid, rho_obs, torch.full_like(rho_obs, float('inf')))
        rho_flat = rho_obs.reshape(-1)
        type_flat = agent_type.unsqueeze(-1).expand_as(a_psi).reshape(-1)
        valid_flat = rho_valid.reshape(-1)

        rho_by_type = {}
        for t in (1, 3):
            sel = valid_flat & (type_flat == t)
            if bool(sel.any()):
                rho_by_type[t] = torch.median(rho_flat[sel])
            else:
                rho_by_type[t] = rho_default

    length0 = length[..., 0]
    r = torch.zeros_like(length0)
    r = torch.where(is_veh, rho_by_type[1] * length0, r)
    r = torch.where(is_cyc, rho_by_type[3] * length0, r)                             # r=0 for PED/UNSET

    # ---- 2. Sequential transition-consistent recovery (Eqs. 13-14) ----
    # executed pose starts from the logged current pose
    x_cur = x_c[..., 0].clone()
    y_cur = y_c[..., 0].clone()
    psi_cur = psi[..., 0].clone()

    action_list = []
    valid_list = []

    for k in range(T - 1):
        nxt = k + 1
        x_tgt = x_c[..., nxt]
        y_tgt = y_c[..., nxt]
        psi_tgt = psi[..., nxt]

        frame_ok = agents_future_valid[..., k] & agents_future_valid[..., nxt]
        a_psi_step = wrap_angle(psi_tgt - psi_cur)
        half = a_psi_step / 2
        sinc_val = torch.where(half.abs() < 1e-6,
                               torch.ones_like(half),
                               torch.sin(half) / half)
        psi_bar = psi_cur + half
        cos_bar = torch.cos(psi_bar)
        sin_bar = torch.sin(psi_bar)

        cos_cur = torch.cos(psi_cur)
        sin_cur = torch.sin(psi_cur)
        cos_tgt = torch.cos(psi_tgt)
        sin_tgt = torch.sin(psi_tgt)

        # non-holonomic inverse: project no-slip-point displacement onto u∥(ψ̄)
        dx_r = (x_tgt - r * cos_tgt) - (x_cur - r * cos_cur)
        dy_r = (y_tgt - r * sin_tgt) - (y_cur - r * sin_cur)
        chord = dx_r * cos_bar + dy_r * sin_bar
        a_par_nh = chord / sinc_val                                                  # arc length a∥

        # holonomic inverse (Eq. 1): Δp = R(ψ)·(a∥, a⊥)ᵀ
        dx_hol_obs = x_tgt - x_cur
        dy_hol_obs = y_tgt - y_cur
        a_par_hol = dx_hol_obs * cos_cur + dy_hol_obs * sin_cur
        a_lat_hol = -dx_hol_obs * sin_cur + dy_hol_obs * cos_cur

        a_par = torch.where(is_nonhol, a_par_nh, a_par_hol)
        a_lat = torch.where(is_nonhol, torch.zeros_like(a_par_hol), a_lat_hol)

        # re-execute with the same transition as roll_out_bicycle (Eqs. 10-11)
        dx_nh = (a_par * sinc_val * cos_bar
                 + 2.0 * r * torch.sin(half) * (-sin_bar))
        dy_nh = (a_par * sinc_val * sin_bar
                 + 2.0 * r * torch.sin(half) * cos_bar)
        dx_hol = a_par_hol * cos_cur - a_lat_hol * sin_cur
        dy_hol = a_par_hol * sin_cur + a_lat_hol * cos_cur
        dx_exec = torch.where(is_nonhol, dx_nh, dx_hol)
        dy_exec = torch.where(is_nonhol, dy_nh, dy_hol)

        x_exec = x_cur + dx_exec
        y_exec = y_cur + dy_exec
        psi_exec = wrap_angle(psi_cur + a_psi_step)

        reexec_err = torch.sqrt((x_exec - x_tgt) ** 2 + (y_exec - y_tgt) ** 2)
        keep = frame_ok & (reexec_err <= reexec_err_tol)
        keep = keep & torch.isfinite(a_par) & torch.isfinite(a_lat)
        keep = keep & torch.isfinite(a_psi_step)
        keep = keep & (a_psi_step.abs() < 0.5)   # rad/step, ~5.0 rad/s equiv
        keep = keep & (a_par.abs() < 5.0)         # m/step, ~50 m/s equiv
        keep = keep & (a_lat.abs() < 5.0)

        action_list.append(torch.stack([
            torch.where(keep, a_par, torch.zeros_like(a_par)),
            torch.where(keep, a_lat, torch.zeros_like(a_lat)),
            torch.where(keep, a_psi_step, torch.zeros_like(a_psi_step)),
        ], dim=-1))                                                                  # [B, A, 3]
        valid_list.append(keep)

        # continue from the executed pose on kept targets; otherwise reset to the
        # logged pose so a filtered/bad interval does not pollute later targets
        x_cur = torch.where(keep, x_exec, x_tgt)
        y_cur = torch.where(keep, y_exec, y_tgt)
        psi_cur = torch.where(keep, psi_exec, psi_tgt)

    action = torch.stack(action_list, dim=-2)                                        # [B, A, T-1, 3]
    action_valid = torch.stack(valid_list, dim=-1)                                   # [B, A, T-1]

    return action, action_valid, r

def roll_out_bicycle(
    current_states: torch.Tensor,
    actions: torch.Tensor,
    r: torch.Tensor,
    agent_type: torch.Tensor,
    dt: float = 0.1,
    valid_mask: torch.Tensor = None,
):
    """
    Paper-compatible forward transition (Eqs. 1-3, 10-11).

    Executes one metric kinematic action per simulation step:
        ψ_{t+1} = wrap(ψ_t + a_psi)
        p_{t+1} = p_t + Δp
    with, for PED (holonomic, WOMD type 2),
        Δp = R(ψ_t)·(a∥, a⊥)ᵀ                                                     (Eq. 1)
    and for VEH/CYC (bicycle, WOMD types 1/3), with ψ̄ = ψ_t + aψ/2,
        Δp = a∥·sinc(aψ/2)·u∥(ψ̄) + 2r·sin(aψ/2)·u⊥(ψ̄)                           (Eq. 2)
    a_lateral is shared but NOT executed for VEH/CYC (Eq. 11).

    Args:
        current_states: [B, N, 6] center state (x, y, cos_psi, sin_psi, vx, vy)
        actions: [B, N, S, 3] per-step actions (a_parallel, a_lateral, a_psi)
        r: [B, N] no-slip offset in meters (0 for PED/UNSET)
        agent_type: [B, N] WOMD object type (1=VEH, 2=PED, 3=CYC; 0/UNSET is
                     treated as holonomic)
        dt: interval duration in seconds (used only for the velocity channels)
        valid_mask: [B, N] optional validity mask

    Returns:
        trajectories: [B, N, S, 6] (x, y, cos_psi, sin_psi, vx, vy) at the center
                      after each executed step
    """
    B, N, _ = current_states.shape
    S = actions.shape[2]

    x = current_states[..., 0].clone()
    y = current_states[..., 1].clone()
    psi = torch.atan2(current_states[..., 3], current_states[..., 2])

    is_veh = agent_type == 1                                                         # WOMD: VEH
    is_cyc = agent_type == 3                                                         # WOMD: CYC
    is_nonhol = is_veh | is_cyc                                                      # bicycle model
    r_eff = torch.where(is_nonhol, r, torch.zeros_like(r))

    a_par = actions[..., 0]                                                          # [B, N, S]
    a_lat = actions[..., 1]
    a_psi = actions[..., 2]

    # ---- Pre-compute all psi values via cumsum (eliminates sequential loop) ----
    psi_seq = psi.unsqueeze(-1) + torch.cumsum(a_psi, dim=-1)                       # [B, N, S]
    psi_all = torch.cat([psi.unsqueeze(-1), psi_seq], dim=-1)                       # [B, N, S+1], psi at start of each step
    psi_bar = psi_all[..., :-1] + a_psi / 2                                         # [B, N, S], psi_bar at each step

    half = a_psi / 2
    sinc_val = torch.where(half.abs() < 1e-6,
                           torch.ones_like(half),
                           torch.sin(half) / half)
    sin_half = torch.sin(half)

    cos_bar = torch.cos(psi_bar)                                                     # [B, N, S]
    sin_bar = torch.sin(psi_bar)
    cos_cur = torch.cos(psi_all[..., :-1])                                           # [B, N, S]
    sin_cur = torch.sin(psi_all[..., :-1])

    # ---- Vectorized displacements for all steps ----
    # non-holonomic (bicycle) center displacement, Eq. (2)
    dx_nh = a_par * sinc_val * cos_bar - 2.0 * r_eff.unsqueeze(-1) * sin_half * sin_bar
    dy_nh = a_par * sinc_val * sin_bar + 2.0 * r_eff.unsqueeze(-1) * sin_half * cos_bar
    # holonomic displacement, Eq. (1)
    dx_hol = a_par * cos_cur - a_lat * sin_cur
    dy_hol = a_par * sin_cur + a_lat * cos_cur

    dx = torch.where(is_nonhol.unsqueeze(-1), dx_nh, dx_hol)                        # [B, N, S]
    dy = torch.where(is_nonhol.unsqueeze(-1), dy_nh, dy_hol)

    # ---- Accumulate positions via cumsum ----
    x_stack = torch.cat([x.unsqueeze(-1), x.unsqueeze(-1) + torch.cumsum(dx, dim=-1)], dim=-1)
    y_stack = torch.cat([y.unsqueeze(-1), y.unsqueeze(-1) + torch.cumsum(dy, dim=-1)], dim=-1)
    psi_stack = psi_all

    # velocity channels derived from the executed center displacement
    vx = (x_stack[..., 1:] - x_stack[..., :-1]) / dt
    vy = (y_stack[..., 1:] - y_stack[..., :-1]) / dt
    x_out = x_stack[..., 1:]
    y_out = y_stack[..., 1:]
    cos_out = torch.cos(psi_stack[..., 1:])
    sin_out = torch.sin(psi_stack[..., 1:])

    trajectories = torch.stack([x_out, y_out, cos_out, sin_out, vx, vy], dim=-1)

    if valid_mask is not None:
        trajectories = trajectories * valid_mask[..., None, None].float()

    return trajectories