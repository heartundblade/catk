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
    local_maps[maps[..., :4] == 0] = 0

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
        local_trajs[trajs[..., :4] == 0] = 0
        return local_trajs

    v_x = trajs[..., 4]
    v_y = trajs[..., 5]

    local_v_x = v_x * ref_cos[:, :, None] + v_y * ref_sin[:, :, None]
    local_v_y = -v_x * ref_sin[:, :, None] + v_y * ref_cos[:, :, None]

    local_trajs = torch.stack([local_x, local_y, local_cos, local_sin, local_v_x, local_v_y], dim=-1)
    local_trajs[trajs[..., :6] == 0] = 0

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