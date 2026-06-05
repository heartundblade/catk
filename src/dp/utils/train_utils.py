import torch
# import random
import numpy as np
# from mmengine import fileio
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
    heading_rotated = rotate_vectors(coords[..., 2:4])
    
    translated = rotated + sdc_states[:, None, None, :2]
    
    if coords.shape[-1] == 4:
        valid_translated = translated[valid_mask]
        valid_heading = heading_rotated[valid_mask]
        transformed[..., 0:2][valid_mask] = valid_translated
        transformed[..., 2:4][valid_mask] = valid_heading
    elif coords.shape[-1] == 6:
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
    action_valid = action_valid.any(dim=-1)
    
    # Filter again
    action = torch.where(action_valid[..., None], action, 0.0)
    
    return action, action_valid


def roll_out(
        current_states: torch.Tensor,
        actions: torch.Tensor,
        dt: float = 0.1,
        action_len: int = 5,
        global_frame: float = True
    ):
        """
        Forward pass of the dynamics model.

        Args:
            current_states (torch.Tensor): Current states tensor of shape [B, N, x, 5]. [x, y, theta, v_x, v_y]
            actions (torch.Tensor): Inputs tensor of shape [B, N, x, T_f//T_a, 2]. [Accel, yaw_rate]
            global_frame (bool): Flag indicating whether to use the global frame of reference. Default is False.

        Returns:
            torch.Tensor: Predicted trajectories.

        """
        x = current_states[..., 0]
        y = current_states[..., 1]
        theta = torch.atan2(current_states[..., 3], current_states[..., 2])
        v_x = current_states[..., 4]
        v_y = current_states[..., 5]
        v = torch.sqrt(v_x**2 + v_y**2)

        a = actions[..., 0].repeat_interleave(action_len, dim=-1) 
        v = v.unsqueeze(-1) + torch.cumsum(a * dt, dim=-1)
        # TODO: this noise may be unnecessary
        v += torch.randn_like(v) * 0.1
        v = torch.clamp(v, min=0)

        yaw_rate = actions[..., 1].repeat_interleave(action_len, dim=-1) 
        yaw_rate += torch.randn_like(yaw_rate) * 0.01

        if global_frame:
            theta = theta.unsqueeze(-1) + torch.cumsum(yaw_rate * dt, dim=-1)
        else:
            theta = torch.cumsum(yaw_rate * dt, dim=2)

        # theta = torch.fmod(theta + torch.pi, 2*torch.pi) - torch.pi
        # theta = wrap_angle(theta)
        
        v_x = v * torch.cos(theta)
        v_y = v * torch.sin(theta)
        
        if global_frame:
            x = x.unsqueeze(-1) + torch.cumsum(v_x * dt, dim=-1)
            y = y.unsqueeze(-1) + torch.cumsum(v_y * dt, dim=-1)
        else:
            x = torch.cumsum(v_x * dt, dim=-1)
            y = torch.cumsum(v_y * dt, dim=-1)

        return torch.stack([x, y, theta, v_x, v_y], dim=-1)

# def opendata(path):
    
#     npz_bytes = fileio.get(path)
#     buff = io.BytesIO(npz_bytes)
#     npz_data = np.load(buff)

#     return npz_data

# def set_seed(CUR_SEED):
#     random.seed(CUR_SEED)
#     np.random.seed(CUR_SEED)
#     torch.manual_seed(CUR_SEED)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# def get_epoch_mean_loss(epoch_loss):
#     epoch_mean_loss = {}
#     for current_loss in epoch_loss:
#         for key, value in current_loss.items():
#             if key in epoch_mean_loss:
#                 epoch_mean_loss[key].append(value if isinstance(value, (int, float)) else value.item())
#             else:
#                 epoch_mean_loss[key] = [value if isinstance(value, (int, float)) else value.item()]


#     for key, values in epoch_mean_loss.items():
#         epoch_mean_loss[key] = np.mean(np.array(values))

#     return epoch_mean_loss

# def save_model(model, optimizer, scheduler, save_path, epoch, train_loss, wandb_id, ema):
#     """
#     save the model to path
#     """
#     save_model = {'epoch': epoch + 1, 
#                   'model': model.state_dict(), 
#                   'ema_state_dict': ema.state_dict(),
#                   'optimizer': optimizer.state_dict(), 
#                   'schedule': scheduler.state_dict(), 
#                   'loss': train_loss,
#                   'wandb_id': wandb_id}

#     with io.BytesIO() as f:
#         torch.save(save_model, f)
#         fileio.put(f.getvalue(), f'{save_path}/model_epoch_{epoch+1}_trainloss_{train_loss:.4f}.pth')
#         fileio.put(f.getvalue(), f"{save_path}/latest.pth")

# def resume_model(path: str, model, optimizer, scheduler, ema, device):
#     """
#     load ckpt from path
#     """
#     path = os.path.join(path, 'latest.pth')
#     ckpt = fileio.get(path)
#     with io.BytesIO(ckpt) as f:
#         ckpt = torch.load(f)

#     # load model
#     try:
#         model.load_state_dict(ckpt['model'])
#     except:
#         model.load_state_dict(ckpt)                   
#     print("Model load done")
    
#     # load optimizer
#     try:
#         optimizer.load_state_dict(ckpt['optimizer'])
#         print("Optimizer load done")
#     except:
#         print("no pretrained optimizer found")
            
#     # load schedule
#     try:
#         scheduler.load_state_dict(ckpt['schedule'])
#         print("Schedule load done")
#     except:
#         print("no schedule found,")
    
#     # load step
#     try:
#         init_epoch = ckpt['epoch']
#         print("Step load done")
#     except:
#         init_epoch = 0

#     # Load wandb id
#     try:
#         wandb_id = ckpt['wandb_id']
#         print("wandb id load done")
#     except:
#         wandb_id = None

#     try:
#         ema.ema.load_state_dict(ckpt['ema_state_dict'])
#         ema.ema.eval()
#         for p in ema.ema.parameters():
#             p.requires_grad_(False)

#         print("ema load done")
#     except:
#         print('no ema shadow found')

#     return model, optimizer, scheduler, init_epoch, wandb_id, ema