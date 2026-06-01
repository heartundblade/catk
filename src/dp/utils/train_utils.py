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
            torch.stack([cos_theta, sin_theta], dim=-1),
            torch.stack([-sin_theta, cos_theta], dim=-1)
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