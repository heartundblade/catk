import torch
import torch.nn.functional as F
import numpy as np

# Global max limits for map elements (should match config values)
_MAX_LANES = 256
_MAX_ROADLINES = 128
_MAX_STATIC_MAPS = 20
_MAX_TRAFFIC_LIGHTS = 20


def _pad_or_truncate(tensor, target_size, pad_value=0.0):
    """Pad or truncate a tensor along dim 0 to target_size.
    Padding is applied on the right side, keeping valid data on the left."""
    n = tensor.shape[0]
    if n > target_size:
        return tensor[:target_size]
    elif n < target_size:
        pad = [0] * (tensor.ndim * 2)
        pad[-1] = target_size - n
        return F.pad(tensor, pad, value=pad_value)
    return tensor


def data_collate_fn(batch_list):
    """
    Collects a batch of data from a list of transitions.
    Handles variable-length map elements by padding/truncating to fixed max sizes.

    Args:
        batch_list (List): a list of transitions.

    Returns:
        Dict[str, torch.Tensor]: a batch of data.
    """
    # Keys that need per-batch padding/truncation
    map_keys_2d = {
        'traffic_light_points': _MAX_TRAFFIC_LIGHTS,
        'lanes_valid': _MAX_LANES,
        'lanes_speed_limit': _MAX_LANES,
        'lanes_has_speed_limit': _MAX_LANES,
        'lanes_stop_point': _MAX_LANES,
        'roadlines_valid': _MAX_ROADLINES,
        'static_maps_valid': _MAX_STATIC_MAPS,
    }
    map_keys_3d = {
        'lanes': _MAX_LANES,
        'roadlines': _MAX_ROADLINES,
        'static_maps': _MAX_STATIC_MAPS,
    }

    special_keys = {
        'scenario_id', 
        'tfrecord_path', 
        'gt_scenario', 
        'agents_history_remaining', 
        'agents_id_remaining'
    }

    # Apply pad/truncate to each sample's map elements
    for sample in batch_list:
        for key, max_size in map_keys_2d.items():
            if key in sample:
                sample[key] = _pad_or_truncate(sample[key], max_size)
        for key, max_size in map_keys_3d.items():
            if key in sample:
                sample[key] = _pad_or_truncate(sample[key], max_size)

    key_to_list = {}
    for key in batch_list[0].keys():
        key_to_list[key] = [batch_list[i][key] for i in range(len(batch_list))]

    input_batch = {}
    for key, value in key_to_list.items():
        if key in special_keys:
            input_batch[key] = value
        else:
            input_batch[key] = torch.stack(value, axis=0)
    
    return input_batch