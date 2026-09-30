import torch
import torch.nn.functional as F
import numpy as np

_MAX_TRAFFIC_LIGHTS = 20

_PACKED_MAP_KEYS = {
    'map_geometry', 'map_type', 'map_light_type',
    'map_stop_point', 'map_has_stop_point',
    'map_speed_limit', 'map_has_speed_limit', 'map_parent_id',
    'map_segment_index',
}


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
    Concatenates variable-length map tokens and records their scenario indices.

    Args:
        batch_list (List): a list of transitions.

    Returns:
        Dict[str, torch.Tensor]: a batch of data.
    """
    special_keys = {
        'scenario_id', 
        'tfrecord_path', 
        'gt_scenario', 
        'agents_history_remaining', 
        'agents_id_remaining'
    }

    # Traffic lights are not map tokens and retain the existing bounded tensor.
    for sample in batch_list:
        if 'traffic_light_points' in sample:
            sample['traffic_light_points'] = _pad_or_truncate(
                sample['traffic_light_points'], _MAX_TRAFFIC_LIGHTS
            )

    key_to_list = {}
    for key in batch_list[0].keys():
        key_to_list[key] = [batch_list[i][key] for i in range(len(batch_list))]

    input_batch = {}
    for key, value in key_to_list.items():
        if key in _PACKED_MAP_KEYS:
            input_batch[key] = torch.cat(value, dim=0)
        elif key in special_keys:
            input_batch[key] = value
        else:
            input_batch[key] = torch.stack(value, axis=0)

    counts = torch.tensor(
        [sample['map_geometry'].shape[0] for sample in batch_list], dtype=torch.long
    )
    input_batch['map_batch'] = torch.repeat_interleave(
        torch.arange(len(batch_list), dtype=torch.long), counts
    )
    input_batch['map_ptr'] = torch.cat([
        torch.zeros(1, dtype=torch.long), counts.cumsum(dim=0)
    ])
    return input_batch
