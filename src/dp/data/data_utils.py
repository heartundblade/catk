import torch
import numpy as np

def data_collate_fn(batch_list):
    """
    Collects a batch of data from a list of transitions.

    Args:
        batch_list (List): a list of transitions.

    Returns:
        Dict[str, torch.Tensor]: a batch of data.
    """
    list_len = len(batch_list)
    key_to_list = {}
    for key in batch_list[0].keys():
        key_to_list[key] = [batch_list[i][key] for i in range(list_len)]
    
    # input_batch = {}
    # for key, value in key_to_list.items():
    #     if 'scenario' not in key:
    #         input_batch[key] = torch.from_numpy(np.stack(value, axis=0))
    #     else:
    #         input_batch[key] = value
    
    special_keys = {
        'scenario_id', 
        'tfrecord_path', 
        'gt_scenario', 
        'agents_history_remaining', 
        'agents_id_remaining'
    }

    input_batch = {}
    for key, value in key_to_list.items():
        if key in special_keys:
            input_batch[key] = value
        else:
            input_batch[key] = torch.stack(value, axis=0)
    
    return input_batch
