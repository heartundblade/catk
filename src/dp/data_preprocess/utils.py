import numpy as np
import torch

# =====================
# 1. Ego, agent, static coordination transformation
# =====================
def _local_to_local_transforms(global_states1, global_states2):
    """
    Converts the global_states1' local coordinates to global_states2's local coordinates.
    """

    local_xform = _state_se2_array_to_transform_matrix(global_states2)
    local_xform_inv = np.linalg.inv(local_xform)

    transforms = _state_se2_array_to_transform_matrix_batch(global_states1)

    transforms = np.matmul(local_xform_inv, transforms)

    return transforms

def _state_se2_array_to_transform_matrix(input_data):


    x: float = float(input_data[0])
    y: float = float(input_data[1])
    h: float = float(input_data[2])

    cosine = np.cos(h)
    sine = np.sin(h)

    return np.array(
        [[cosine, -sine, x], [sine, cosine, y], [0.0, 0.0, 1.0]]
    )

def _state_se2_array_to_transform_matrix_batch(input_data):

    # Transform the incoming coordinates so transformation can be done with a simple matrix multiply.
    #
    # [x1, y1, phi1]  => [x1, y1, cos1, sin1, 1]
    # [x2, y2, phi2]     [x2, y2, cos2, sin2, 1]
    # ...          ...
    # [xn, yn, phiN]     [xn, yn, cosN, sinN, 1]
    processed_input = np.column_stack(
        (
            input_data[:, 0],
            input_data[:, 1],
            np.cos(input_data[:, 2]),
            np.sin(input_data[:, 2]),
            np.ones_like(input_data[:, 0]),
        )
    )

    # See below for reshaping example
    reshaping_array = np.array(
        [
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, -1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]
    )
    # Builds the transform matrix
    # First computes the components of each transform as rows of a Nx9 array, and then reshapes to a Nx3x3 array
    # Below is outlined how the Nx9 representation looks like (s1 and c1 are cos1 and sin1)
    # [x1, y1, c1, s1, 1]  => [c1, -s1, x1, s1, c1, y1, 0, 0, 1]  =>  [[c1, -s1, x1], [s1, c1, y1], [0, 0, 1]]
    # [x2, y2, c2, s2, 1]     [c2, -s2, x2, s2, c2, y2, 0, 0, 1]  =>  [[c2, -s2, x2], [s2, c2, y2], [0, 0, 1]]
    # ...          ...
    # [xn, yn, cN, sN, 1]     [cN, -sN, xN, sN, cN, yN, 0, 0, 1]
    return (processed_input @ reshaping_array).reshape(-1, 3, 3)

def _transform_matrix_to_state_se2_array_batch(input_data):
    """
    Converts a Nx3x3 batch transformation matrix into a Nx3 array of [x, y, heading] rows.
    :param input_data: The 3x3 transformation matrix.
    :return: The converted array.
    """

    # Picks the entries, the third column will be overwritten with the headings [x, y, _]
    first_columns = input_data[:, :, 0].reshape(-1, 3)
    angles = np.arctan2(first_columns[:, 1], first_columns[:, 0])

    result = input_data[:, :, 2]
    result[:, 2] = angles

    return result

def _global_state_se2_array_to_local(
    global_states, local_state
):
    """
    Transforms the StateSE2 in array from to the frame of reference in local_frame.

    :param global_states: A array of Nx3, where the columns are [x, y, heading].
    :param local_state: A array of [x, y, h] of the frame to which to transform.
    :return: The transformed coordinates.
    """

    local_xform = _state_se2_array_to_transform_matrix(local_state)
    local_xform_inv = np.linalg.inv(local_xform)

    transforms = _state_se2_array_to_transform_matrix_batch(global_states)

    transforms = np.matmul(local_xform_inv, transforms)

    output = _transform_matrix_to_state_se2_array_batch(transforms)

    return output

def _global_velocity_to_local(velocity, anchor_heading):
    velocity_x = velocity[:, 0] * np.cos(anchor_heading) + velocity[:, 1] * np.sin(anchor_heading)
    velocity_y = velocity[:, 1] * np.cos(anchor_heading) - velocity[:, 0] * np.sin(anchor_heading)

    return np.stack([velocity_x, velocity_y], axis=-1)

# =====================
# 2. Map coordination transformation
# =====================
def coordinates_to_local_frame(
    coords, anchor_state, precision = None
):
    """
    Transform a set of [x, y] coordinates without heading to the the given frame.
    :param coords: <np.array: num_coords, 2> Coordinates to be transformed, in the form [x, y].
    :param anchor_state: The coordinate frame to transform to, in the form [x, y, heading].
    :param precision: The precision with which to allocate the intermediate array. If None, then it will be inferred from the input precisions.
    :return: <np.array: num_coords, 2> Transformed coordinates.
    """
    if len(coords.shape) != 2 or coords.shape[1] != 2:
        raise ValueError(f"Unexpected coords shape: {coords.shape}")

    if precision is None:
        if coords.dtype != anchor_state.dtype:
            raise ValueError("Mixed datatypes provided to coordinates_to_local_frame without precision specifier.")
        precision = coords.dtype

    # torch.nn.functional.pad will crash with 0-length inputs.
    # In that case, there are no coordinates to transform.
    if coords.shape[0] == 0:
        return coords

    # Extract transform
    transform = _state_se2_array_to_transform_matrix(anchor_state)
    transform = np.linalg.inv(transform)

    # Transform the incoming coordinates to homogeneous coordinates
    #  So translation can be done with a simple matrix multiply.
    #
    # [x1, y1]  => [x1, y1, 1]
    # [x2, y2]     [x2, y2, 1]
    # ...          ...
    # [xn, yn]     [xn, yn, 1]
    coords = np.pad(coords, pad_width=((0, 0), (0, 1)), mode='constant', constant_values=1.0)

    # Perform the transformation, transposing so the shapes match
    coords = np.matmul(transform, coords.T)

    # Transform back from homogeneous coordinates to standard coordinates.
    #   Get rid of the scaling dimension and transpose so output shape matches input shape.
    result = coords.T
    result = result[:, :2]

    return result


def vector_set_coordinates_to_local_frame(
    coords,
    avails,
    anchor_state,
    output_precision = np.float32,
):
    """
    Transform the vector set map element coordinates from global frame to ego vehicle frame, as specified by
        anchor_state.
    :param coords: Coordinates to transform. <np.array: num_elements, num_points, 2>.
    :param avails: Availabilities mask identifying real vs zero-padded data in coords.
        <np.array: num_elements, num_points>.
    :param anchor_state: The coordinate frame to transform to, in the form [x, y, heading].
    :param output_precision: The precision with which to allocate output array.
    :return: Transformed coordinates.
    :raise ValueError: If coordinates dimensions are not valid or don't match availabilities.
    """


    # Flatten coords from (num_map_elements, num_points_per_element, 2) to
    #   (num_map_elements * num_points_per_element, 2) for easier processing.
    num_map_elements, num_points_per_element, _ = coords.shape
    coords = coords.reshape(num_map_elements * num_points_per_element, 2)

    # Apply transformation using adequate precision
    coords = coordinates_to_local_frame(coords, anchor_state, precision=np.float64)

    # Reshape to original dimensionality
    coords = coords.reshape(num_map_elements, num_points_per_element, 2)

    # Output with specified precision
    coords = coords.astype(output_precision)

    # ignore zero-padded data
    coords[~avails] = 0.0

    return coords


# =====================
# 3. Numpy-Tensor transformation
# =====================
def convert_to_model_inputs(data, device):
    tensor_data = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray) and v.dtype == np.bool_:
            tensor_data[k] = torch.tensor(v, dtype=torch.bool).unsqueeze(0).to(device)
        else:
            tensor_data[k] = torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(device)

    return tensor_data

def wrap_angle(angle):
    """
    Wrap the angle to [-pi, pi].

    Args:
        angle (torch.Tensor): Angle tensor.

    Returns:
        torch.Tensor: Wrapped angle.

    """
    # return torch.atan2(torch.sin(angle), torch.cos(angle))
    return (angle + torch.pi) % (2 * torch.pi) - torch.pi

def wrap_angle_np(angle):
    """
    Wrap the angle to [-pi, pi] (numpy version).

    Args:
        angle (np.ndarray): Angle array.

    Returns:
        np.ndarray: Wrapped angle.

    """
    return (angle + np.pi) % (2 * np.pi) - np.pi

def batch_transform_polylines_to_local_frame(polylines):
    """
    Batch transform polylines to the local frame of reference.

    Args:
        polylines (torch.Tensor): Polylines tensor of shape [B, M, W, 5].

    Returns:
        torch.Tensor: Transformed polylines in the local frame.

    """
    x = polylines[..., 0]
    y = polylines[..., 1]
    theta = polylines[..., 2]
    
    local_x = (x - x[:, :, 0, None]) * torch.cos(theta[:, :, 0, None]) + \
        (y - y[:, :, 0, None]) * torch.sin(theta[:, :, 0, None])
    local_y = -(x - x[:, :, 0, None]) * torch.sin(theta[:, :, 0, None]) + \
        (y - y[:, :, 0, None]) * torch.cos(theta[:, :, 0, None])
    
    local_theta = theta - theta[:, :, 0, None]
    local_theta = wrap_angle(local_theta)

    local_polylines = torch.stack([local_x, local_y, local_theta], dim=-1)
    local_polylines[polylines[..., :3] == 0] = 0
    polylines = torch.cat([local_polylines, polylines[..., 3:]], dim=-1)

    return polylines

def batch_transform_trajs_to_local_frame(trajs, ref_idx=-1):
    """
    Batch transform trajectories to the local frame of reference.

    Args:
        trajs (torch.Tensor): Trajectories tensor of shape [B, N, T, x].
        ref_idx (int): Reference index for the local frame. Default is -1.

    Returns:
        torch.Tensor: Transformed trajectories in the local frame.

    """
    x = trajs[..., 0]
    y = trajs[..., 1]
    theta = trajs[..., 2]
    v_x = trajs[..., 3]
    v_y = trajs[..., 4]
    
    local_x = (x - x[:, :, ref_idx, None]) * torch.cos(theta[:, :, ref_idx, None]) + \
        (y - y[:, :, ref_idx, None]) * torch.sin(theta[:, :, ref_idx, None])
    local_y = -(x - x[:, :, ref_idx, None]) * torch.sin(theta[:, :, ref_idx, None]) + \
        (y - y[:, :, ref_idx, None]) * torch.cos(theta[:, :, ref_idx, None])
    
    local_theta = theta - theta[:, :, ref_idx, None]
    local_theta = wrap_angle(local_theta)

    local_v_x = v_x * torch.cos(theta[:, :, ref_idx, None]) + v_y * torch.sin(theta[:, :, ref_idx, None])
    local_v_y = -v_x * torch.sin(theta[:, :, ref_idx, None]) + v_y * torch.cos(theta[:, :, ref_idx, None])

    local_trajs = torch.stack([local_x, local_y, local_theta, local_v_x, local_v_y], dim=-1)
    local_trajs[trajs[..., :5] == 0] = 0

    if trajs.shape[-1] > 5:
        trajs = torch.cat([local_trajs, trajs[..., 5:]], dim=-1)
    else:
        trajs = local_trajs

    return trajs

def transform_polylines_to_sdc_frame(polylines, sdc_states):
    """
    Batch transform polylines to the SDC frame of reference.

    Args:
        polylines (np.ndarray): Polylines array of shape [N, x, 2]. [x, y]
        sdc_states (np.ndarray): SDC states array of shape [3]. [x, y, theta]

    Returns:
        np.ndarray: Transformed polylines in the SDC frame. [N, x, 2]

    """
    x_sdc, y_sdc, theta_sdc = sdc_states
    
    # First translate to SDC origin, then rotate
    g_x = polylines[..., 0] - x_sdc
    g_y = polylines[..., 1] - y_sdc
    
    cos_theta = np.cos(theta_sdc)
    sin_theta = np.sin(theta_sdc)
    
    x = g_x * cos_theta + g_y * sin_theta
    y = -g_x * sin_theta + g_y * cos_theta
    
    # # Convert sin(heading), cos(heading) to theta
    # sin_heading = polylines[..., 2]
    # cos_heading = polylines[..., 3]
    # theta = np.arctan2(sin_heading, cos_heading)
    
    # # Transform heading to SDC frame
    # theta_sdc_frame = theta - theta_sdc
    # theta_sdc_frame = wrap_angle_np(theta_sdc_frame)
    
    # # Convert back to sin, cos
    # sin_heading_sdc = np.sin(theta_sdc_frame)
    # cos_heading_sdc = np.cos(theta_sdc_frame)
    
    # transformed = np.stack([x, y, sin_heading_sdc, cos_heading_sdc], axis=-1)
    
    transformed = np.stack([x, y], axis=-1)
    return transformed
