# Copyright (c) 2025 Julien Fischer, Chair of Computer Graphics and Visualization, TUD Dresden University of Technology.
# All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

import numpy as np
import torch

def batched_dot(a: torch.Tensor,
                b: torch.Tensor
                ) -> torch.Tensor:
    r"""Batched dot product along dimension 1"""
    return torch.einsum('bs,bs->b', a, b)


def batch_project(a: torch.Tensor,
                  b: torch.Tensor
                  ) -> torch.Tensor:
    r"""Projects vector a onto vector b
    Args:
        a: Shape (B, 3)
        b: Shape (B, 3)
    """
    proj = batched_dot(a,b).reshape(-1, 1) * b
    return proj


def closest_point_on_line_segment(line_start: torch.Tensor,
                                  line_end: torch.Tensor,
                                  p: torch.Tensor
                                  ) -> torch.Tensor:
    r"""For the given point p, finds the closest point on the given line segment
    Args:
        line_start: Start of the line segment. Shape (B, 3)
        line_end: End of the line segment. Shape (B, 3)
        p: Point for which the closest point on the line segment should be
            found. Shape (B, 3)
    Returns:
        The closest point. Shape (B, 3)
    """
    if len(line_start.shape) == 1:
        line_start = line_start.unsqueeze(0)
    if len(line_end.shape) == 1:
        line_end = line_end.unsqueeze(0)
    if len(p.shape) == 1:
        p = p.unsqueeze(0)
    assert line_start.shape[0] == line_end.shape[0] == p.shape[0], f"Expected batch dimension to be equal for all inputs, but got {line_start.shape[0]}, {line_end.shape[0]}, {p.shape[0]}"

    direction = line_end - line_start
    e = torch.nn.functional.normalize(direction, dim=1)
    # vector from line segment start point to p
    start_p = p - line_start
    # Project this vector onto the normalized line segment's direction
    start_p_proj = batch_project(start_p, e)

    # Make sure the closest point stays on the line segment
    # 'Before' start point
    start_p_proj = torch.where(
        (batched_dot(direction, start_p_proj) < 0).reshape(-1, 1),
        torch.zeros(3, dtype=torch.float32),
        start_p_proj
    )
    # 'After' end point
    start_p_proj = torch.where(
        (
            torch.linalg.vector_norm(start_p_proj, ord=2, dim=1) > torch.linalg.vector_norm(direction, ord=2, dim=1)
        ).reshape(-1, 1),
        direction,
        start_p_proj
    )
    x = line_start + start_p_proj
    return x


def batched_rot_mat_to_quat(
    rot_matrices: torch.Tensor, epsilon: float = 1e-7
) -> torch.Tensor:
    """
        Converts the given rotation matrix to quaterion representation
    """
    # batch_size = rot_matrices.shape[0]
    # quaternions = torch.zeros((batch_size, 4), device=rot_matrices.device)

    # trace = torch.einsum("bii->b", rot_matrices)  # trace of each matrix

    # for i in range(batch_size):
    #     r = rot_matrices[i]
    #     t = trace[i]
    #     if t > 0:
    #         s = torch.sqrt(t + 1.0 + epsilon) * 2  # 4w
    #         quaternions[i, 0] = 0.25 * s
    #         quaternions[i, 1] = (r[2, 1] - r[1, 2]) / (s + epsilon)
    #         quaternions[i, 2] = (r[0, 2] - r[2, 0]) / (s + epsilon)
    #         quaternions[i, 3] = (r[1, 0] - r[0, 1]) / (s + epsilon)
    #     else:
    #         diag = torch.diagonal(r)
    #         if diag[0] > diag[1] and diag[0] > diag[2]:
    #             s = torch.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2] + epsilon) * 2
    #             quaternions[i, 0] = (r[2, 1] - r[1, 2]) / (s + epsilon)
    #             quaternions[i, 1] = 0.25 * s
    #             quaternions[i, 2] = (r[0, 1] + r[1, 0]) / (s + epsilon)
    #             quaternions[i, 3] = (r[0, 2] + r[2, 0]) / (s + epsilon)
    #         elif diag[1] > diag[2]:
    #             s = torch.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2] + epsilon) * 2
    #             quaternions[i, 0] = (r[0, 2] - r[2, 0]) / (s + epsilon)
    #             quaternions[i, 1] = (r[0, 1] + r[1, 0]) / (s + epsilon)
    #             quaternions[i, 2] = 0.25 * s
    #             quaternions[i, 3] = (r[1, 2] + r[2, 1]) / (s + epsilon)
    #         else:
    #             s = torch.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1] + epsilon) * 2
    #             quaternions[i, 0] = (r[1, 0] - r[0, 1]) / (s + epsilon)
    #             quaternions[i, 1] = (r[0, 2] + r[2, 0]) / (s + epsilon)
    #             quaternions[i, 2] = (r[1, 2] + r[2, 1]) / (s + epsilon)
    #             quaternions[i, 3] = 0.25 * s

    # return quaternions
    batch_dim = rot_matrices.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        rot_matrices.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = torch.sqrt(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        ).clamp(min=epsilon)
    )

    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    out = quat_candidates[
        torch.nn.functional.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))

    # Ensure right-handed coordinate system (X right, Y up, Z forward)
    #out[..., 1:] *= -1  # Negate x, y, z components to match standard convention

    return out



# def batched_quaternion_to_rot_vec(quaternions: torch.Tensor, eps: float = 1e-6):
#     # angles = 2 * torch.acos(torch.clamp(quaternions[:, 0], -1.0, 1.0))
#     # sin_half_theta = torch.sqrt(1 - quaternions[:, 0] ** 2)

#     # axes = torch.zeros_like(quaternions[:, 1:])
#     # mask = sin_half_theta > 1e-6  # Avoid division by zero
#     # axes[mask] = quaternions[mask, 1:] / sin_half_theta[mask, None]

#     # # Handle zero rotation case (angle == 0)
#     # axes[~mask] = torch.tensor([0.0, 0.0, 0.0], device=quaternions.device)  # Default axis

#     # # Handle pi rotation case (angle == pi)
#     # pi_mask = torch.isclose(angles, torch.tensor(torch.pi, device=angles.device))
#     # if pi_mask.any():
#     #     max_idx = torch.argmax(quaternions[pi_mask, 1:], dim=1)  # Choose dominant axis
#     #     for i, idx in enumerate(max_idx):
#     #         sign = 1.0 if quaternions[pi_mask][i, idx + 1] > 0 else -1.0
#     #         axes[pi_mask][i] = sign * torch.eye(3, device=quaternions.device)[idx]

#     # return axes * angles[:, None]
#     angles = 2 * torch.acos(torch.clamp(quaternions[:, 0], -1.0, 1.0))
#     sin_half_theta = torch.sqrt(1 - quaternions[:, 0] ** 2 + 1e-6)  # Avoid sqrt(0)

#     axes = quaternions[:, 1:] / sin_half_theta.unsqueeze(1)

#     # Handle zero rotation case (angle == 0)
#     zero_mask = angles.abs() < 1e-6
#     axes[zero_mask] = 0  # Zero rotation vector

#     # Handle pi rotation case (angle == pi) using soft weighting
#     pi_mask = torch.isclose(angles, torch.tensor(torch.pi, device=angles.device))
#     if pi_mask.any():
#         dominant_axis = torch.nn.functional.normalize(quaternions[pi_mask, 1:], dim=1)
#         axes[pi_mask] = dominant_axis

#     return axes * angles.unsqueeze(1)
