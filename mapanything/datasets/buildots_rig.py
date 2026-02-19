# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Dataset adapter that wraps BuildotsDataset output into the MapAnything
view-dict list format for 4-camera equirectangular rig pose estimation.
"""

from typing import Dict, List

import torch
from torch.utils.data import Dataset

from mapanything.utils.geometry import (
    get_rays_in_camera_frame,
    rotation_matrix_to_quaternion,
)


class BuildotsRigAdapter(Dataset):
    """
    Wraps an existing BuildotsDataset and reshapes its flat-tensor output into
    a list of per-view dictionaries that MapAnything's forward() expects.

    Each sample produces ``num_timestamps * 4`` views (4 cameras per timestamp).
    The first 4 views (t=0) act as anchors: their GT poses are provided as
    model *input* and are excluded from the pose loss.
    """

    def __init__(
        self,
        buildots_dataset: Dataset,
        num_timestamps: int = 3,
        data_norm_type: str = "imagenet",
    ):
        """
        Args:
            buildots_dataset: Instance of BuildotsDataset (from research.positioning_net).
            num_timestamps: How many consecutive timestamps to include per sample.
                            Total views = num_timestamps * 4.
            data_norm_type: Image normalization type expected by the encoder.
        """
        self.dataset = buildots_dataset
        self.num_timestamps = num_timestamps
        self.data_norm_type = data_norm_type

        # Pre-compute ray directions from the fixed intrinsics (shared across all cameras)
        intrinsics = self.dataset.intrinsic  # (3, 3)
        h, w = self.dataset.target_size
        _, ray_dirs = get_rays_in_camera_frame(
            intrinsics, h, w, normalize_to_unit_sphere=True
        )
        self.ray_directions_cam = ray_dirs  # (H, W, 3)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> List[Dict[str, torch.Tensor]]:
        data = self.dataset[idx]

        num_views = self.num_timestamps * 4
        views = []

        for i in range(num_views):
            is_anchor = i < 4  # first timestamp = anchor

            ext = data["normed_extrinsic"][i]  # (4, 4) w2c
            yaw = data["yaw"][i]  # scalar
            c2w = torch.inverse(ext)
            gt_trans = c2w[:3, 3]  # (3,)
            gt_yaw_vec = torch.stack(
                [torch.cos(yaw), torch.sin(yaw)]
            )  # (2,)

            view: Dict[str, torch.Tensor] = {
                "img": data["images"][i],  # (3, H, W)
                "data_norm_type": self.data_norm_type,
                "ray_directions_cam": self.ray_directions_cam.clone(),  # (H, W, 3)
                "rig_rotation": data["rig_rotation"][i],  # (4,)
                "timestamp": data["timestamps"][i:i + 1] / 1000.0,  # (1,) in [0, 1]
                "camera_pose_trans_gt": gt_trans,  # (3,)
                "camera_pose_yaw_gt": gt_yaw_vec,  # (2,)
                "pose_loss_mask": torch.tensor(not is_anchor),
                "is_metric_scale": torch.tensor(True),
                "camera_id": data["camera_id"][i],  # scalar
            }

            # For t=0 anchors, provide GT pose as model input
            if is_anchor:
                quat = rotation_matrix_to_quaternion(c2w[:3, :3])
                view["camera_pose_quats"] = quat  # (4,)
                view["camera_pose_trans"] = gt_trans  # (3,)

            views.append(view)

        return views


def buildots_rig_collate_fn(
    batch: List[List[Dict[str, torch.Tensor]]],
) -> List[Dict[str, torch.Tensor]]:
    """
    Custom collate that stacks per-view tensors across the batch dimension.

    Input:  list of B samples, each a list of V view-dicts.
    Output: list of V view-dicts where each tensor has a leading batch dim B.
    """
    batch_size = len(batch)
    num_views = len(batch[0])

    collated: List[Dict[str, torch.Tensor]] = []
    for v in range(num_views):
        view_dict: Dict[str, torch.Tensor] = {}
        keys = batch[0][v].keys()
        for key in keys:
            vals = [batch[b][v][key] for b in range(batch_size)]
            if isinstance(vals[0], torch.Tensor):
                view_dict[key] = torch.stack(vals, dim=0)
            elif isinstance(vals[0], str):
                view_dict[key] = vals
            else:
                view_dict[key] = vals
        collated.append(view_dict)

    return collated
