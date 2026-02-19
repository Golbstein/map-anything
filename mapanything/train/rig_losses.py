# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Losses for 4-camera equirectangular rig pose estimation.
Predicts xyz + yaw(cos,sin) per view with rig consistency regularizers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CameraRigConsistencyLoss(nn.Module):
    """
    Enforces that all 4 cameras in the rig agree on the yaw direction after
    de-rotating each by its known rig offset.

    Input: yaw vectors of shape (B, S, 4, 2) where dim-2 order is
           [Front, Left, Right, Back] and dim-3 is (cos, sin).
    """

    def forward(self, vector: torch.Tensor) -> torch.Tensor:
        v_front = vector[..., 0, :]
        v_left = vector[..., 1, :]
        v_right = vector[..., 2, :]
        v_back = vector[..., 3, :]

        # De-rotate to align with Front frame
        aligned_front = v_front
        # Left is at +90 deg: rotate -90 -> (y, -x)
        aligned_left = torch.stack([v_left[..., 1], -v_left[..., 0]], dim=-1)
        # Right is at -90 deg: rotate +90 -> (-y, x)
        aligned_right = torch.stack([-v_right[..., 1], v_right[..., 0]], dim=-1)
        # Back is at 180 deg: rotate 180 -> (-x, -y)
        aligned_back = -v_back

        all_aligned = torch.stack(
            [aligned_front, aligned_left, aligned_right, aligned_back], dim=-2
        )
        consensus = all_aligned.mean(dim=-2, keepdim=True)
        similarity = F.cosine_similarity(all_aligned, consensus, dim=-1)
        return 1.0 - similarity.mean()


class Rig360PoseLoss(nn.Module):
    """
    Combined loss for rig pose estimation:
      - Translation L2 loss (supervised, t > 0 only)
      - Yaw cosine loss (supervised, t > 0 only)
      - Rig translation variance regularizer (all timestamps)
      - Rig yaw consistency regularizer (all timestamps)
    """

    def __init__(
        self,
        trans_weight: float = 1.0,
        yaw_weight: float = 1.0,
        rig_trans_var_weight: float = 0.1,
        rig_yaw_consistency_weight: float = 0.1,
        num_cameras: int = 4,
    ):
        super().__init__()
        self.trans_weight = trans_weight
        self.yaw_weight = yaw_weight
        self.rig_trans_var_weight = rig_trans_var_weight
        self.rig_yaw_consistency_weight = rig_yaw_consistency_weight
        self.num_cameras = num_cameras
        self.rig_yaw_loss = CameraRigConsistencyLoss()

    def forward(self, preds, batch):
        """
        Args:
            preds: list of N dicts, each with:
                "cam_trans" (B, 3), "cam_yaw" (B, 2)
            batch: list of N dicts, each with:
                "camera_pose_trans_gt" (B, 3), "camera_pose_yaw_gt" (B, 2),
                "pose_loss_mask" (B,) bool
        Returns:
            total_loss: scalar
            details: dict of individual loss terms
        """
        num_views = len(preds)
        device = preds[0]["cam_trans"].device

        # Collect all predictions and GT
        all_pred_trans = torch.cat([p["cam_trans"] for p in preds], dim=0)
        all_pred_yaw = torch.cat([p["cam_yaw"] for p in preds], dim=0)
        all_gt_trans = torch.cat(
            [b["camera_pose_trans_gt"].to(device) for b in batch], dim=0
        )
        all_gt_yaw = torch.cat(
            [b["camera_pose_yaw_gt"].to(device) for b in batch], dim=0
        )
        all_mask = torch.cat(
            [b["pose_loss_mask"].to(device) for b in batch], dim=0
        )

        # ---- Supervised losses (only where pose_loss_mask is True) ----
        details = {}

        if all_mask.any():
            trans_loss = F.mse_loss(
                all_pred_trans[all_mask], all_gt_trans[all_mask]
            )
            yaw_loss = (
                1.0
                - F.cosine_similarity(
                    all_pred_yaw[all_mask], all_gt_yaw[all_mask], dim=-1
                ).mean()
            )
        else:
            trans_loss = torch.tensor(0.0, device=device)
            yaw_loss = torch.tensor(0.0, device=device)

        details["trans_loss"] = trans_loss.item()
        details["yaw_loss"] = yaw_loss.item()

        # ---- Rig consistency regularizers ----
        batch_size = preds[0]["cam_trans"].shape[0]
        num_timestamps = num_views // self.num_cameras

        # Reshape to (B, T, C, ...) where C = num_cameras
        pred_trans_rig = all_pred_trans.view(
            num_views, batch_size, 3
        ).permute(1, 0, 2)  # (B, V, 3)
        pred_trans_rig = pred_trans_rig.view(
            batch_size, num_timestamps, self.num_cameras, 3
        )

        pred_yaw_rig = all_pred_yaw.view(
            num_views, batch_size, 2
        ).permute(1, 0, 2)  # (B, V, 2)
        pred_yaw_rig = pred_yaw_rig.view(
            batch_size, num_timestamps, self.num_cameras, 2
        )

        # Translation variance: cameras at same timestamp should predict same xyz
        trans_var = pred_trans_rig.var(dim=2, unbiased=False).mean()
        details["rig_trans_var"] = trans_var.item()

        # Yaw consistency via CameraRigConsistencyLoss
        yaw_consistency = self.rig_yaw_loss(pred_yaw_rig)
        details["rig_yaw_consistency"] = yaw_consistency.item()

        total_loss = (
            self.trans_weight * trans_loss
            + self.yaw_weight * yaw_loss
            + self.rig_trans_var_weight * trans_var
            + self.rig_yaw_consistency_weight * yaw_consistency
        )
        details["total_loss"] = total_loss.item()

        return total_loss, details
