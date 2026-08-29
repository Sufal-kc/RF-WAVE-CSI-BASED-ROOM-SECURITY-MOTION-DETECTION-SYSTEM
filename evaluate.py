#!/usr/bin/env python3
"""
Evaluate a trained CSIPoseNet checkpoint on a labeled window set.

Metrics:
  - MPJPE: mean per-joint position error, in normalized (0-1) image
    coordinates, averaged only over joints with confidence above
    CONFIDENCE_THRESHOLD (comparing against unreliable/absent ground
    truth would just measure MediaPipe's own noise, not the model).
  - PCK@alpha: fraction of confident joints predicted within
    alpha * torso_size of the ground truth. torso_size = distance from
    neck to mid_hip in each frame, the usual normalizer so the metric
    isn't sensitive to how close the subject is to the camera.
"""

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import JOINT_ORDER, NUM_JOINTS, CONFIDENCE_THRESHOLD
from dataset import CSIPoseDataset
from model import CSIPoseNet

NECK_IDX = JOINT_ORDER.index("neck")
MID_HIP_IDX = JOINT_ORDER.index("mid_hip")


def torso_size(y):
    """y: (N, NUM_JOINTS*2) normalized coords -> (N,) torso length."""
    y = y.view(-1, NUM_JOINTS, 2)
    neck = y[:, NECK_IDX]
    hip = y[:, MID_HIP_IDX]
    return torch.norm(neck - hip, dim=1).clamp_min(1e-4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True, help="windows_labeled.npz to evaluate on (e.g. a held-out session)")
    ap.add_argument("--pck-alpha", type=float, default=0.2)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    ds = CSIPoseDataset(args.data, stats=ckpt["stats"])
    loader = DataLoader(ds, batch_size=64)

    model = CSIPoseNet(in_channels=4, embed_dim=ckpt["embed_dim"], num_joints=NUM_JOINTS).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    per_joint_err = np.zeros(NUM_JOINTS)
    per_joint_count = np.zeros(NUM_JOINTS)
    per_joint_hits = np.zeros(NUM_JOINTS)

    with torch.no_grad():
        for x, y, conf in loader:
            x, y, conf = x.to(device), y.to(device), conf.to(device)
            pred = model(x)

            t_size = torso_size(y).unsqueeze(1)  # (N,1)
            pred_j = pred.view(-1, NUM_JOINTS, 2)
            y_j = y.view(-1, NUM_JOINTS, 2)
            err = torch.norm(pred_j - y_j, dim=2)  # (N, NUM_JOINTS)

            mask = conf > CONFIDENCE_THRESHOLD
            hit = (err < args.pck_alpha * t_size) & mask

            per_joint_err += (err * mask).sum(0).cpu().numpy()
            per_joint_count += mask.sum(0).cpu().numpy()
            per_joint_hits += hit.sum(0).cpu().numpy()

    print(f"\n{'joint':<18}{'MPJPE':>10}{'PCK@'+str(args.pck_alpha):>10}{'n':>8}")
    total_err, total_n, total_hits = 0.0, 0.0, 0.0
    for i, joint in enumerate(JOINT_ORDER):
        n = per_joint_count[i]
        if n == 0:
            print(f"{joint:<18}{'--':>10}{'--':>10}{0:>8}")
            continue
        mpjpe = per_joint_err[i] / n
        pck = per_joint_hits[i] / n
        print(f"{joint:<18}{mpjpe:>10.4f}{pck:>10.3f}{int(n):>8}")
        total_err += per_joint_err[i]
        total_n += n
        total_hits += per_joint_hits[i]

    print("-" * 46)
    if total_n > 0:
        print(f"{'OVERALL':<18}{total_err/total_n:>10.4f}{total_hits/total_n:>10.3f}{int(total_n):>8}")
    else:
        print("No confident joints found in evaluation set.")


if __name__ == "__main__":
    main()
