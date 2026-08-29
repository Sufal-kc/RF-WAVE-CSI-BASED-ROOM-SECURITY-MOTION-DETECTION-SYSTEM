#!/usr/bin/env python3
"""
Build training-ready tensors from a collect.py session
(<session>_csi.csv + <session>_pose.csv), or from many sessions at once.

Two things get produced, saved to --out-dir:

1. windows_unlabeled.npz
   Every CSI window in the whole recording (dense sliding window over the
   full stream, ignoring the camera entirely). This is the large,
   effectively "free" pool used for self-supervised pretraining -- it
   exists precisely because CSI arrives far more densely than confident
   camera-derived labels do.

2. windows_labeled.npz
   One CSI feature stack per camera frame, time-aligned via
   pandas.merge_asof exactly as the collect.py docstring recommends, plus
   the pose label vector and a per-joint confidence mask. Only pose rows
   that have at least one confident joint are kept -- fully-zero rows
   (nothing detected) are dropped from the labeled set (they still show
   up in the unlabeled set, since the CSI itself is still valid signal
   for "no person" / background).

Feature representation
-----------------------
For a window of raw CSI packets:
  - unwrap + per-packet detrend phase (bug fix from csi_rt_cld2.py) BEFORE
    any averaging
  - reject outlier packets (MAD-based), same as the live plotter
  - per-subcarrier: amplitude mean, amplitude std, calibrated-phase
    circular mean, calibrated-phase circular variance
  -> a (num_subcarriers, 4) frame

--history windows are stacked to give the CNN temporal context, so a
training example is a (T, F, 4) tensor: time x subcarrier x
{amp_mean, amp_std, phase_mean, phase_circvar}. This is deliberately
"image-shaped" (time-frequency map, like a spectrogram) so a standard 2D
CNN can be used directly.
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from common import (
    JOINT_ORDER, NUM_JOINTS, CONFIDENCE_THRESHOLD,
    calibrate_packet_phase, circular_mean, circular_variance,
    reject_outlier_packets,
)


def load_csi_csv(path):
    df = pd.read_csv(path)
    df["amplitude"] = df["amplitude_json"].apply(json.loads).apply(np.array)
    df["phase"] = df["phase_json"].apply(json.loads).apply(np.array)
    # keep only the dominant subcarrier count in this file -- a handful of
    # corrupted lines can slip through parse_csi_line with the wrong length
    target_len = df["num_subcarriers"].mode().iloc[0]
    df = df[df["num_subcarriers"] == target_len].reset_index(drop=True)
    return df, int(target_len)


def load_pose_csv(path):
    return pd.read_csv(path).reset_index(drop=True)


def csi_window_features(sub_df, num_subcarriers, outlier_thresh=3.5):
    """sub_df: rows of the csi dataframe falling inside one time window.
    Returns a (num_subcarriers, 4) array, or None if there's nothing usable."""
    if len(sub_df) == 0:
        return None
    amps = np.stack(sub_df["amplitude"].to_numpy())
    raw_phases = np.stack(sub_df["phase"].to_numpy())

    calibrated = np.stack([calibrate_packet_phase(p) for p in raw_phases])
    calibrated = reject_outlier_packets(calibrated, thresh=outlier_thresh)

    amp_mean = np.mean(amps, axis=0)
    amp_std = np.std(amps, axis=0)
    phase_mean = circular_mean(calibrated, axis=0)
    phase_var = circular_variance(calibrated, axis=0)

    return np.stack([amp_mean, amp_std, phase_mean, phase_var], axis=-1)  # (F, 4)


def build_unlabeled_windows(csi_df, num_subcarriers, window_s, stride_s, history):
    """Dense sliding window over the whole CSI stream, independent of the
    camera. This is the self-supervised pretraining pool."""
    t = csi_df["t_rel"].to_numpy()
    if len(t) == 0:
        return np.empty((0, history, num_subcarriers, 4), dtype=np.float32)

    t_start, t_end = t[0], t[-1]
    centers = np.arange(t_start + window_s, t_end, stride_s)

    frames = []  # single-window (F,4) features, one per stride step
    for c in centers:
        mask = (t >= c - window_s / 2) & (t < c + window_s / 2)
        feat = csi_window_features(csi_df[mask], num_subcarriers)
        frames.append(feat)

    # stack `history` consecutive frames into one training example, sliding
    # by 1 so the pretraining pool stays large; drop examples with any
    # missing (None) frame.
    examples = []
    for i in range(history - 1, len(frames)):
        chunk = frames[i - history + 1:i + 1]
        if any(f is None for f in chunk):
            continue
        examples.append(np.stack(chunk, axis=0))  # (history, F, 4)

    if not examples:
        return np.empty((0, history, num_subcarriers, 4), dtype=np.float32)
    return np.stack(examples).astype(np.float32)


def build_labeled_windows(csi_df, pose_df, num_subcarriers, window_s, history, stride_frames=1):
    """For each pose frame (subsampled by stride_frames), pull the CSI
    history-window ending at that frame's timestamp via merge_asof-style
    nearest alignment, and pair it with the pose label + confidence mask."""
    t_csi = csi_df["t_rel"].to_numpy()

    X, Y, CONF = [], [], []
    pose_rows = pose_df.iloc[::stride_frames]

    for _, row in pose_rows.iterrows():
        t_center = row["t_rel"]

        # confidence mask + label vector for this frame
        conf = np.array([row[f"{j}_confidence"] for j in JOINT_ORDER], dtype=np.float32)
        if np.all(conf < CONFIDENCE_THRESHOLD):
            continue  # nothing detected in this frame -> not usable as a label

        label = np.array(
            [[row[f"{j}_x_norm"], row[f"{j}_y_norm"]] for j in JOINT_ORDER],
            dtype=np.float32,
        ).reshape(-1)  # (NUM_JOINTS*2,)

        # `history` consecutive windows ending at t_center
        chunk = []
        ok = True
        for h in range(history - 1, -1, -1):
            c = t_center - h * window_s
            mask = (t_csi >= c - window_s / 2) & (t_csi < c + window_s / 2)
            feat = csi_window_features(csi_df[mask], num_subcarriers)
            if feat is None:
                ok = False
                break
            chunk.append(feat)
        if not ok:
            continue

        X.append(np.stack(chunk, axis=0))
        Y.append(label)
        CONF.append(conf)

    if not X:
        return (np.empty((0, history, num_subcarriers, 4), dtype=np.float32),
                np.empty((0, NUM_JOINTS * 2), dtype=np.float32),
                np.empty((0, NUM_JOINTS), dtype=np.float32))

    return np.stack(X).astype(np.float32), np.stack(Y).astype(np.float32), np.stack(CONF).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True,
                     help="Directory containing <session>_csi.csv / <session>_pose.csv pairs")
    ap.add_argument("--out-dir", default="./prepared")
    ap.add_argument("--window", type=float, default=0.1,
                     help="Seconds per CSI feature window (roughly one camera frame period)")
    ap.add_argument("--stride", type=float, default=0.05,
                     help="Stride (s) for the dense unlabeled sliding window")
    ap.add_argument("--history", type=int, default=8,
                     help="Number of consecutive windows stacked as temporal context (T dim)")
    ap.add_argument("--pose-stride-frames", type=int, default=1,
                     help="Use every Nth pose frame as a label (downsample if camera fps is high)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    csi_files = sorted(glob.glob(os.path.join(args.data_dir, "*_csi.csv")))
    if not csi_files:
        raise SystemExit(f"No *_csi.csv files found in {args.data_dir}")

    all_unlabeled, all_X, all_Y, all_conf = [], [], [], []

    for csi_path in csi_files:
        session = os.path.basename(csi_path)[: -len("_csi.csv")]
        pose_path = os.path.join(args.data_dir, f"{session}_pose.csv")
        if not os.path.exists(pose_path):
            print(f"[skip] no matching pose file for {session}")
            continue

        print(f"[session] {session}")
        csi_df, num_subcarriers = load_csi_csv(csi_path)
        pose_df = load_pose_csv(pose_path)
        print(f"  csi packets={len(csi_df)} (subcarriers={num_subcarriers})  pose frames={len(pose_df)}")

        unl = build_unlabeled_windows(csi_df, num_subcarriers, args.window, args.stride, args.history)
        X, Y, conf = build_labeled_windows(csi_df, pose_df, num_subcarriers, args.window,
                                            args.history, args.pose_stride_frames)
        print(f"  -> unlabeled windows={len(unl)}  labeled examples={len(X)}")

        all_unlabeled.append(unl)
        all_X.append(X)
        all_Y.append(Y)
        all_conf.append(conf)

    unlabeled = np.concatenate(all_unlabeled, axis=0)
    X = np.concatenate(all_X, axis=0)
    Y = np.concatenate(all_Y, axis=0)
    conf = np.concatenate(all_conf, axis=0)

    np.savez_compressed(os.path.join(args.out_dir, "windows_unlabeled.npz"), X=unlabeled)
    np.savez_compressed(os.path.join(args.out_dir, "windows_labeled.npz"), X=X, Y=Y, conf=conf)

    print(f"\n[done] unlabeled windows: {unlabeled.shape}")
    print(f"[done] labeled examples : X={X.shape} Y={Y.shape} conf={conf.shape}")
    print(f"[done] saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
