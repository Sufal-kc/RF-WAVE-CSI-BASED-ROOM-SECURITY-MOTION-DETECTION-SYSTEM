"""
Shared constants + CSI calibration helpers for the CSI -> Pose pipeline.

The calibration math here is copied verbatim (in spirit) from
csi_rt_cld2.py: each CSI packet has its own random linear phase slope
(sampling-time-offset artifact) that MUST be removed per-packet, before
any averaging across packets. Averaging raw phase first is the bug that
pipeline already fixed once for the live plotter -- we must not
reintroduce it here.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Joint layout -- must match JOINT_ORDER / POSE_CSV_HEADER in collect.py
# ---------------------------------------------------------------------------
JOINT_ORDER = [
    "nose", "neck",
    "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist",
    "mid_hip",
    "right_hip", "right_knee", "right_ankle",
    "left_hip", "left_knee", "left_ankle",
]
NUM_JOINTS = len(JOINT_ORDER)
CONFIDENCE_THRESHOLD = 0.5  # same threshold collect.py uses to zero-out a joint

# Skeleton bone list (parent, child) -- used for an optional bone-length
# consistency loss / sanity checks. Matches the skeleton drawn in collect.py.
SKELETON_BONES = [
    ("nose", "neck"), ("neck", "right_shoulder"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("neck", "left_shoulder"), ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"), ("neck", "mid_hip"),
    ("mid_hip", "right_hip"), ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"), ("mid_hip", "left_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
]


# ---------------------------------------------------------------------------
# Per-packet CSI phase calibration (unwrap + detrend), copied from
# csi_rt_cld2.py so preprocessing uses the exact same fix.
# ---------------------------------------------------------------------------
def detrend_phase(phase):
    x = np.arange(len(phase))
    slope, intercept = np.polyfit(x, phase, 1)
    return phase - (slope * x + intercept)


def calibrate_packet_phase(phase):
    return detrend_phase(np.unwrap(phase))


def circular_mean(phases_stack, axis=0):
    sin_mean = np.mean(np.sin(phases_stack), axis=axis)
    cos_mean = np.mean(np.cos(phases_stack), axis=axis)
    return np.arctan2(sin_mean, cos_mean)


def circular_variance(phases_stack, axis=0):
    """0 = perfectly stable, -> 1 = fully decorrelated. This is the same
    'activity' signal csi_rt_cld2.py plots as the presence-detection cue,
    and it is exactly the kind of feature a static amplitude/phase mean
    can't see -- so we keep it as a channel, not just an average."""
    resultant = np.abs(np.mean(np.exp(1j * phases_stack), axis=axis))
    return 1.0 - resultant


def reject_outlier_packets(calibrated_stack, thresh=3.5):
    n_packets = calibrated_stack.shape[0]
    if n_packets < 5:
        return calibrated_stack
    median_vec = np.median(calibrated_stack, axis=0)
    deviations = np.mean(np.abs(calibrated_stack - median_vec), axis=1)
    med_dev = np.median(deviations)
    mad = np.median(np.abs(deviations - med_dev)) + 1e-9
    robust_z = 0.6745 * (deviations - med_dev) / mad
    keep = np.abs(robust_z) < thresh
    if keep.sum() == 0:
        return calibrated_stack
    return calibrated_stack[keep]
