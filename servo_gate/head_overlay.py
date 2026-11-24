"""Head-offset overlay on the policy's motor targets (Playground's disabled motor_targets[5:9] path).

BEST_WALK_ONNX_2 ignores the head command slots (obs 9-12), so head offsets are added to the
policy's own head targets after inference. The overlaid targets are what the next observation
feeds back (obs 83-96), exactly as a runtime overlay would.
"""
import numpy as np

HEAD = slice(5, 9)  # neck_pitch, head_pitch, head_yaw, head_roll in policy-actuator orde
HEAD_JOINTS = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")


def parse_fracs(text: str) -> np.ndarray:
    """'n,p,y,r' fractions of each head joint's full range -> array of 4 floats in [-1, 1]."""
    vals = np.array([float(v) for v in text.split(",")])
    if vals.shape != (4,):
        raise ValueError(f"need 4 comma-separated fractions (neck,pitch,yaw,roll), got {text!r}")
    if np.any(np.abs(vals) > 1):
        raise ValueError(f"fractions must be within [-1, 1], got {vals.tolist()}")
    return vals


def offsets_rad(fracs: np.ndarray, joint_range: np.ndarray) -> np.ndarray:
    """Fractions of full range (hi - lo) -> radians. joint_range has shape (4, 2)."""
    return fracs * (joint_range[:, 1] - joint_range[:, 0])


def overlay(targets: np.ndarray, offset: np.ndarray, ctrl_lo: np.ndarray, ctrl_hi: np.ndarray):
    """Add offset to the 4 head targets and clip to the head ctrl range.

    Returns (new 14-vector, number of head targets that hit the ctrl range). Leg targets are untouched.
    """
    out = targets.copy()
    raw = out[HEAD] + offset
    out[HEAD] = np.clip(raw, ctrl_lo, ctrl_hi)
    return out, int(np.sum((raw < ctrl_lo) | (raw > ctrl_hi)))
