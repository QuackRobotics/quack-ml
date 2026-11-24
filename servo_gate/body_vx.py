"""Body-forward speed: base displacement projected onto the base's own yaw, step by step.

World-x speed reads a run that turns as slow even when it walks at full speed;
the gate scores vx body-forward (Chief, 2026-11-24).
"""
import numpy as np


def body_forward_speed(px, py, yaw, dt):
    """Mean speed along the heading: sum of (dx cos yaw + dy sin yaw) over steps, over elapsed time.

    px, py, yaw are samples at a fixed interval dt (s); yaw in rad, taken at the start of each step.
    """
    px, py, yaw = (np.asarray(a, dtype=float) for a in (px, py, yaw))
    if not (len(px) == len(py) == len(yaw)) or len(px) < 2:
        raise ValueError("px, py, yaw need equal length >= 2")
    d = np.diff(px) * np.cos(yaw[:-1]) + np.diff(py) * np.sin(yaw[:-1])
    return float(d.sum() / ((len(px) - 1) * dt))
