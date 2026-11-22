"""Seeded perturbation of the home pose, so a deterministic gate run can be repeated as a spread."""
import numpy as np

# 0.02 rad (~1.1 deg) per joint: small enough to stay near the trained home pose,
# large enough to move the run off the single deterministic trajectory.
JOINT_NOISE_RAD = 0.02


def perturb_joints(qpos: np.ndarray, qadr: np.ndarray, seed: int,
                   amplitude: float = JOINT_NOISE_RAD) -> np.ndarray:
    """Return qpos with uniform +-amplitude noise on the joint addresses qadr."""
    if amplitude <= 0:
        raise ValueError(f"amplitude must be > 0, got {amplitude}")
    rng = np.random.default_rng(seed)
    out = qpos.copy()
    out[qadr] += rng.uniform(-amplitude, amplitude, size=len(qadr))
    return out
