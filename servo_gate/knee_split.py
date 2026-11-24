"""Stance/swing split of actuator torque RMS using the 50 Hz foot-floor contact log.

Physics steps (500 Hz) in control interval j take the contact sampled at the end of that interval,
which is what the harness logs at tick j, so phase resolution is one 20 ms control step.
"""
import numpy as np


def phase_rms(force: np.ndarray, contact: np.ndarray, decim: int, dt: float, t0: float) -> dict:
    """RMS of one actuator's physics-step force overall, in stance (contact) and in swing, for t >= t0."""
    if len(force) != len(contact) * decim:
        raise ValueError(f"force has {len(force)} steps, expected {len(contact)} * {decim}")
    in_contact = np.repeat(np.asarray(contact) > 0.5, decim)
    t = np.arange(1, len(force) + 1) * dt
    steady = t >= t0
    rms = lambda x: float(np.sqrt(np.mean(np.square(x)))) if len(x) else float("nan")
    return {"all": rms(force[steady]), "stance": rms(force[steady & in_contact]),
            "swing": rms(force[steady & ~in_contact]), "stance_frac": float(np.mean(in_contact[steady]))}
