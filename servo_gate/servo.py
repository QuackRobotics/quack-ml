"""Per-joint-group servo overrides for the gate harness.

A servo spec is JSON:
  {"name": "<servo_name>", "source": "<where the numbers come from>",
   "groups": {"knee": {"forcerange": 1.91, "kp": 13.37, "armature": 0.027,
                       "damping": 0.56, "frictionloss": 0.068, "added_mass": 0.034}, ...}}
Groups are the leg joint groups below; each applies to the left and right actuator.
Every field is optional; an omitted group or field keeps the model's own value.
forcerange is the symmetric limit in Nm, kp the position gain in Nm/rad, armature
the reflected rotor inertia in kg m^2 of the joint's DOF, damping (N m s/rad) and
frictionloss (N m) the joint's passive terms. added_mass (kg) is the extra servo mass
per side; it goes on the joint's parent body (where the servo housing is assumed to
sit), at that body's centre of mass, with its inertia scaled by the new/old mass ratio.
"""
import json
import re

import mujoco
import numpy as np

LEG_GROUPS = ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")
FIELDS = ("forcerange", "kp", "armature", "damping", "frictionloss", "added_mass")
NAME_RE = re.compile(r"[A-Za-z0-9_.-]+")


def load_spec(path: str) -> dict:
    """Read and validate a servo spec; raise ValueError on anything unexpected."""
    with open(path) as fh:
        spec = json.load(fh)
    unknown = set(spec) - {"name", "source", "groups"}
    if unknown:
        raise ValueError(f"unknown top-level keys: {sorted(unknown)}")
    if not isinstance(spec.get("name"), str) or not NAME_RE.fullmatch(spec["name"]):
        raise ValueError("name must be a non-empty string of letters, digits, _ . -")
    groups = spec.get("groups")
    if not isinstance(groups, dict) or not groups:
        raise ValueError("groups must be a non-empty object")
    for group, params in groups.items():
        if group not in LEG_GROUPS:
            raise ValueError(f"unknown group {group!r}; expected one of {LEG_GROUPS}")
        if not isinstance(params, dict) or not params:
            raise ValueError(f"group {group!r} must set at least one of {FIELDS}")
        for field, value in params.items():
            if field not in FIELDS:
                raise ValueError(f"group {group!r}: unknown field {field!r}")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"group {group!r}: {field} must be a positive number")
    return spec


def group_actuators(model: mujoco.MjModel, group: str) -> list:
    ids = [k for k in range(model.nu)
           if model.actuator(k).name in (f"left_{group}", f"right_{group}")]
    if len(ids) != 2:
        raise ValueError(f"group {group!r}: expected left_ and right_ actuators, found {len(ids)}")
    return ids


def apply_spec(model: mujoco.MjModel, spec: dict) -> list:
    """Write the spec into the compiled model in place; return one row per changed actuator."""
    rows = []
    mass_changed = False
    for group, params in spec["groups"].items():
        for k in group_actuators(model, group):
            if model.actuator_biastype[k] != mujoco.mjtBias.mjBIAS_AFFINE or not np.isclose(
                    model.actuator_gainprm[k, 0], -model.actuator_biasprm[k, 1]):
                raise ValueError(f"{model.actuator(k).name} is not a position actuator")
            dof = model.jnt_dofadr[model.actuator_trnid[k, 0]]
            if "kp" in params:
                model.actuator_gainprm[k, 0] = params["kp"]
                model.actuator_biasprm[k, 1] = -params["kp"]
            if "forcerange" in params:
                model.actuator_forcelimited[k] = 1
                model.actuator_forcerange[k] = (-params["forcerange"], params["forcerange"])
            if "armature" in params:
                model.dof_armature[dof] = params["armature"]
            if "damping" in params:
                model.dof_damping[dof] = params["damping"]
            if "frictionloss" in params:
                model.dof_frictionloss[dof] = params["frictionloss"]
            body = model.body_parentid[model.jnt_bodyid[model.actuator_trnid[k, 0]]]
            if "added_mass" in params:
                old = model.body_mass[body]
                model.body_mass[body] = old + params["added_mass"]
                model.body_inertia[body] *= model.body_mass[body] / old
                mass_changed = True
            rows.append((model.actuator(k).name, float(model.actuator_gainprm[k, 0]),
                         float(model.actuator_forcerange[k, 1]), float(model.dof_armature[dof]),
                         float(model.dof_damping[dof]), float(model.dof_frictionloss[dof]),
                         model.body(body).name, float(model.body_mass[body])))
    if mass_changed:
        mujoco.mj_setConst(model, mujoco.MjData(model))
    return rows
