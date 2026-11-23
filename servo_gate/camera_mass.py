"""Camera point mass on the head front face (Woz 2026-11-23: +10 g budget, <= 14 g upper bound).

The camera is a massless-geometry child body of head_assembly (the head_roll body) carrying a point
mass. Position, at the home keyframe of Playground b9be205 (world frame, m): the front face of the
`head` mesh (max x of its vertices, 0.101), on the head centreline (y of head_assembly), at mid height
of the head shell (z 0.354, used as eye height: the MJCF has no eye geoms). Replace CAMERA_POS_WORLD_HOME
when Woz gives the real mount point.
"""
import mujoco
import numpy as np

HEAD_BODY = "head_assembly"
CAMERA_POS_WORLD_HOME = np.array([0.101, 0.0001, 0.354])
# Woz 2026-11-23 01:33, ray-cast on head.stl: 10 g at (10, 0, -51) mm in the head_assembly frame
# (+x up, -z forward, front face at z = -59.1 mm). Opt-in via pos_head so earlier runs reproduce.
WOZ_POS_HEAD = np.array([0.010, 0.0, -0.051])
POINT_INERTIA = 1e-9  # kg m^2, keeps the compiler happy; a 10 g point mass has ~0 own inertia


def _home(m: mujoco.MjModel) -> mujoco.MjData:
    d = mujoco.MjData(m)
    d.qpos[:] = m.keyframe("home").qpos
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    return d


def load_with_camera(xml_path: str, grams: float, pos_head=None) -> mujoco.MjModel:
    """Compile xml_path; if grams > 0, add the camera point mass under head_assembly.

    pos_head: mount point in the head_assembly frame (m); default is CAMERA_POS_WORLD_HOME at home.
    """
    if grams < 0:
        raise ValueError(f"camera mass must be >= 0 g, got {grams}")
    spec = mujoco.MjSpec.from_file(xml_path)
    if grams == 0:
        return spec.compile()
    m0 = spec.compile()
    d0 = _home(m0)
    b = m0.body(HEAD_BODY).id
    if pos_head is None:
        local = d0.xmat[b].reshape(3, 3).T @ (CAMERA_POS_WORLD_HOME - d0.xpos[b])
    else:
        local = np.asarray(pos_head, dtype=float)
    cam = spec.find_body(HEAD_BODY).add_body()
    cam.name = "camera_mass"
    cam.pos = local
    cam.mass = grams / 1000.0
    cam.ipos = [0.0, 0.0, 0.0]  # MjSpec does not zero it for a new body in this tree
    cam.iquat = [1.0, 0.0, 0.0, 0.0]
    cam.inertia = [POINT_INERTIA] * 3
    cam.explicitinertial = True
    return spec.compile()


def head_com_home(m: mujoco.MjModel) -> tuple[float, np.ndarray]:
    """(mass kg, world CoM m) of the head_assembly subtree at the home keyframe."""
    d = _home(m)
    b = m.body(HEAD_BODY).id
    return float(m.body_subtreemass[b]), d.subtree_com[b].copy()
