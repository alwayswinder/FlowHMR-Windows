"""Shared source skeleton and reference-pose geometry, independent of FBX SDK."""
import numpy as np

# The source rig vocabulary understood by FlowHMR 2 MMR and RGB MoCap.
# Order follows SMPL-H, including the order of its finger joints and weights.
FBX_JOINT_NAMES = (
    "Hips", "LeftUpLeg", "RightUpLeg", "Spine", "LeftLeg", "RightLeg", "Spine1",
    "LeftFoot", "RightFoot", "Spine2", "LeftToeBase", "RightToeBase", "Neck",
    "LeftShoulder", "RightShoulder", "Head", "LeftArm", "RightArm", "LeftForeArm",
    "RightForeArm", "LeftHand", "RightHand",
    *(f"{side}Hand{finger}{joint}" for side in ("Left", "Right")
      for finger in ("Index", "Middle", "Pinky", "Ring", "Thumb") for joint in (1, 2, 3)),
)
HEAD_JOINT = FBX_JOINT_NAMES.index("Head")


def _align_direction(source, target):
    source = source / np.linalg.norm(source)
    x, y, z = np.cross(source, target)
    cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    # Each target points out along the same side as its rest arm.
    return np.eye(3) + cross + cross @ cross / (1. + source @ target)


def _reference_rotations(joints, initial_root, pose):
    """Level the body and set both arm segments to the requested angle from horizontal."""
    rotations = np.repeat(np.eye(3)[None], len(joints), axis=0)
    forward = initial_root[[0, 2], 2]
    yaw = (np.arctan2(*forward) if np.linalg.norm(forward) > 1e-6
           else np.arctan2(-initial_root[2, 0], initial_root[0, 0]))
    c, s = np.cos(yaw), np.sin(yaw)
    rotations[0] = [[c, 0., s], [0., 1., 0.], [-s, 0., c]]
    angle = 0. if pose == "t" else np.pi / 4
    for upper, lower, hand in ((16, 18, 20), (17, 19, 21)):
        arm = joints[lower] - joints[upper]
        target = np.array([np.copysign(np.cos(angle), arm[0]), -np.sin(angle), 0.])
        rotations[upper] = _align_direction(arm, target)
        rotations[lower] = _align_direction(joints[hand] - joints[lower], rotations[upper].T @ target)
    return rotations


def _display_frames(joints, parents, *, head_top_y):
    """A right-handed frame with +Y along each bone's reference direction."""
    main_child = {0: 3, 9: 12, 20: 25, 21: 40}
    frames, tips = [], {}
    for joint, parent in enumerate(parents):
        children = np.flatnonzero(parents == joint)
        if len(children):
            direction = joints[main_child.get(joint, children[0])] - joints[joint]
        elif joint == HEAD_JOINT:
            # SMPL-H's Neck -> Head offset points toward the face. It is
            # the neck axis, not the skull's up axis used by retargeters.
            direction = np.array([0., head_top_y - joints[joint, 1], 0.])
            if direction[1] <= 0.:
                raise ValueError("Invalid SMPL-H head height for FBX export")
        else:
            direction = joints[joint] - joints[parent]
            if joint in (10, 11):  # Toe tips follow the floor.
                direction[1] = 0.
                direction *= .65
        length = float(np.linalg.norm(direction))
        if length < 1e-6:
            raise ValueError(f"Degenerate rest bone: {FBX_JOINT_NAMES[joint]}")
        if not len(children):
            tips[joint] = length
        y = direction / length
        guide = (0., 0., 1.) if abs(y[2]) < .95 else (0., 1., 0.)
        x = np.cross(y, guide)
        x /= np.linalg.norm(x)
        frames.append(np.stack((x, y, np.cross(x, y)), axis=1))
    return np.asarray(frames), tips
