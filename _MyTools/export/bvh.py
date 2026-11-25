"""Export SMPL-H animation specifically for 3ds Max Biped -> BIP.

Body naming and ZXY continuity follow RGB MoCap's Biped profile. Finger
channels compensate Biped's importer; this is not a generic BVH animation.
Units remain metres and the world is Y up. Validated with Max 2026.3.
"""
from collections.abc import Mapping
from operator import index
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
import warnings

import numpy as np
from scipy.spatial.transform import Rotation

from flowhmr.core.bodymodels.smplh_data import load_smplh_data
from .smplh_rig import FBX_JOINT_NAMES, HEAD_JOINT, _display_frames, _reference_rotations
from .biped_fingers import encode_fingers
from flowhmr.multishot.timeline import held_indices, validity


def biped_names():
    mapping = dict(Hips='Hips', Spine='Chest', Spine1='Chest2', Spine2='Chest3', Neck='Neck', Head='Head')
    for side in ('Left', 'Right'):
        for source, target in (('UpLeg', 'Hip'), ('Leg', 'Knee'), ('Foot', 'Ankle'),
                               ('ToeBase', 'Toe'), ('Shoulder', 'Collar'), ('Arm', 'Shoulder'),
                               ('ForeArm', 'Elbow'), ('Hand', 'Wrist')):
            mapping[side + source] = side + target
        for digit, finger in enumerate(('Thumb', 'Index', 'Middle', 'Ring', 'Pinky')):
            for segment in range(1, 4):
                mapping[f'{side}Hand{finger}{segment}'] = f'{side}Finger{digit}' + (str(segment - 1) if segment > 1 else '')
    return tuple(mapping[name] for name in FBX_JOINT_NAMES)


def continuous_angles(rotations):
    """Choose equivalent intrinsic ZXY triples close to the previous key."""
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Gimbal lock detected')
        angles = rotations.as_euler('ZXY')
    previous = np.zeros(3)
    for frame, angle in enumerate(angles):
        alternative = angle + (np.pi, 0., np.pi)
        alternative[1] = np.pi - angle[1]
        angle = angle + 2 * np.pi * np.round((previous - angle) / (2 * np.pi))
        alternative += 2 * np.pi * np.round((previous - alternative) / (2 * np.pi))
        angles[frame] = min((angle, alternative), key=lambda value: np.linalg.norm(value - previous))
        previous = angles[frame]
    return np.degrees(angles)


def _write(motion, output, reference_path, prepend_tpose, start_frame, units_per_meter):
    poses = np.asarray(motion['poses'], dtype=np.float64)
    if poses.ndim == 2 and poses.shape[1] == 156:
        poses = poses.reshape(-1, 52, 3)
    trans = np.asarray(motion['trans'], dtype=np.float64)
    fps = float(np.asarray(motion.get('mocap_framerate', 30)).item())
    start = index(start_frame)
    if (poses.ndim != 3 or poses.shape[1:] != (52, 3) or not len(poses)
            or trans.shape != (len(poses), 3) or start < 0
            or not np.isfinite(fps) or fps <= 0
            or not np.isfinite(poses).all() or not np.isfinite(trans).all()):
        raise ValueError('Invalid SMPL-H animation for BVH export')
    if 'valid_frames' in motion:
        held = held_indices(validity(motion['valid_frames'], len(poses)))
        poses, trans = poses[held], trans[held]
    model = load_smplh_data(reference_path)
    betas = np.atleast_2d(np.asarray(motion.get('betas', ()), dtype=np.float64))
    if betas.ndim != 2 or not len(betas) or not np.isfinite(betas).all():
        raise ValueError('Invalid SMPL-H shape for BVH export')
    betas = betas[0, :model['shapedirs'].shape[-1]]
    mesh = model['v_template'] + model['shapedirs'][:, :, :len(betas)] @ betas
    joints = model['J_regressor'] @ mesh
    parents = model['kintree_table'][0].astype(np.int64, copy=True)
    parents[0] = -1
    head = mesh[model['weights'][:, HEAD_JOINT] > .5, 1].max()
    basis, lengths = _display_frames(joints, parents, head_top_y=head)
    ends = {joint: basis[joint, :, 1] * length for joint, length in lengths.items()}
    offsets = joints.copy()
    offsets[0] = 0.  # Absolute hip location is encoded in the position channels.
    offsets[1:] -= joints[parents[1:]]
    roots = trans + joints[0]
    rotations = Rotation.from_rotvec(poses.reshape(-1, 3)).as_matrix().reshape(-1, 52, 3, 3)
    if start:
        # BVH cannot encode an absolute start time: hold the first pose until entry.
        roots = np.concatenate((np.repeat(roots[:1], start, axis=0), roots))
        rotations = np.concatenate((np.repeat(rotations[:1], start, axis=0), rotations))
    if prepend_tpose:
        reference = _reference_rotations(joints, np.eye(3), 't')
        root = joints[0] + (trans[0, 0], -mesh[:, 1].min(), trans[0, 2])
        roots = np.concatenate((root[None], roots))
        rotations = np.concatenate((reference[None], rotations))
    rotations = encode_fingers(rotations, joints, ends)
    angles = np.stack([continuous_angles(Rotation.from_matrix(rotations[:, joint])) for joint in range(52)], axis=1)
    order = []
    names = biped_names()

    def visit(joint, depth):
        indent = '  ' * depth
        order.append(joint)
        output.write(indent + ('ROOT ' if joint == 0 else 'JOINT ') + names[joint] + '\n' + indent + '{\n')
        output.write(indent + '  OFFSET ' + ' '.join(f'{v:.9f}' for v in offsets[joint] * units_per_meter) + '\n')
        channels = '6 Xposition Yposition Zposition ' if joint == 0 else '3 '
        output.write(indent + '  CHANNELS ' + channels + 'Zrotation Xrotation Yrotation\n')
        children = np.flatnonzero(parents == joint)
        for child in children:
            visit(int(child), depth + 1)
        if not len(children):
            output.write(indent + '  End Site\n' + indent + '  {\n' + indent + '    OFFSET '
                         + ' '.join(f'{v:.9f}' for v in ends[joint] * units_per_meter) + '\n' + indent + '  }\n')
        output.write(indent + '}\n')

    output.write('HIERARCHY\n')
    visit(0, 0)
    output.write(f'MOTION\nFrames: {len(roots)}\nFrame Time: {1 / fps:.12f}\n')
    for root, angle in zip(roots, angles):
        output.write(' '.join(f'{v:.9f}' for v in np.r_[root * units_per_meter, angle[order].reshape(-1)]) + '\n')


def export_bvh(motion, path, *, reference_path=None, prepend_tpose=False, start_frame=0, units_per_meter=1):
    """Atomically export an NPZ path or motion mapping. Never alters the source."""
    if not isinstance(motion, Mapping):
        with np.load(motion, allow_pickle=False) as data:
            return export_bvh(dict(data), path, reference_path=reference_path,
                              prepend_tpose=prepend_tpose, start_frame=start_frame, units_per_meter=units_per_meter)
    path = Path(path)
    temporary = None
    try:
        with NamedTemporaryFile(mode='w', encoding='ascii', newline='\n', dir=path.parent,
                                prefix='.bvh-', suffix='.tmp', delete=False) as output:
            temporary = Path(output.name)
            _write(motion, output, reference_path, prepend_tpose, start_frame, units_per_meter)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
