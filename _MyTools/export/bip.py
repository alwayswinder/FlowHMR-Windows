"""Local 3ds Max Biped export and readback for the FlowHMR viewer."""
import base64
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np


RUNTIME_DIR = Path(__file__).with_name('max')
REVISION = 'v2'


def max_batch_path():
    if os.name != 'nt':
        return None
    configured = os.environ.get('FLOWHMR_3DSMAX_BATCH')
    if configured:
        path = Path(configured)
        return path.resolve() if path.is_file() and path.name.lower() == '3dsmaxbatch.exe' else None
    root = Path(os.environ.get('ProgramFiles', 'C:/Program Files')) / 'Autodesk'
    return next((path for directory in sorted(root.glob('3ds Max 20[0-9][0-9]'), reverse=True)
                 if (path := directory / '3dsmaxbatch.exe').is_file()), None)


def _prepare_runtime(folder):
    for name in ('run.ms', 'finish.ms'):
        shutil.copyfile(RUNTIME_DIR / name, folder / name)
    preset = (RUNTIME_DIR / 'Mixamo-Standard.list').read_text(encoding='utf-8-sig').splitlines()
    slots = [name.removeprefix('mixamorig:') for name in preset[1:70]]
    for index in (11, 16, 21, 29, 33, 37, 41, 45, 53, 57, 61, 65, 69):
        slots[index - 1] = slots[index - 2] + '_end'
    (folder / 'FlowHMR.list').write_text('\n'.join(['v2', *slots, 'FlowHMR Motion', *preset[71:]]) + '\n', encoding='ascii')
    script = (RUNTIME_DIR / 'BsRetargetTools.ms').read_text(encoding='utf-8-sig')
    lines = script.splitlines(keepends=True)
    completion = [i for i, line in enumerate(lines) if 'messageBox "重定向 Biped 创建成功！' in line]
    lookup = 'try (bipRoot = GetBipedNode 1) catch (bipRoot = undefined)'
    if len(completion) != 1 or script.count(lookup) != 1:
        raise RuntimeError('Unsupported BsKeyTools runtime')
    # Avoid the UI completion dialog, and resolve the target after a batch scene load.
    lines[completion[0]] = '            format "[BsRetarget] Mapping construction completed.\\n"\n'
    script = ''.join(lines).replace(lookup, 'bipRoot = getNodeByName bipedRootName exact:true')
    (folder / 'BsRetargetTools.batch.ms').write_text(script, encoding='utf-8-sig')


def _batch(executable, folder, script, stage, frames, fps):
    args = [str(executable), str(folder / script), '-log', str(folder / f'{stage}.log'),
            '-mxsString', f'motion:{(folder / "motion.fbx").as_posix()}', '-mxsString', f'stage:{stage}',
            '-mxsString', f'frames:{frames}', '-mxsString', f'fps:{fps}']
    console = folder / f'{stage}-console.log'
    report = folder / 'trial.txt'
    report.unlink(missing_ok=True)
    try:
        with console.open('wb') as stream:
            result = subprocess.run(args, stdout=stream, stderr=subprocess.STDOUT, timeout=300,
                                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f'3ds Max did not finish the {stage} stage') from error
    summary = report.read_text(encoding='utf-8', errors='replace') if report.is_file() else ''
    if result.returncode or 'FLOWHMR_OK' not in summary:
        failure = next((line.partition(': ')[2] for line in summary.splitlines() if line.startswith('FAIL: ')), '')
        raise RuntimeError(f'3ds Max {stage} failed: {failure or result.returncode or "missing completion report"}')


def _preview(folder, frames, fps):
    with (folder / 'skeleton.tsv').open(encoding='ascii') as stream:
        entries = [line.rstrip('\r\n').split('\t') for line in stream]
    parents = [int(parent) for _, parent in entries]
    if not parents or parents[0] != -1 or any(not 0 <= parent < i for i, parent in enumerate(parents[1:], 1)):
        raise ValueError('Invalid Biped hierarchy')
    stride = 4
    samples = np.loadtxt(folder / 'samples.tsv', dtype=np.float32, ndmin=2)
    if samples.shape != ((frames - 1) * stride + 1, len(parents) * 6) or not np.isfinite(samples).all():
        raise ValueError('3ds Max BIP readback does not match the source timeline')
    xyz = samples.reshape(samples.shape[0], len(parents), 2, 3)
    # Max Z-up centimetres to FlowHMR Y-up metres; each sample includes a bone-axis tip.
    xyz = (xyz[..., [0, 2, 1]] * np.array([.01, .01, -.01], np.float32)).astype('<f4')
    return dict(version=1, frames=frames, fps=fps, samples_per_frame=stride, parents=parents,
                names=[name for name, _ in entries], samples=base64.b64encode(xyz.tobytes()).decode('ascii'))


def _write_reference(folder):
    import fbx

    manager = fbx.FbxManager.Create()
    try:
        settings = fbx.FbxIOSettings.Create(manager, fbx.IOSROOT)
        manager.SetIOSettings(settings)
        scene = fbx.FbxScene.Create(manager, '')
        importer = fbx.FbxImporter.Create(manager, '')
        if not importer.Initialize(str(folder / 'motion.fbx'), -1, settings) or not importer.Import(scene):
            raise ValueError('Could not read the source FBX reference')
        stacks = fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId)
        for index in reversed(range(scene.GetSrcObjectCount(stacks))):
            scene.GetSrcObject(stacks, index).Destroy()
        for index in reversed(range(scene.GetPoseCount())):
            scene.GetPose(index).Destroy()
        exporter = fbx.FbxExporter.Create(manager, '')
        if not exporter.Initialize(str(folder / 'reference.fbx'), -1, settings) or not exporter.Export(scene):
            raise ValueError('Could not save the source FBX reference')
    finally:
        manager.Destroy()


def export_bip(motion, destination, preview, scene, *, reference_path, executable):
    """Publish the BIP, animated Max scene and preview from the same readback."""
    from flowhmr.core.export.smplh2fbx import SMPLH2FBX

    fps = int(np.asarray(motion.get('mocap_framerate', 30)).item())
    frames = len(motion['trans'])
    if fps != 30 or frames < 2:
        raise ValueError('BIP export requires a 30 FPS clip with at least two frames')
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix='.biped-') as workspace:
        folder = Path(workspace)
        _prepare_runtime(folder)
        with SMPLH2FBX(smplh_model_path=reference_path) as converter:
            converter.convert_npz_to_fbx(motion, str(folder / 'motion.fbx'), export_mesh=False)
        _write_reference(folder)
        for stage, script in (('mapping', 'run.ms'), ('export', 'run.ms'), ('readback', 'finish.ms')):
            _batch(executable, folder, script, stage, frames, fps)
        result = folder / 'biped.bip'
        if not result.is_file() or not result.stat().st_size:
            raise ValueError('3ds Max did not write a BIP file')
        scene_file = folder / 'biped.max'
        if not scene_file.is_file() or not scene_file.stat().st_size:
            raise ValueError('3ds Max did not write the animated Biped scene')
        data = _preview(folder, frames, fps)
        encoded = folder / 'preview.json'
        encoded.write_text(json.dumps(data, separators=(',', ':')), encoding='utf-8')
        os.replace(scene_file, scene)
        os.replace(result, destination)
        os.replace(encoded, preview)


def export_bvh_scene(motion, destination, *, reference_path, executable, prepend_tpose=False, start_frame=0):
    """Import the Biped BVH into Character Studio and save the animated rig."""
    from .bvh import export_bvh

    if isinstance(motion, (str, os.PathLike)):
        with np.load(motion, allow_pickle=False) as data:
            motion = dict(data)
    fps = float(np.asarray(motion.get('mocap_framerate', 30)).item())
    if fps != 30:
        raise ValueError('BVH Max scene export requires a 30 FPS clip')
    frames = len(motion['trans']) + start_frame + int(prepend_tpose)
    with tempfile.TemporaryDirectory(dir=Path(destination).parent, prefix='.bvh-max-') as workspace:
        folder = Path(workspace)
        shutil.copyfile(RUNTIME_DIR / 'bvh_scene.ms', folder / 'bvh_scene.ms')
        export_bvh(motion, folder / 'motion.bvh', reference_path=reference_path,
                   prepend_tpose=prepend_tpose, start_frame=start_frame, units_per_meter=100)
        _batch(executable, folder, 'bvh_scene.ms', 'bvh-scene', frames, int(fps))
        scene = folder / 'biped.max'
        if not scene.is_file() or not scene.stat().st_size:
            raise ValueError('3ds Max did not write an animated BVH scene')
        os.replace(scene, destination)
