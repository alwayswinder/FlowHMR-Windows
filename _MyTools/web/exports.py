"""Cache Biped exports and serialize local 3ds Max jobs."""
import hashlib
import json
from pathlib import Path
from threading import Lock

import numpy as np

from _MyTools.export import bip, bvh


MAX_EXPORT_LOCK = Lock()
BVH_EXPORT_LOCK = Lock()


def cached_bvh(source, chosen, signature, references, prepend, output_format,
               executable, file_stamp, write_motion):
    extension = "zip" if chosen is not None and len(chosen) > 1 else output_format
    stamp = (file_stamp(source), signature, prepend,
             [(number, str(model), file_stamp(model)) for number, model in references.items()])
    if executable:
        stamp += ("max-v1", str(executable), file_stamp(executable))
    key = hashlib.sha256(json.dumps(stamp, separators=(",", ":")).encode()).hexdigest()[:24]
    path = Path(source).parent / "exports" / f"bvh_biped_v3_{key}.{extension}"

    def write(motion, output, person):
        number = person["person"] if person else None
        options = dict(reference_path=references[number], start_frame=person["start"] if person else 0,
                       prepend_tpose=prepend == "1")
        if executable:
            bip.export_bvh_scene(motion, output, executable=executable, **options)
        else:
            bvh.export_bvh(motion, output, **options)

    with (MAX_EXPORT_LOCK if executable else BVH_EXPORT_LOCK):
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            write_motion(source, str(path), chosen, output_format, write)
    return str(path), extension


def cached_bip(source, reference, executable, file_stamp):
    stamp = (file_stamp(source), file_stamp(reference), str(executable), file_stamp(executable))
    key = hashlib.sha256(repr(stamp).encode()).hexdigest()[:24]
    path = Path(source).parent / "exports" / f"biped_{bip.REVISION}_{key}.bip"
    preview, scene = path.with_suffix(".json"), path.with_suffix(".max")
    with MAX_EXPORT_LOCK:
        if not all(output.is_file() for output in (path, preview, scene)):
            with np.load(source, allow_pickle=False) as data:
                bip.export_bip(dict(data), path, preview, scene, reference_path=reference, executable=executable)
    return path, preview, scene
