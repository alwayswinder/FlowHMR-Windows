"""Route contracts for motion export selections and local template handling."""
from contextlib import ExitStack
from importlib.util import module_from_spec, spec_from_file_location
from io import BytesIO
from pathlib import Path
import struct
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np

_spec = spec_from_file_location("flowhmr_motion_export_web", Path(__file__).resolve().parents[1] / "tools/web/server.py")
server = module_from_spec(_spec)
_spec.loader.exec_module(server)


def center_samples(vmd):
    """Read the standard Center track without using the exporter's code."""
    count, = struct.unpack_from("<I", vmd, 50)
    result = {}
    for index in range(count):
        offset = 54 + index * 111
        if vmd[offset:offset + 15].split(b"\0", 1)[0].decode("cp932") == "センター":
            frame, x, y, z = struct.unpack_from("<I3f", vmd, offset + 15)
            result[frame] = (x, y, z)
    return result


class MotionExportsTest(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.temp = stack.enter_context(TemporaryDirectory())
        self.root = Path(self.temp)
        stack.enter_context(patch.object(server, "ARGS", SimpleNamespace(work_dir=self.temp)))
        stack.enter_context(patch.object(server, "MEMBER_DIRS", {}))
        single = self.root / "single"
        scene = self.root / "group" / "scene"
        single.mkdir()
        scene.mkdir(parents=True)
        self.reference = self.root / "fake-reference.npz"
        self.reference.write_bytes(b"reference")
        poses = np.zeros((3, 52, 3), dtype=np.float32)
        trans = np.zeros((3, 3), dtype=np.float32)
        betas = np.zeros(16, dtype=np.float32)
        np.savez(single / "motion.npz", poses=poses, trans=trans, betas=betas, mocap_framerate=30)
        np.savez(scene / "motion.npz", poses_1=poses, trans_1=trans, betas_1=betas,
                 poses_2=poses, trans_2=trans + [1.25, 0, 0], betas_2=betas, mocap_framerate=30)
        people = [{"person": 1, "start": 0, "end": 2}, {"person": 2, "start": 7, "end": 9}]
        stack.enter_context(patch.object(server, "JOBS", {
            "single": {"status": "done", "filename": "sample.mp4"},
            "group": {"status": "done", "filename": "duet.mp4", "person_batch": "group", "scene": {"people": people}},
        }))
        self.client = stack.enter_context(server.app.test_client())

    def request(self, method, path, **kwargs):
        response = self.client.open(path, method=method, buffered=True, **kwargs)
        try:
            return response.status_code, response.headers, response.data, response.json if response.is_json else None
        finally:
            response.close()

    def test_vmd_exports_single_and_aligned_scene_without_fbx_or_models(self):
        with patch.object(server, "model_path", side_effect=ValueError("not installed")):
            code, headers, vmd, _ = self.request("GET", "/api/jobs/single/vmd")
            self.assertEqual(code, 200)
            self.assertIn("sample_vmd.vmd", headers["Content-Disposition"])
            self.assertTrue(vmd.startswith(b"Vocaloid Motion Data 0002"))
            self.assertEqual(set(center_samples(vmd)), {0, 1, 2})
            code, headers, archive, _ = self.request("GET", "/api/jobs/group/vmd")
            self.assertEqual(code, 200)
            self.assertEqual(headers["Content-Type"], "application/zip")
            with ZipFile(BytesIO(archive)) as packed:
                self.assertEqual(packed.namelist(), ["person_01.vmd", "person_02.vmd"])
                first = center_samples(packed.read("person_01.vmd"))
                second = center_samples(packed.read("person_02.vmd"))
                self.assertEqual(set(first), {0, 1, 2})
                self.assertEqual(set(second), {7, 8, 9})
                self.assertGreater(abs(second[7][0] - first[0][0]), 10.0)
            code, headers, chosen, _ = self.request("GET", "/api/jobs/group/vmd?person=2")
            self.assertEqual(code, 200)
            self.assertIn("duet_vmd_person2.vmd", headers["Content-Disposition"])
            self.assertEqual(center_samples(chosen), second)
            code, _, _, error = self.request("GET", "/api/jobs/group/vmd?person=99")
            self.assertEqual((code, error["error_code"]), (400, "scene_people"))

    def test_biped_preview_and_both_downloads_share_one_local_conversion(self):
        executable = self.root / '3dsmaxbatch.exe'
        executable.write_bytes(b'max')

        def convert(motion, output, preview, scene, **options):
            self.assertEqual(len(motion['poses']), 3)
            self.assertEqual(options['executable'], executable)
            output.parent.mkdir(exist_ok=True)
            output.write_bytes(b'BIP fixture')
            preview.write_text('{"frames":3}', encoding='utf-8')
            scene.write_bytes(b'MAX fixture')

        with (patch.object(server, 'fbx_export_available', return_value=True),
              patch.object(server, 'model_path', return_value=self.reference),
              patch('_MyTools.export.bip.max_batch_path', return_value=executable),
              patch('_MyTools.export.bip.export_bip', side_effect=convert) as export):
            status, _, _, data = self.request('GET', '/api/jobs/single/bip-preview')
            self.assertEqual((status, data), (200, {'frames': 3}))
            status, headers, body, _ = self.request('GET', '/api/jobs/single/bip')
            self.assertEqual((status, body), (200, b'BIP fixture'))
            self.assertIn('sample_biped.bip', headers['Content-Disposition'])
            status, headers, body, _ = self.request('GET', '/api/jobs/single/bip?format=max')
            self.assertEqual((status, body), (200, b'MAX fixture'))
            self.assertIn('sample_biped.max', headers['Content-Disposition'])
            self.assertEqual(self.request('GET', '/api/jobs/group/bip')[0], 409)
            self.assertEqual(self.request('GET', '/api/jobs/single/bip', base_url='http://example.com')[0], 403)
            export.assert_called_once()

    def test_bvh_max_uses_character_studio_export_with_the_selected_options(self):
        executable = self.root / '3dsmaxbatch.exe'
        executable.touch()

        def write(motion, path, *, start_frame, prepend_tpose, **options):
            self.assertEqual(options['executable'], executable)
            Path(path).write_bytes(f'BVH MAX:{start_frame}:{prepend_tpose}'.encode())

        with (patch.object(server, 'model_path', return_value=self.reference),
              patch('_MyTools.export.bip.max_batch_path', return_value=executable),
              patch('_MyTools.export.bip.export_bvh_scene', side_effect=write) as export):
            url = '/api/jobs/single/bvh?format=max&prepend_tpose=1'
            for _ in range(2):
                code, headers, body, _ = self.request('GET', url)
                self.assertEqual((code, body), (200, b'BVH MAX:0:True'))
                self.assertIn('sample_biped_bvh.max', headers['Content-Disposition'])
            export.assert_called_once()
            self.assertEqual(self.request('GET', url, base_url='http://example.com')[0], 403)
            code, headers, body, _ = self.request('GET', '/api/jobs/group/bvh?format=max')
            self.assertEqual(code, 200)
            self.assertIn('duet_biped_bvh_max_scene.zip', headers['Content-Disposition'])
            with ZipFile(BytesIO(body)) as archive:
                self.assertEqual(archive.namelist(), ['person_01.max', 'person_02.max'])
                self.assertEqual(archive.read('person_02.max'), b'BVH MAX:7:False')

    def test_bvh_selection_options_archives_and_cache_are_independent(self):
        calls = []

        def write(motion, path, **options):
            calls.append(options)
            Path(path).write_bytes(f"BVH:{options['start_frame']}:{options['prepend_tpose']}".encode())

        with (patch.object(server, 'model_path', return_value=self.reference),
              patch('_MyTools.export.bvh.export_bvh', side_effect=write)):
            for option in ('0', '0', '1'):
                code, headers, data, _ = self.request('GET', '/api/jobs/single/bvh?prepend_tpose=' + option)
                self.assertEqual(code, 200)
                self.assertIn('sample_biped_bvh.bvh', headers['Content-Disposition'])
                self.assertEqual(data, f'BVH:0:{option == "1"}'.encode())
            self.assertEqual(len(calls), 2)
            code, headers, data, _ = self.request('GET', '/api/jobs/group/bvh')
            self.assertEqual((code, headers['Content-Type']), (200, 'application/zip'))
            with ZipFile(BytesIO(data)) as archive:
                self.assertEqual(archive.namelist(), ['person_01.bvh', 'person_02.bvh'])
                self.assertEqual(archive.read('person_02.bvh'), b'BVH:7:False')
            code, headers, data, _ = self.request('GET', '/api/jobs/group/bvh?person=2')
            self.assertEqual(code, 200)
            self.assertIn('duet_biped_bvh_person2.bvh', headers['Content-Disposition'])
            self.assertEqual(data, b'BVH:7:False')
            code, _, _, error = self.request('GET', '/api/jobs/group/bvh?person=99')
            self.assertEqual((code, error['error_code']), (400, 'scene_people'))
            code, _, _, error = self.request('GET', '/api/jobs/single/bvh?prepend_tpose=2')
            self.assertEqual((code, error['error_code']), (400, 'bvh_options'))

    def test_metahuman_validates_template_and_selection_before_export(self):
        query = "?profile=metahuman&mesh=0"
        base = "/api/jobs/single/fbx" + query
        with patch.object(server, "find_spec", return_value=object()), patch.object(server, "model_path", return_value=self.reference):
            code, _, _, error = self.request("GET", base)
            self.assertEqual((code, error["error_code"]), (400, "metahuman_template_required"))
            code, _, _, error = self.request("POST", base, headers={"Origin": "https://example.invalid"})
            self.assertEqual((code, error["error_code"]), (403, "metahuman_local_only"))
            code, _, _, error = self.request("POST", "/api/jobs/group/fbx" + query)
            self.assertEqual((code, error["error_code"]), (400, "metahuman_export_options"))
            code, _, _, error = self.request("POST", base, data={"template": (BytesIO(b"content"), "avatar.pmx")})
            self.assertEqual((code, error["error_code"]), (400, "metahuman_template_required"))
            with patch.object(server, "FBX_TEMPLATE_LIMIT", 7):
                code, _, _, error = self.request("POST", base, data={"template": (BytesIO(b"x" * 8), "body.fbx")})
            self.assertEqual((code, error["error_code"]), (413, "metahuman_template_size"))

    def test_fbx_profiles_and_template_bytes_use_independent_cached_outputs(self):
        calls = []

        class TestConverter:
            def __init__(self, smplh_model_path):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

            def convert_npz_to_fbx(self, source, outname, *, profile, template_path, **options):
                value = Path(template_path).read_bytes() if template_path else b""
                calls.append((profile, value, Path(template_path) if template_path else None))
                path = Path(outname)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(profile.encode() + value)

        fake = SimpleNamespace(SMPLH2FBX=TestConverter)
        with (patch.object(server, "find_spec", return_value=object()),
              patch.object(server, "model_path", return_value=self.reference),
              patch.dict(sys.modules, {"flowhmr.core.export.smplh2fbx": fake})):
            base = "/api/jobs/single/fbx?mesh=0"
            for profile in ("flowhmr", "mixamo"):
                code, headers, data, _ = self.request("GET", base + "&profile=" + profile)
                self.assertEqual(code, 200)
                self.assertEqual(data, profile.encode())
                self.assertIn(f"sample_{profile}_skeleton.fbx", headers["Content-Disposition"])
            for content in (b"A", b"A", b"B"):
                code, _, data, _ = self.request("POST", base + "&profile=metahuman",
                                                data={"template": (BytesIO(content), "body.fbx")})
                self.assertEqual(code, 200)
                self.assertEqual(data, b"metahuman" + content)
            self.assertEqual([(profile, data) for profile, data, _ in calls],
                             [("flowhmr", b""), ("mixamo", b""), ("metahuman", b"A"), ("metahuman", b"B")])
            self.assertTrue(all(not path.exists() for _, _, path in calls if path is not None))


if __name__ == "__main__":
    unittest.main()
