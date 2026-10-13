"""Lightweight checks for old nested bundles and new root-format releases."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ReleaseHelpers(unittest.TestCase):
    def run_script(self, name, *args):
        return subprocess.run([sys.executable, str(ROOT / 'scripts' / name), *map(str, args)],
                              capture_output=True, text=True)

    def test_runtime_supports_both_layouts(self):
        for flat in (False, True):
            with self.subTest(flat=flat), tempfile.TemporaryDirectory() as temporary:
                base = Path(temporary)
                bundle, source = base / 'bundle', base / 'source'
                assets = bundle if flat else bundle / 'inference_fp32'
                model = bundle if flat else assets / 'hf_ckpt'
                model.mkdir(parents=True)
                (model / 'model.safetensors.index.json').write_text(json.dumps(
                    {'weight_map': {'test.weight': 'model-00001-of-00001.safetensors'}}))
                (model / 'model-00001-of-00001.safetensors').write_bytes(b'test fixture only')
                (bundle / 'qwen3vl_config_tokenizer').mkdir()
                (bundle / 'qwen3vl_config_tokenizer/config.json').write_text('{}')
                (assets / 'lingbotvla_cli.yaml').write_text('data: {}\nmodel: {}\ntrain: {}\n')
                (assets / 'robot_configs').mkdir()
                (assets / 'robot_configs/robotwin.yaml').write_text('norm_stats: old.json\n')
                (assets / 'norm_stats').mkdir()
                (assets / 'norm_stats/robotwin.json').write_text('{}')
                (source / 'deploy').mkdir(parents=True)
                (source / 'deploy/lingbot_vla_v2_policy.py').write_text('# fixture\n')
                output = base / 'runtime'
                result = self.run_script('configure_runtime.py', '--bundle', bundle,
                                         '--source-root', source, '--output', output)
                self.assertEqual(result.returncode, 0, result.stderr)
                link = output / 'checkpoints/global_step_10000/hf_ckpt/model-00001-of-00001.safetensors'
                self.assertEqual(link.resolve(), model / link.name)
                (model / 'model-00001-of-00001.safetensors').unlink()
                result = self.run_script('configure_runtime.py', '--bundle', bundle,
                                         '--source-root', source, '--output', base / 'missing')
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('Missing or unsafe model shard', result.stderr)

    def test_flat_manifest_requires_all_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            (base / 'config.json').write_bytes(b'{}')
            records = [{'path': name, 'size': 2, 'sha256': hashlib.sha256(b'{}').hexdigest()}
                       for name in ('config.json', 'required.safetensors')]
            (base / 'manifest.json').write_text(json.dumps({'package': 'inference-root', 'files': records}))
            result = self.run_script('verify_download.py', base)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('required inference files missing', result.stderr)
            (base / 'required.safetensors').write_bytes(b'{}')
            result = self.run_script('verify_download.py', base)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = self.run_script('verify_download.py', base, '--package', 'resume')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('does not contain optimizer', result.stderr)


if __name__ == '__main__':
    unittest.main()
