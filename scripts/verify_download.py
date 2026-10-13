#!/usr/bin/env python3
"""Verify downloaded release files against manifest SHA256 values."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('bundle', type=Path)
parser.add_argument('--package', choices=['inference', 'resume', 'all'], default='inference')
args = parser.parse_args()
root = args.bundle.resolve()
manifest = json.loads((root / 'manifest.json').read_text())
if manifest.get('package') == 'inference-root' and args.package == 'resume':
    raise ValueError('This inference-only repository does not contain optimizer or resume state')
checked = skipped = 0
missing = []
for item in manifest['files']:
    path = (root / item['path']).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Unsafe manifest path')
    if not path.is_file():
        required = args.package == 'all' or (manifest.get('package') == 'inference-root' and args.package == 'inference') or (
            args.package == 'inference' and item['path'].startswith(
                ('inference_fp32/', 'qwen3vl_config_tokenizer/'))) or (
            args.package == 'resume' and item['path'].startswith('resume_dcp/'))
        if required:
            missing.append(item['path'])
        skipped += 1
        continue
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    if path.stat().st_size != item['size'] or digest.hexdigest() != item['sha256']:
        raise ValueError('Integrity mismatch: ' + item['path'])
    checked += 1
if checked == 0:
    raise ValueError('No manifest-listed model files downloaded')
if missing:
    raise FileNotFoundError(f'{len(missing)} required {args.package} files missing: {missing[:5]}')
print(f'PASS: package={args.package}; {checked} files checked; {skipped} optional files skipped.')
