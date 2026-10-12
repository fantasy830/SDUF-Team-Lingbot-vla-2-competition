#!/usr/bin/env python3
"""Verify downloaded release files against manifest SHA256 values."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('bundle', type=Path)
args = parser.parse_args()
root = args.bundle.resolve()
manifest = json.loads((root / 'manifest.json').read_text())
checked = skipped = 0
for item in manifest['files']:
    path = (root / item['path']).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Unsafe manifest path')
    if not path.is_file():
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
print(f'PASS: {checked} present files checked; {skipped} absent files skipped for selective download.')
print('For a complete package, ensure no required files in that package are absent.')
