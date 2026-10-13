#!/usr/bin/env python3
"""Read local resumable-upload metadata. No network calls or credential access."""
import argparse
import json
from pathlib import Path
import re

parser = argparse.ArgumentParser()
parser.add_argument('--stage', type=Path, required=True)
parser.add_argument('--status-json', type=Path, required=True)
parser.add_argument('--log', type=Path)
args = parser.parse_args()
status = json.loads(args.status_json.read_text()) if args.status_json.is_file() else {'phase':'starting'}
print('Phase:',status.get('phase'))
if status.get('transport'):
    print('Transport:',status['transport'])
if status.get('error'):
    print('Error:', status['error'])
manifest = json.loads((args.stage/'manifest.json').read_text())
cache = args.stage / '.cache/huggingface/upload'
committed = uploaded = hashed = 0
committed_bytes = uploaded_bytes = 0
for item in manifest['files']:
    path = cache / (item['path'] + '.metadata')
    try:
        lines = path.read_text().splitlines()
        if len(lines) < 8:
            continue
        hashed += int(bool(lines[3]))
        if lines[6] == '1':
            uploaded += 1
            uploaded_bytes += item['size']
        if lines[7] == '1':
            committed += 1
            committed_bytes += item['size']
    except (FileNotFoundError, OSError):
        continue
total = manifest['total_bytes']
print(f'Files hashed: {hashed}/{len(manifest["files"])}')
print(f'Files uploaded: {uploaded}; committed: {committed}')
print(f'Committed logical file size: {committed_bytes/1e9:.3f}/{total/1e9:.3f} GB ({committed_bytes/total:.1%})')
print('This counts complete files; an in-progress large file is not included yet. Xet wire bytes can differ.')
if args.log and args.log.is_file():
    with args.log.open('rb') as stream:
        stream.seek(max(0,args.log.stat().st_size-16000))
        tail=stream.read().decode('utf-8',errors='replace')
    clean=re.sub(r'\x1b\[[0-9;?]*[A-Za-z]','',tail).replace('\r','\n')
    lines=[line.strip() for line in clean.splitlines() if 'Processing Files' in line or 'New Data Upload' in line]
    print('Latest transfer snapshot (possibly one worker):')
    print('\n'.join(lines[-2:]) or '(no transfer sample)')
