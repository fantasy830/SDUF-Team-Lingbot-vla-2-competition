#!/usr/bin/env python3
"""Read local resumable-upload metadata. No network calls or credential access."""
import argparse
import json
import os
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
print(f'Confirmed pre-uploaded logical file size: {uploaded_bytes/1e9:.3f}/{total/1e9:.3f} GB ({uploaded_bytes/total:.1%})')
print(f'Committed logical file size: {committed_bytes/1e9:.3f}/{total/1e9:.3f} GB ({committed_bytes/total:.1%})')
print('This counts complete files; an in-progress large file is not included yet. Xet wire bytes can differ.')
pid = status.get('pid')
if isinstance(pid,int):
    base = Path(f'/proc/{pid}')
    try:
        command=(base/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        if 'scripts/upload_public.py' in command and str(args.stage.resolve()) in command:
            active=[]
            for fd in (base/'fd').iterdir():
                try:
                    target=Path(os.readlink(fd))
                    if not target.is_relative_to(args.stage.resolve()) or target.suffix not in ('.distcp','.safetensors'):
                        continue
                    info=(base/'fdinfo'/fd.name).read_text()
                    position=int(next(line.split(':',1)[1] for line in info.splitlines() if line.startswith('pos:')))
                    size=target.stat().st_size
                    active.append(f'  {target.name}: {position/1e6:.1f}/{size/1e6:.1f} MB read')
                except (OSError,StopIteration):
                    continue
            if active:
                print('Active file reads (can include hashing/buffering; not a network byte count):')
                print('\n'.join(active))
    except OSError:
        pass
if args.log and args.log.is_file():
    with args.log.open('rb') as stream:
        stream.seek(max(0,args.log.stat().st_size-16000))
        tail=stream.read().decode('utf-8',errors='replace')
    clean=re.sub(r'\x1b\[[0-9;?]*[A-Za-z]','',tail).replace('\r','\n')
    lines=[line.strip() for line in clean.splitlines() if 'Processing Files' in line or 'New Data Upload' in line]
    print('Latest transfer snapshot (possibly one worker):')
    print('\n'.join(lines[-2:]) or '(no transfer sample)')
