#!/usr/bin/env python3
"""Commit confirmed, cached release uploads without re-uploading weight bytes.

Uses the same commit routine as upload_large_folder, after checking cached
SHA256 and size against the release manifest. Can run while remaining distinct
shards upload; the main uploader may later repeat identical pointer commits.
"""
import argparse
import json
from pathlib import Path
import time

from huggingface_hub import HfApi
from huggingface_hub._local_folder import get_local_upload_paths, read_upload_metadata
from huggingface_hub._upload_large_folder import _commit

parser = argparse.ArgumentParser()
parser.add_argument('--repo-id', required=True)
parser.add_argument('--stage', type=Path, required=True)
parser.add_argument('--batch-size', type=int, default=20)
parser.add_argument('--watch-status', type=Path, help='Flush every 30s while this exact uploader is running')
args = parser.parse_args()
if args.batch_size < 1:
    raise ValueError('Invalid batch size')
api = HfApi()
if api.repo_info(args.repo_id, repo_type='model').private:
    raise ValueError('Expected authorized public repository')
manifest = json.loads((args.stage/'manifest.json').read_text())
def flush_once():
    ready = []
    for item in manifest['files']:
        metadata = read_upload_metadata(args.stage, item['path'])
        if metadata.is_committed or metadata.should_ignore:
            continue
        if metadata.sha256 != item['sha256'] or metadata.size != item['size']:
            continue
        if metadata.upload_mode == 'lfs' and metadata.is_uploaded:
            ready.append((get_local_upload_paths(args.stage, item['path']), metadata))
        elif metadata.upload_mode == 'regular':
            ready.append((get_local_upload_paths(args.stage, item['path']), metadata))
    if not ready:
        return
    print(json.dumps({'ready_files':len(ready), 'ready_bytes':sum(m.size for _,m in ready)}),flush=True)
    committed = 0
    for offset in range(0,len(ready),args.batch_size):
        batch = ready[offset:offset+args.batch_size]
        _commit(batch,api=api,repo_id=args.repo_id,repo_type='model',revision='main')
        remote = {x.path:x.size for x in api.get_paths_info(args.repo_id,
                  [p.path_in_repo for p,_ in batch],repo_type='model')}
        for paths, metadata in batch:
            if remote.get(paths.path_in_repo) != metadata.size:
                raise RuntimeError('Committed file size mismatch: '+paths.path_in_repo)
        committed += len(batch)
        print(json.dumps({'committed_this_flush':committed, 'total_ready':len(ready)}),flush=True)

if args.watch_status:
    print('INCREMENTAL_COMMIT_WORKER_STARTED',flush=True)
    while True:
        status = json.loads(args.watch_status.read_text())
        if status.get('repo_id') != args.repo_id:
            raise ValueError('Uploader repo identity changed')
        if status.get('phase') in ('complete','failed'):
            break
        pid = status.get('pid')
        try:
            command=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
        except OSError:
            print('Uploader no longer running; stopping commit worker',flush=True)
            break
        if 'scripts/upload_public.py' not in command or str(args.stage.resolve()) not in command:
            raise ValueError('Uploader process identity mismatch')
        flush_once()
        time.sleep(30)
else:
    flush_once()
print('FLUSH_COMPLETE',flush=True)
