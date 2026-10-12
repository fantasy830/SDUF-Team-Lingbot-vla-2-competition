#!/usr/bin/env python3
"""Resumable public release upload. Does not change existing repo visibility.

Upload original resume files first while CPU export runs, then upload the
verified FP32 package. Check remote file sizes against the completed manifest.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo-id', required=True)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--status-json', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    api = HfApi()
    info = api.repo_info(args.repo_id, repo_type='model')
    if info.private:
        raise ValueError('Expected an already-authorized public repository')

    def status(phase, **extra):
        value = {'phase': phase, 'repo_id': args.repo_id,
                 'updated_utc': datetime.now(timezone.utc).isoformat(), **extra}
        args.status_json.parent.mkdir(parents=True, exist_ok=True)
        args.status_json.write_text(json.dumps(value, indent=2) + '\n')
        print(json.dumps(value), flush=True)

    try:
        status('uploading_resume')
        api.upload_large_folder(args.repo_id, folder_path=args.stage, repo_type='model',
                                allow_patterns=['resume_dcp/*', 'qwen3vl_config_tokenizer/*',
                                                'README.md', 'provenance.json'],
                                num_workers=args.workers, print_report_every=60)
        status('waiting_for_verified_export')
        manifest_path = args.stage / 'manifest.json'
        deadline = time.monotonic() + 6 * 3600
        while not manifest_path.is_file():
            if time.monotonic() > deadline:
                raise TimeoutError('No complete export manifest after six hours; inspect export log')
            time.sleep(15)
        manifest = json.loads(manifest_path.read_text())
        if not manifest.get('exact_tensor_equality_verified'):
            raise ValueError('Missing export equality verification')
        status('uploading_verified_inference')
        api.upload_large_folder(args.repo_id, folder_path=args.stage, repo_type='model',
                                allow_patterns=['inference_fp32/*', 'manifest.json',
                                                'README.md', 'provenance.json'],
                                num_workers=args.workers, print_report_every=60)
        status('checking_remote_file_sizes')
        remote = {x.path: x.size for x in api.list_repo_tree(args.repo_id, repo_type='model',
                                                          recursive=True) if hasattr(x, 'size')}
        for item in manifest['files']:
            if remote.get(item['path']) != item['size']:
                raise ValueError('Remote file missing/size mismatch: ' + item['path'])
        if 'manifest.json' not in remote:
            raise ValueError('Remote manifest missing')
        status('complete', file_count=len(manifest['files']),
               verified_remote_bytes=sum(x['size'] for x in manifest['files']),
               revision=api.repo_info(args.repo_id, repo_type='model').sha)
        print('PUBLIC_UPLOAD_COMPLETE', flush=True)
    except Exception as error:
        status('failed', error_type=type(error).__name__, error=str(error))
        raise


if __name__ == '__main__':
    main()
