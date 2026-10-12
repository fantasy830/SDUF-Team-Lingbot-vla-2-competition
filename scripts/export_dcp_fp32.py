#!/usr/bin/env python3
"""CPU-only DCP model export, preserving and verifying every FP32 parameter.

Uses the same PyTorch DCP empty-state planner as LingBot's dcp_to_torch_state_dict.
No optimizer loading, GPU allocation, or dtype conversion.
"""
import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import shutil

import torch
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict
from safetensors import safe_open
from safetensors.torch import save_file


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--shard-bytes', type=int, default=5_000_000_000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite ' + str(args.output))
    if args.shard_bytes <= 0 or not (args.checkpoint / 'model' / '.metadata').is_file():
        raise ValueError('Invalid shard size or incomplete DCP model directory')
    metadata = FileSystemReader(args.checkpoint / 'model').read_metadata()
    if any(v.properties.dtype != torch.float32 for v in metadata.state_dict_metadata.values()):
        raise ValueError('This release expects source model tensors to be FP32')
    state = {}
    print('Loading DCP model on CPU; source files are read-only', flush=True)
    _load_state_dict(state, storage_reader=FileSystemReader(args.checkpoint / 'model'),
                     planner=_EmptyStateDictLoadPlanner(), no_dist=True)
    if 'state' in state:
        state = state['state']
    state = state['model']
    if len(state) != len(metadata.state_dict_metadata):
        raise ValueError('Unexpected model entry count after DCP reconstruction')
    if any(not isinstance(t, torch.Tensor) or t.dtype != torch.float32 for t in state.values()):
        raise ValueError('Unexpected non-FP32 model entry')
    total_bytes = sum(t.numel() * t.element_size() for t in state.values())
    shards = []
    current, size = [], 0
    for key, value in state.items():
        nbytes = value.numel() * value.element_size()
        if current and size + nbytes > args.shard_bytes:
            shards.append(current)
            current, size = [], 0
        current.append(key)
        size += nbytes
    if current:
        shards.append(current)
    args.output.mkdir(parents=True)
    weight_map, files = {}, []
    for index, keys in enumerate(shards, 1):
        name = f'model-{index:05d}-of-{len(shards):05d}.safetensors'
        payload = OrderedDict((k, state[k].detach().cpu().contiguous()) for k in keys)
        print(f'Saving and checking {name}: {len(keys)} tensors', flush=True)
        save_file(payload, args.output / name, metadata={'format': 'pt'})
        with safe_open(args.output / name, framework='pt', device='cpu') as saved:
            if set(saved.keys()) != set(keys):
                raise ValueError('Serialized key mismatch')
            for key in keys:
                value = saved.get_tensor(key)
                if value.dtype != state[key].dtype or value.shape != state[key].shape:
                    raise ValueError('dtype/shape mismatch: ' + key)
                if not torch.equal(value, state[key]):
                    raise ValueError('Exact tensor value mismatch: ' + key)
                weight_map[key] = name
        files.append({'file': name, 'size': (args.output / name).stat().st_size,
                      'sha256': sha256(args.output / name)})
        del payload
    index = {'metadata': {'total_size': total_bytes}, 'weight_map': weight_map}
    (args.output / 'model.safetensors.index.json').write_text(json.dumps(index, indent=2) + '\n')
    for source in args.assets.iterdir():
        if source.is_file() and not source.name.endswith(('.safetensors', '.bin', '.index.json')):
            shutil.copy2(source, args.output / source.name)
    report = {'global_step': 10000, 'tensor_count': len(state),
              'tensor_elements': sum(t.numel() for t in state.values()),
              'tensor_bytes': total_bytes, 'dtype': 'float32',
              'source_format': 'pytorch_dcp', 'exact_tensor_equality_verified': True,
              'files': files}
    (args.output.parent / 'export_verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(8)
    main()
