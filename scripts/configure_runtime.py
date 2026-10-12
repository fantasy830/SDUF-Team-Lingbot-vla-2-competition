#!/usr/bin/env python3
"""Create local paths for a downloaded 10k inference bundle, without touching it."""
import argparse
from pathlib import Path
import shutil
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    bundle, source, output = args.bundle.resolve(), args.source_root.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError('Choose a new output directory: ' + str(output))
    model = bundle / 'inference_fp32/hf_ckpt'
    qwen = bundle / 'qwen3vl_config_tokenizer'
    if not (model / 'model.safetensors.index.json').is_file() or not (qwen / 'config.json').is_file():
        raise FileNotFoundError('Download inference_fp32 and qwen3vl_config_tokenizer first')
    if not (source / 'deploy/lingbot_vla_v2_policy.py').is_file():
        raise FileNotFoundError('Invalid LingBot source root')
    config = yaml.safe_load((bundle / 'inference_fp32/lingbotvla_cli.yaml').read_text())
    output.mkdir(parents=True)
    shutil.copytree(bundle / 'inference_fp32/robot_configs', output / 'robot_configs')
    shutil.copytree(bundle / 'inference_fp32/norm_stats', output / 'norm_stats')
    norm = output / 'norm_stats/robotwin.json'
    robot_path = output / 'robot_configs/robotwin.yaml'
    robot = yaml.safe_load(robot_path.read_text())
    robot['norm_stats'] = str(norm)
    robot_path.write_text(yaml.safe_dump(robot, sort_keys=False))
    config['data']['norm_stats_file'] = str(norm)
    config['data']['robot_config_root'] = str(output / 'robot_configs')
    config['model'].update(model_path=str(model), config_path=str(model), tokenizer_path=str(qwen))
    config['train']['output_dir'] = str(output / 'training_output')
    (output / 'lingbotvla_cli.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    runtime_model = output / 'checkpoints/global_step_10000/hf_ckpt'
    runtime_model.mkdir(parents=True)
    for item in model.iterdir():
        if item.is_file():
            (runtime_model / item.name).symlink_to(item)
    original_robot = source / 'configs/robot_configs/robotwin.yaml'
    matches = original_robot.is_file() and yaml.safe_load(original_robot.read_text()) == yaml.safe_load(
        (bundle / 'inference_fp32/robot_configs/robotwin.yaml').read_text())
    print('Runtime config:', output / 'lingbotvla_cli.yaml')
    print('QWEN3VL_PATH:', qwen)
    print('Runtime model path:', runtime_model)
    print('Source robot config matches archived mapping:', matches)
    print('Confirm the deployment implementation reads these paths before evaluating.')


if __name__ == '__main__':
    main()
