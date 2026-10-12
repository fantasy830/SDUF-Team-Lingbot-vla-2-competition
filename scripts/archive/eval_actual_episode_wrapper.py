#!/usr/bin/env python3
"""Isolated diagnostic wrapper: count completed policy rollouts, not accepted seeds.

The original RoboTwin file is imported read-only. Only the in-memory loop
condition is patched; scene generation, expert checks and policy are unchanged.
"""
import importlib.util
import inspect
import os
from pathlib import Path
import sys


def corrected_source(source):
    old = "while succ_seed < test_num:"
    if source.count(old) != 1:
        raise RuntimeError("Evaluator version changed: expected exactly one seed-count loop")
    source = source.replace(old, "while task_env.test_num < test_num:")
    insertion = "    while task_env.test_num < test_num:\n"
    source = source.replace(insertion, insertion +
        "        if now_seed - st_seed >= max(50, test_num * 20):\n"
        "            raise RuntimeError('Actual-policy episode target not reached within bounded seed attempts')\n")
    record_at = "        task_env.test_num += 1\n"
    if source.count(record_at) != 1:
        raise RuntimeError("Evaluator version changed: ambiguous policy result position")
    source = source.replace(record_at, record_at +
        "        print('ACTUAL_POLICY_RESULT ' + json.dumps({'task': task_name, 'setting': args['task_config'], 'seed': now_seed, 'success': bool(succ), 'episode': task_env.test_num, 'instruction': instruction}), flush=True)\n")
    instruction_at = "        instruction = build_instruction(args, episode_info, instruction_type, test_num)\n"
    if source.count(instruction_at) != 1:
        raise RuntimeError("Evaluator version changed: instruction generation anchor missing")
    source = source.replace(instruction_at,
        "        import random as _instruction_random\n"
        "        _instruction_py_state = _instruction_random.getstate()\n"
        "        _instruction_np_state = np.random.get_state()\n"
        "        _instruction_random.seed(now_seed)\n"
        "        np.random.seed(now_seed)\n" + instruction_at +
        "        _instruction_random.setstate(_instruction_py_state)\n"
        "        np.random.set_state(_instruction_np_state)\n")
    return source


def main():
    root = Path(os.environ.get("ROBOTWIN_ROOT", "/RoboTwin"))
    path = root / "scripts/eval_policy_xpolicylab.py"
    for entry in (root, root / "scripts"):
        sys.path.insert(0, str(entry))
    spec = importlib.util.spec_from_file_location("actual_episode_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = corrected_source(inspect.getsource(module.eval_remote_policy))
    exec(compile(source, str(path) + ":actual_policy_count", "exec"), module.__dict__)
    print("ACTUAL_POLICY_COUNT_FIX enabled; original evaluator file unchanged", flush=True)
    from test_render import Sapien_TEST
    Sapien_TEST()
    args = module.parse_args()
    if module.parse_bool(args.get("eval_batch", False)):
        raise ValueError("This diagnostic wrapper only supports non-batch evaluation")
    module.main(args)


if __name__ == "__main__":
    main()
