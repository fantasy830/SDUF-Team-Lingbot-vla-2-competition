"""CPU-only regression checks for the isolated diagnostic patch."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import random
from types import SimpleNamespace
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location('wrapper', Path(__file__).with_name('eval_actual_episode_wrapper.py'))
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)

SOURCE = '''def eval_remote_policy(task_env, test_num):
    succ_seed = 0
    st_seed = 0
    now_seed = 0
    task_name = 'synthetic'
    args = {'task_config': 'demo_clean'}
    episode_info = {}
    instruction_type = 'seen'
    succ = False
    while succ_seed < test_num:
        succ_seed += 1
        if now_seed < 2:
            now_seed += 1
            continue
        instruction = build_instruction(args, episode_info, instruction_type, test_num)
        task_env.test_num += 1
        now_seed += 1
    return task_env.test_num, now_seed
'''


class PatchTests(unittest.TestCase):
    def test_skips_and_reproducible_instructions(self):
        namespace = dict(json=json, np=np, build_instruction=lambda *args:
                         random.choice(['a', 'b', 'c']) + str(np.random.randint(100)))
        exec(wrapper.corrected_source(SOURCE), namespace)
        outputs = []
        for seed in [5, 91]:
            random.seed(seed)
            np.random.seed(seed)
            before_python, before_numpy = random.getstate(), np.random.get_state()
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                result = namespace['eval_remote_policy'](SimpleNamespace(test_num=0), 3)
            self.assertEqual(result, (3, 5))
            self.assertEqual(before_python, random.getstate())
            after_numpy = np.random.get_state()
            self.assertTrue(np.array_equal(before_numpy[1], after_numpy[1]))
            self.assertEqual(before_numpy[2:], after_numpy[2:])
            outputs.append(stream.getvalue())
        self.assertEqual(outputs[0], outputs[1])

    def test_version_change_rejected(self):
        with self.assertRaises(RuntimeError):
            wrapper.corrected_source(SOURCE.replace('while succ_seed < test_num:', 'while False:'))


if __name__ == '__main__':
    unittest.main()
