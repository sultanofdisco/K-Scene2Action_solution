import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from test_filter import ROOT, inputs


class CliTests(unittest.TestCase):
    def run_plan(self, raw, expected, format='normalized'):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'plan.json'
            path.write_text(raw, encoding='utf-8')
            proc = subprocess.run([sys.executable, '-m', 'f2', '--plan', str(path),
                                   '--context', str(ROOT / 'examples/context.json'), '--format', format],
                                  cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
        self.assertEqual(proc.returncode, {'PASS': 0, 'BLOCK': 2, 'DEFER': 3}[expected], proc.stderr)
        result = json.loads(proc.stdout)
        self.assertEqual(result['decision'], expected)
        if expected != 'PASS':
            self.assertIsNone(result['approved_plan'])

    def test_duplicate_keys_rejected(self):
        self.run_plan('{"actions": [], "actions": []}', 'DEFER')

    def test_nan_invalid_json_and_oversized_inputs_rejected(self):
        for raw in ['NaN', '{bad json', '"' + 'x' * 1048577 + '"']:
            self.run_plan(raw, 'DEFER')

    def test_block_returns_nonzero_exit_and_no_executable_plan(self):
        plan, _ = inputs()
        plan['actions'][0]['target_id'] = 'blue_ball'
        self.run_plan(json.dumps(plan), 'BLOCK')

    def test_native_pass_cli(self):
        self.run_plan((ROOT / 'examples/vlm_steps_pass.json').read_text(encoding='utf-8'), 'PASS', 'steps')


if __name__ == '__main__':
    unittest.main()
