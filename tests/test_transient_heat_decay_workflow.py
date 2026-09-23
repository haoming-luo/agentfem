"""End-to-end public workflow: initial field, constraints, solve, history and output."""
import importlib.util
from pathlib import Path
import json
import pytest
s = importlib.util.spec_from_file_location('heat_decay_example',
    Path(__file__).resolve().parents[1]/'examples/transient_heat_decay.py')
example = importlib.util.module_from_spec(s)
s.loader.exec_module(example)

@pytest.mark.parametrize('dimension', [2,3])
def test_heat_decay_converges_and_exports(dimension, tmp_path):
    coarse = example.run(dimension=dimension, cells=4, steps=8)
    output = tmp_path/f'heat_{dimension}d.xdmf'
    fine = example.run(dimension=dimension, cells=12, steps=72, output=output)
    assert coarse['status'] == fine['status'] == 'completed'
    assert fine['relative_l2_error'] < 0.65 * coarse['relative_l2_error']
    assert fine['relative_l2_error'] < 0.04
    assert 300 < fine['center_temperature'] < 301
    assert output.is_file()
    manifest = json.loads(output.with_suffix('.result.json').read_text())
    assert 'center_temperature' in manifest['histories']
