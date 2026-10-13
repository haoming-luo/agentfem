"""Installed public example: analytical linear limit and refinement evidence."""
import importlib.util
from pathlib import Path


def test_free_bar_refinement(tmp_path):
    path = Path(__file__).resolve().parents[1] / 'examples/elastic_bar_wave.py'
    spec = importlib.util.spec_from_file_location('bar_wave_example', path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    coarse = example.run(tmp_path/'coarse', cells=20, backend='auto')
    fine = example.run(tmp_path/'fine', cells=40, backend='auto')
    assert coarse['completed'] and fine['completed']
    assert fine['relative_displacement_error'] < .4*coarse['relative_displacement_error']
    assert fine['linear_limit_check_passed']
    assert not fine['independent_large_strain_validation']
    assert (tmp_path/'fine'/'fields.h5').is_file()
