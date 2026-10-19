"""Nonuniform exact-solution refinement, distinct from a homogeneous patch."""

import json
from pathlib import Path
import runpy
import sys


def test_augmented_plane_nonuniform_manufactured_solution(monkeypatch, tmp_path):
    script = Path(__file__).resolve().parents[1] / 'tools/verify_normal_augmentation_spatial.py'
    output = tmp_path / 'spatial.json'
    monkeypatch.setattr(sys, 'argv', [str(script), '--cells', '2', '4', '--output', str(output)])
    runpy.run_path(str(script), run_name='__main__')
    report = json.loads(output.read_text())
    assert report['verification_passed']
    assert len(report['records']) == 2
    assert report['records'][1]['force_error'] / abs(report['records'][1]['exact_force_x']) < 1e-3
