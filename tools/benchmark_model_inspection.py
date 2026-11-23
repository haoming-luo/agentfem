# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded, paired metadata-owner timings; never a FEM-solve speed claim."""
import argparse
from contextlib import contextmanager
import gc
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import tempfile
from time import perf_counter
import tracemalloc

from mpi4py import MPI
from agentfem import checkpointing, constitutive, fields, mesh, models, studies
from agentfem import _model_review, _model_settings


ROOT = Path(__file__).resolve().parents[1]


def baseline(ref, name, directory):
    if not re.fullmatch('[0-9a-fA-F]{7,40}', ref):
        raise ValueError('Use a recorded hexadecimal Git commit, not a moving branch name.')
    content = subprocess.check_output(['git', 'show', f'{ref}:src/agentfem/{name}.py'], cwd=ROOT)
    path = directory / (name + '.py')
    path.write_bytes(content)
    spec = importlib.util.spec_from_file_location('agentfem._baseline' + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build(n):
    domain = mesh.rectangle((0.,0.), (2.,1.), (n,n//2), comm=MPI.COMM_SELF)
    model = models.create(mesh=domain, study=studies.static_solid(dimension=2, assumption='plane_stress'))
    u = model.field(fields.displacement(domain))
    model.material(constitutive.isotropic_elastic(young=1000., poisson=.3, density=1.))
    model.clamp(u, on=mesh.face(domain, axis='x', value=0., tag=1, name='left'))
    model.traction((2.,0.), on=mesh.face(domain, axis='x', value=2., tag=2, name='right'))
    return model, u


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-ref', required=True)
    parser.add_argument('--sizes', nargs='+', type=int, default=[64,128,256])
    parser.add_argument('--pairs', type=int, default=5)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if MPI.COMM_WORLD.size != 1 or args.pairs < 3 or any(n < 4 or n % 2 for n in args.sizes):
        parser.error('Use one process, at least three pairs and even grid sizes >= 4.')
    report = {'baseline_ref': args.baseline_ref, 'python': sys.version, 'platform': platform.platform(),
        'boundary': 'metadata owner functions with shared current dependencies; warmed JIT; no solve, output or Python startup',
        'source_sha256': {m.__name__: sha256(Path(m.__file__).read_bytes()).hexdigest()
                          for m in (_model_review, _model_settings, checkpointing)}, 'sizes': []}
    with tempfile.TemporaryDirectory(prefix='agentfem-inspection-baseline-') as directory:
        old_settings = baseline(args.baseline_ref, '_model_settings', Path(directory))
        old_review = baseline(args.baseline_ref, '_model_review', Path(directory))
        old_checkpointing = baseline(args.baseline_ref, 'checkpointing', Path(directory))
        original_identity = checkpointing.mesh_portable_identity
        @contextmanager
        def old_identity():
            checkpointing.mesh_portable_identity = old_checkpointing.mesh_portable_identity
            try:
                yield
            finally:
                checkpointing.mesh_portable_identity = original_identity
        def old_snapshot(model, **kwargs):
            with old_identity():
                return old_settings.snapshot(model, **kwargs)
        for n in args.sizes:
            model,u = build(n)
            item = {'cells': n*n//2}
            for name, old, new in [('snapshot', old_snapshot, _model_settings.snapshot),
                                   ('review', old_review.review_model, _model_review.review_model)]:
                a,b = old(model,target=u), new(model,target=u)
                if name == 'review':
                    a.pop('resource_budget'); b.pop('resource_budget')  # live OS measurements
                if a != b:
                    raise AssertionError(f'{name} changed scientific records at grid size {n}')
                pairs = []
                for i in range(args.pairs):
                    row = {}
                    routes = [('baseline',old), ('candidate',new)]
                    if i % 2:
                        routes.reverse()
                    for label, fn in routes:
                        start = perf_counter(); fn(model,target=u)
                        row[label] = perf_counter()-start
                    pairs.append(row)
                peaks = {}
                for label,fn in [('baseline',old), ('candidate',new)]:
                    gc.collect()  # outside MPI operations, excluded from timing
                    tracemalloc.start(); fn(model,target=u)
                    peaks[label] = tracemalloc.get_traced_memory()[1]
                    tracemalloc.stop()
                medians = {label: statistics.median(p[label] for p in pairs) for label in ('baseline','candidate')}
                item[name] = {'pairs_seconds': pairs, 'median_seconds': medians,
                    'time_reduction_fraction': 1-medians['candidate']/medians['baseline'],
                    'python_peak_bytes': peaks, 'records_equal': True}
            report['sizes'].append(item)
            print(json.dumps(item), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
