# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Frozen views of registered settings; no Python parsing or model replay."""
import json
from pathlib import Path

import numpy as np

from ._mpi_collectives import _defer_automatic_gc, object_allgather


def snapshot(model, *, target=None, step_options=None, source_files=()):
    from . import __version__, checkpointing, provenance
    from ._model_support import domain
    from ._model_review import _bounds
    from .ir.export import describe
    from .step_providers import step_capability

    mesh = domain(model.mesh)
    if mesh is None:
        raise ValueError('model_settings requires a registered mesh.')
    # This existing identity routine performs object collectives. Keep its
    # communicator-finalization callbacks out of the same mpi4py lock path.
    with _defer_automatic_gc():
        identity = checkpointing.mesh_portable_identity(mesh)
    with _defer_automatic_gc():
        root = provenance.collective_call(lambda: model.to_ir(include_validation=False)['root'],
            comm=mesh.comm, label='model settings declarations')
    root.pop('name', None)
    root.pop('execution_backend', None)
    registered_steps = root.pop('steps', ())
    root['mesh'] = {'identity': identity,
        'bounds': _bounds(mesh, mesh.geometry.x[:, :mesh.geometry.dim])}
    root['target'] = None if target is None else getattr(target, 'name', None)
    options = dict(step_options or {})
    operational = {k: options.pop(k) for k in ('output', 'name', 'progress', 'print_every') if k in options}
    root['step_options'] = describe(options)
    with _defer_automatic_gc():
        provenance.collective_canonical_record(root, comm=mesh.comm, label='model settings declarations')
    capability = step_capability(model, target=target, options=step_options)
    root['provider'] = capability.get('provider')
    root['procedure'] = capability.get('procedure')
    gaps = []
    if registered_steps:
        gaps.append({'path': 'steps', 'reason': 'use explicit pre-build step_options; built Step state is not replayed'})
    # Preserve actual tag membership, not just a region's display name.
    regions = list(model.regions)
    from .constraints import dirichlet_constraints
    members = dirichlet_constraints(model.constraints)
    for item in (*model.loads, *members, *model.materials):
        region = getattr(item, 'location', None)
        if region is None:
            region = getattr(item, 'region', None)
        if region is not None and all(region is not r for r in regions):
            regions.append(region)
    root['region_membership'] = []
    for region in regions:
        tags = getattr(region, 'facet_tags', None)
        if tags is None:
            tags = getattr(region, 'cell_tags', None)
        if tags is None or getattr(region, 'domain', None) is not mesh:
            gaps.append({'path': 'regions', 'reason': 'unidentified or non-tagged region'})
            continue
        from dolfinx.mesh import entities_to_geometry
        mesh.topology.create_entities(tags.dim)
        indices = np.asarray(tags.find(region.tag), dtype=np.int32)
        indices = indices[indices < mesh.topology.index_map(tags.dim).size_local]
        local = []
        for nodes in entities_to_geometry(mesh, tags.dim, indices):
            points = mesh.geometry.x[nodes, :mesh.geometry.dim]
            local.append(provenance.content_fingerprint(sorted(points.tolist())))
        entries = sorted(v for chunk in object_allgather(mesh.comm, local) for v in chunk)
        root['region_membership'].append({'name': region.name, 'dimension': tags.dim,
            'tag': region.tag, 'entities': len(entries), 'fingerprint': provenance.content_fingerprint(entries)})
    root['constraint_components'] = []
    for item in members:
        bc = getattr(item, 'bc', None)
        if bc is not None:
            root['constraint_components'].append(list(bc.function_space.component()))
        if getattr(item, 'location', None) is None:
            gaps.append({'path': 'constraints', 'reason': 'direct constraint lacks a registered physical region'})
    _gaps(root, 'model', gaps)
    sources = provenance.scientific_input_manifest(
        {str(Path(p)): Path(p) for p in source_files}, label='source_files')
    gaps.extend(sources['missing'])
    record = provenance.scientific_input_manifest(root, label='model_settings')
    gaps.extend(record['missing'])
    return {'schema': 'agentfem.model-settings.v1', 'agentfem_version': __version__,
        'scientific_inputs': record, 'operational': describe(operational),
        'source_files': sources, 'coverage': {'complete': not gaps, 'gaps': gaps,
            'scope': 'registered declarations and explicit Step options; no arbitrary Python or physical-equivalence proof; mesh identity uses checkpoint coordinate policy'}}


def _gaps(value, path, output):
    if isinstance(value, dict):
        if value.get('serializable') is False or str(value.get('kind', '')).startswith('opaque'):
            output.append({'path': path, 'reason': 'opaque declaration'})
        for key, item in value.items():
            _gaps(item, f'{path}.{key}', output)
    elif isinstance(value, (tuple, list)):
        for i, item in enumerate(value):
            _gaps(item, f'{path}[{i}]', output)


def compare(reference, candidate):
    from . import provenance
    def load(value):
        record = json.loads(Path(value).read_text()) if isinstance(value, (str, Path)) else value
        if not isinstance(record, dict) or record.get('schema') != 'agentfem.model-settings.v1':
            raise ValueError('Expected a model_settings snapshot, not a result or arbitrary model summary.')
        if not isinstance(record.get('coverage'), dict) or not isinstance(record['coverage'].get('complete'), bool):
            raise ValueError('Model settings require explicit coverage metadata.')
        for name in ('scientific_inputs', 'source_files'):
            manifest = record.get(name)
            if not isinstance(manifest, dict) or not all(k in manifest for k in ('label', 'record', 'fingerprint')):
                raise ValueError(f'Model settings require a complete {name} manifest.')
            if not isinstance(manifest['record'], dict):
                raise ValueError(f'{name} record must be a mapping.')
            expected = provenance.content_fingerprint({'label': manifest['label'], 'record': manifest['record']})
            if manifest['fingerprint'] != expected:
                raise ValueError(f'{name} fingerprint does not match its record.')
        return record
    a, b = load(reference), load(candidate)
    differences = []
    _differences(a['scientific_inputs']['record'], b['scientific_inputs']['record'], 'model', differences)
    operations, sources = [], []
    _differences(a['operational'], b['operational'], 'operational', operations)
    _differences(a['source_files']['record'], b['source_files']['record'], 'source_files', sources)
    complete = a['coverage']['complete'] and b['coverage']['complete']
    return {'schema': 'agentfem.model-settings-comparison.v1',
        'recorded_settings_equal': not differences, 'coverage_complete': complete,
        'status': 'changed' if differences else ('unchanged_within_recorded_scope' if complete else 'inconclusive'),
        'scientific_changes': differences, 'operational_changes': operations, 'source_changes': sources,
        'reference_coverage': a['coverage'], 'candidate_coverage': b['coverage'],
        'solver_executed': False}


def _differences(a, b, path, output):
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(a.keys() | b.keys()):
            if key not in a or key not in b:
                output.append({'path': f'{path}.{key}', 'kind': 'added' if key in b else 'removed',
                    'expected': a.get(key), 'actual': b.get(key)})
            else:
                _differences(a[key], b[key], f'{path}.{key}', output)
    elif isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)) and len(a) == len(b):
        for i, (left, right) in enumerate(zip(a, b)):
            _differences(left, right, f'{path}[{i}]', output)
    elif type(a) is not type(b) or a != b:
        output.append({'path': path, 'kind': 'changed', 'expected': a, 'actual': b})
