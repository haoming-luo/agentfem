# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Collective, non-solving review of the actual registered engineering model."""
from copy import deepcopy
import numpy as np
import ufl
from mpi4py import MPI
from ._model_support import domain, describe
from ._mpi_collectives import _defer_automatic_gc


def _review_communication_plan(model, mesh, target, step_options):
    """Agree on collective branches before validation or boundary assembly.

    Local entity counts and field coefficients intentionally stay out of the
    plan: they normally differ across partitions. Python aliases are likewise
    irrelevant. Region ownership and call order must agree, however.
    """
    from .constraints import dirichlet_constraints
    from .provenance import collective_call, collective_canonical_record

    def region_plan(region):
        if region is None:
            return None
        if getattr(region, 'domain', None) is not None and region.domain is not mesh:
            raise ValueError('Model review region must belong to the model mesh.')
        tags = getattr(region, 'facet_tags', None)
        if tags is None:
            tags = getattr(region, 'cell_tags', None)
        if tags is None:
            return {'status': 'not_tagged'}
        dim = int(tags.dim)
        if not 0 <= dim <= mesh.topology.dim:
            raise ValueError('Review region entity dimension exceeds the model mesh dimension.')
        return {'name': region.name, 'dimension': dim, 'tag': int(region.tag)}

    def field_plan(item):
        space = getattr(item, 'function_space', None)
        if space is not None and space.mesh is not mesh:
            raise ValueError('Model review field must belong to the model mesh.')
        return {'name': getattr(item, 'name', None), 'has_space': space is not None}

    def prepare():
        return {
            'study': describe(model.study),
            'fields': [field_plan(item) for item in model.fields],
            'target': None if target is None else field_plan(target),
            'constraints': [[{'has_bc': getattr(member, 'bc', None) is not None,
                'region': region_plan(getattr(member, 'location', None))}
                for member in dirichlet_constraints((item,))] for item in model.constraints],
            'loads': [{'region': region_plan(getattr(item, 'location', None)),
                'shape': getattr(getattr(item, 'value', None), 'ufl_shape', None)} for item in model.loads],
            'materials': [region_plan(item.region) for item in model.materials],
            'regions': [region_plan(item) for item in model.regions],
            'reference_thickness': (step_options or {}).get('reference_thickness'),
        }

    with _defer_automatic_gc():
        plan = collective_call(prepare, comm=mesh.comm, label='model review communication plan')
        collective_canonical_record(plan, comm=mesh.comm, label='model review communication plan')


def _bounds(mesh, points):
    lower = points.min(axis=0) if len(points) else np.full(mesh.geometry.dim, np.inf)
    upper = points.max(axis=0) if len(points) else np.full(mesh.geometry.dim, -np.inf)
    mesh.comm.Allreduce(MPI.IN_PLACE, lower, op=MPI.MIN)
    mesh.comm.Allreduce(MPI.IN_PLACE, upper, op=MPI.MAX)
    return [lower.tolist(), upper.tolist()] if np.isfinite(lower).all() else None


def discretization(model):
    """Count independent registered unknowns, not a reduced system size."""
    mesh = domain(model.mesh)
    fields = []
    seen = set()
    for item in model.fields:
        space = getattr(item, 'function_space', None)
        if space is None:
            continue
        element = space.ufl_element()
        index = space.dofmap.index_map
        fields.append({'name': getattr(item, 'name', None), 'element': str(element),
            'degree': element.degree, 'value_shape': list(element.reference_value_shape),
            'global_scalar_dofs': int(index.size_global * space.dofmap.index_map_bs),
            'shared_unknown': id(getattr(item, 'value', item)) in seen})
        seen.add(id(getattr(item, 'value', item)))
    x = np.asarray(mesh.geometry.x)[:, :mesh.geometry.dim]
    return {'cell_type': mesh.topology.cell_name(),
        'global_cells': int(mesh.topology.index_map(mesh.topology.dim).size_global),
        'bounds': _bounds(mesh, x),
        'fields': fields,
        'registered_scalar_dofs': sum(f['global_scalar_dofs'] for f in fields if not f['shared_unknown']),
        'dof_scope': 'registered independent unknowns before constraints; aliases counted once; not reduced solver size'}


def region_evidence(region, *, study):
    """Owned entity counts and physical measure of a registered tagged region."""
    from . import results
    mesh = getattr(region, 'domain', None)
    tags = getattr(region, 'facet_tags', None)
    if tags is None:
        tags = getattr(region, 'cell_tags', None)
    if mesh is None or tags is None:
        return {'status': 'not_checked', 'reason': 'requires a tagged cell or boundary region'}
    mesh.topology.create_entities(tags.dim)
    from dolfinx.mesh import entities_to_geometry
    from .provenance import collective_call
    def sample_local():
        selected = np.asarray(tags.find(region.tag), dtype=np.int32)
        index_map = mesh.topology.index_map(tags.dim)
        if np.any(selected < 0) or np.any(selected >= index_map.size_local + index_map.num_ghosts):
            raise ValueError('Review region tags contain out-of-range entity indices.')
        selected = selected[selected < index_map.size_local]
        indices = entities_to_geometry(mesh, tags.dim, selected).reshape(-1)
        return len(selected), mesh.geometry.x[indices, :mesh.geometry.dim]
    with _defer_automatic_gc():
        local_count, points = collective_call(sample_local, comm=mesh.comm, label='model review region sampling')
    count = int(mesh.comm.allreduce(local_count, op=MPI.SUM))
    return {'status': 'checked', 'name': region.name, 'tag': int(region.tag),
        'entity_dimension': int(tags.dim), 'global_owned_entities': count,
        'bounds': _bounds(mesh, points),
        'physical_measure': float(results.region_measure(on=region, study=study)),
        'measure_convention': 'study-weighted reference measure; axisymmetry includes full revolution'}


class _ReviewRegions:
    """Reuse integrations within one review, never across geometry updates."""

    def __init__(self, mesh, study):
        self.mesh, self.study, self.cache = mesh, study, {}

    def __call__(self, region):
        key = id(region)
        ready = np.array(int(key in self.cache), dtype=np.int32)
        # Rank-local Python aliasing need not match across ranks. Recompute
        # everywhere if any rank misses; never skip a collective on one rank.
        self.mesh.comm.Allreduce(MPI.IN_PLACE, ready, op=MPI.MIN)
        if not ready.item():
            self.cache[key] = region_evidence(region, study=self.study)
        # Report entries remain detached even when they refer to one region.
        return deepcopy(self.cache[key])


def review_model(model, *, target=None, step_options=None, output_fields=(), output_frames=None, output_directory=None):
    from . import __version__, results
    from . import constraints as constraint_api
    from .step_providers import step_capability
    from ._physical_semantics import model_physics

    mesh = domain(model.mesh)
    if mesh is None:
        raise ValueError('Model review requires a mesh.')
    _review_communication_plan(model, mesh, target, step_options)
    comm = mesh.comm
    evidence = _ReviewRegions(mesh, model.study)
    validation = model.validate(target=target, step_options=step_options).as_dict()
    capability = step_capability(model, target=target, options=step_options)
    physics = model_physics(model, options=step_options)
    thickness = 1.
    if ((capability.get('provider') or {}).get('name') == 'finite_strain_user_material_static'
        and capability.get('ready') and mesh.geometry.dim == 2):
        thickness = float((step_options or {}).get('reference_thickness') or 1.)
        physics['section'] = {'status': 'declared', 'reference_thickness': thickness,
            'configuration': 'reference', 'distribution': 'uniform',
            'virtual_work_multiplier': thickness,
            'force_convention': 'integrated through reference thickness; no second scaling'}
        if (step_options or {}).get('reference_thickness') is None:
            physics['section'].update(status='unit_reference_thickness', reference_thickness=None,
                force_convention='per unit reference thickness')
    constraints = []
    findings = []
    for item in model.constraints:
        members = constraint_api.dirichlet_constraints((item,))
        entry = {'definition': describe(item), 'status': 'not_checked'}
        counts = []
        for member in members:
            bc = getattr(member, 'bc', None)
            if bc is None:
                continue
            _, owned = bc.dof_indices()
            count = comm.allreduce(int(owned) * bc.function_space.dofmap.bs, op=MPI.SUM)
            counts.append(int(count))
            if count == 0:
                findings.append({'code': 'AFM-REVIEW-EMPTY-CONSTRAINT',
                                 'name': getattr(member, 'name', getattr(item, 'name', 'constraint'))})
        if counts:
            entry.update(status='checked', owned_scalar_dof_count=sum(counts),
                         component_constraint_count=len(counts))
        entry['regions'] = [evidence(member.location)
                            for member in members if getattr(member, 'location', None) is not None]
        constraints.append(entry)
    loads = []
    for item in model.loads:
        entry = {'definition': describe(item), 'status': 'not_checked'}
        region = getattr(item, 'location', None)
        if region is not None and getattr(region, 'domain', None) is mesh:
            region_info = evidence(region)
            area = (region_info['physical_measure'] if region_info['status'] == 'checked'
                    else results.region_measure(on=region, study=model.study))
            entry.update(status='checked', reference_measure=float(area),
                         evaluation='current input snapshot; not an integrated load history')
            entry['region'] = region_info
            if area <= 0:
                findings.append({'code': 'AFM-REVIEW-EMPTY-LOAD', 'name': item.name})
            value = getattr(item, 'value', None)
            if getattr(value, 'ufl_shape', None) == (mesh.geometry.dim,):
                entry['resultant'] = np.asarray(results.boundary_resultant(value, on=region, study=model.study)).tolist()
                entry['reference_thickness_multiplier'] = thickness
                entry['resultant'] = (thickness*np.asarray(entry['resultant'])).tolist()
                # Planar/3D Cartesian moment only. Axisymmetric resultants need
                # their own revolution semantics, not a planar cross product.
                if getattr(model.study, 'assumption', '') != 'axisymmetric':
                    x = ufl.SpatialCoordinate(mesh)
                    moment = ufl.cross(x, value) if mesh.geometry.dim == 3 else x[0]*value[1]-x[1]*value[0]
                    entry['moment_about_origin'] = (thickness*np.asarray(results.integral(moment, measure=region.measure, comm=comm))).tolist()
        loads.append(entry)
    selected = target if target is not None else (model.fields[0] if model.fields else None)
    rigid = {'status': 'not_checked', 'reason': 'not a supported displacement field/constraint combination'}
    if selected is not None and getattr(model.study, 'is_solid_mechanics', False):
        try:
            audit = model.rigid_mode_audit(selected).summary()
            rigid = {'status': 'checked', **audit, 'scope': 'global analytical rigid modes; not a mechanism, contact or buckling stability proof'}
        except (ValueError, TypeError, NotImplementedError) as error:
            rigid['reason'] = str(error)
    grid = discretization(model)
    from ._resource_budget import budget
    resources = budget(model, target=target, step_options=step_options, fields=output_fields,
        frames=output_frames, output_directory=output_directory, grid=grid)
    return {
        'schema': 'agentfem.model-review.v1', 'agentfem_version': __version__,
        'model': model.summary(), 'physics': physics,
        'discretization': grid, 'resource_budget': resources,
        'material_regions': [{'definition': describe(item),
            'region': ({'status': 'whole_domain'} if item.region is None else
                       evidence(item.region))} for item in model.materials],
        'capability': capability, 'validation': validation,
        'constraints': constraints, 'loads': loads, 'rigid_modes': rigid,
        'findings': findings, 'solver_executed': False,
        'checks': {'boundary_quadrature': 'checked', 'tangent_rank': 'not_checked',
                   'physical_model_validity': 'not_checked'},
        'step_options': {name: describe(value) for name, value in dict(step_options or {}).items()},
    }
