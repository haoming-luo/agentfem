# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Collective, non-solving review of the actual registered engineering model."""
import numpy as np
import ufl
from mpi4py import MPI
from ._model_support import domain, describe


def _bounds(mesh, points):
    lower = points.min(axis=0) if len(points) else np.full(mesh.geometry.dim, np.inf)
    upper = points.max(axis=0) if len(points) else np.full(mesh.geometry.dim, -np.inf)
    mesh.comm.Allreduce(MPI.IN_PLACE, lower, op=MPI.MIN)
    mesh.comm.Allreduce(MPI.IN_PLACE, upper, op=MPI.MAX)
    return [lower.tolist(), upper.tolist()] if np.isfinite(lower).all() else None


def discretization(model):
    """Count registered spaces, not a guessed assembled/reduced system size."""
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
            'shared_space': id(space) in seen})
        seen.add(id(space))
    x = np.asarray(mesh.geometry.x)[:, :mesh.geometry.dim]
    return {'cell_type': mesh.topology.cell_name(),
        'global_cells': int(mesh.topology.index_map(mesh.topology.dim).size_global),
        'bounds': _bounds(mesh, x),
        'fields': fields,
        'registered_scalar_dofs': sum(f['global_scalar_dofs'] for f in fields if not f['shared_space']),
        'dof_scope': 'registered distinct spaces before constraints; not reduced solver size'}


def region_evidence(region, *, study):
    """Owned entity counts and physical measure of a registered tagged region."""
    from . import results
    mesh = getattr(region, 'domain', None)
    tags = getattr(region, 'facet_tags', None)
    if tags is None:
        tags = getattr(region, 'cell_tags', None)
    if mesh is None or tags is None:
        return {'status': 'not_checked', 'reason': 'requires a tagged cell or boundary region'}
    selected = np.asarray(tags.find(region.tag), dtype=np.int32)
    mesh.topology.create_entities(tags.dim)
    selected = selected[selected < mesh.topology.index_map(tags.dim).size_local]
    count = int(mesh.comm.allreduce(len(selected), op=MPI.SUM))
    from dolfinx.mesh import entities_to_geometry
    indices = entities_to_geometry(mesh, tags.dim, selected).reshape(-1)
    points = mesh.geometry.x[indices, :mesh.geometry.dim]
    return {'status': 'checked', 'name': region.name, 'tag': int(region.tag),
        'entity_dimension': int(tags.dim), 'global_owned_entities': count,
        'bounds': _bounds(mesh, points),
        'physical_measure': float(results.region_measure(on=region, study=study)),
        'measure_convention': 'study-weighted reference measure; axisymmetry includes full revolution'}


def review_model(model, *, target=None, step_options=None):
    from . import __version__, results
    from . import constraints as constraint_api
    from .step_providers import step_capability
    from ._physical_semantics import model_physics

    mesh = domain(model.mesh)
    if mesh is None:
        raise ValueError('Model review requires a mesh.')
    comm = mesh.comm
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
        entry['regions'] = [region_evidence(member.location, study=model.study)
                            for member in members if getattr(member, 'location', None) is not None]
        constraints.append(entry)
    loads = []
    for item in model.loads:
        entry = {'definition': describe(item), 'status': 'not_checked'}
        region = getattr(item, 'location', None)
        if region is not None and getattr(region, 'domain', None) is mesh:
            area = results.region_measure(on=region, study=model.study)
            entry.update(status='checked', reference_measure=float(area),
                         evaluation='current input snapshot; not an integrated load history')
            entry['region'] = region_evidence(region, study=model.study)
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
    return {
        'schema': 'agentfem.model-review.v1', 'agentfem_version': __version__,
        'model': model.summary(), 'physics': physics,
        'discretization': discretization(model),
        'material_regions': [{'definition': describe(item),
            'region': ({'status': 'whole_domain'} if item.region is None else
                       region_evidence(item.region, study=model.study))} for item in model.materials],
        'capability': capability, 'validation': validation,
        'constraints': constraints, 'loads': loads, 'rigid_modes': rigid,
        'findings': findings, 'solver_executed': False,
        'checks': {'boundary_quadrature': 'checked', 'tangent_rank': 'not_checked',
                   'physical_model_validity': 'not_checked'},
        'step_options': {name: describe(value) for name, value in dict(step_options or {}).items()},
    }
