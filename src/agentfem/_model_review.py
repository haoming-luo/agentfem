# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Collective, non-solving review of the actual registered engineering model."""
import numpy as np
import ufl
from mpi4py import MPI
from ._model_support import domain, describe


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
        constraints.append(entry)
    loads = []
    for item in model.loads:
        entry = {'definition': describe(item), 'status': 'not_checked'}
        region = getattr(item, 'location', None)
        if region is not None and getattr(region, 'domain', None) is mesh:
            area = results.region_measure(on=region, study=model.study)
            entry.update(status='checked', reference_measure=float(area),
                         evaluation='current input snapshot; not an integrated load history')
            if area <= 0:
                findings.append({'code': 'AFM-REVIEW-EMPTY-LOAD', 'name': item.name})
            value = getattr(item, 'value', None)
            if getattr(value, 'ufl_shape', None) == (mesh.geometry.dim,):
                entry['resultant'] = np.asarray(results.boundary_resultant(value, on=region, study=model.study)).tolist()
                # Planar/3D Cartesian moment only. Axisymmetric resultants need
                # their own revolution semantics, not a planar cross product.
                if getattr(model.study, 'assumption', '') != 'axisymmetric':
                    x = ufl.SpatialCoordinate(mesh)
                    moment = ufl.cross(x, value) if mesh.geometry.dim == 3 else x[0]*value[1]-x[1]*value[0]
                    entry['moment_about_origin'] = np.asarray(results.integral(moment, measure=region.measure, comm=comm)).tolist()
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
        'model': model.summary(), 'physics': model_physics(model, options=step_options),
        'capability': capability, 'validation': validation,
        'constraints': constraints, 'loads': loads, 'rigid_modes': rigid,
        'findings': findings, 'solver_executed': False,
        'checks': {'boundary_quadrature': 'checked', 'tangent_rank': 'not_checked',
                   'physical_model_validity': 'not_checked'},
        'step_options': {name: describe(value) for name, value in dict(step_options or {}).items()},
    }
