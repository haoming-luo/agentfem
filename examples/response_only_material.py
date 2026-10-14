# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Small public example: a user elastic law, a moving tool and accepted export.

This example demonstrates the experimental response-only API, not an
industrial forming validation. All quantities use one consistent unit system.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from mpi4py import MPI
from dolfinx import mesh as dmesh
from agentfem import boundary_models, checkpointing, constitutive, elements, fields, mesh, models, results, studies
from agentfem.constitutive.user_material import MaterialStateSchema, MaterialStateVariable


def user_elastic_material():
    """Saint-Venant–Kirchhoff example; no fabricated algorithmic tangent."""
    schema = MaterialStateSchema('elastic-energy-v1', (MaterialStateVariable('energy'),))
    lam, mu = 4., 3.

    def response(request):
        f = request.deformation_gradient_new
        e = .5*(np.swapaxes(f,1,2)@f-np.eye(3))
        trace = np.trace(e,axis1=1,axis2=2)
        second_piola = lam*trace[:,None,None]*np.eye(3)+2*mu*e
        stress = (f@second_piola@np.swapaxes(f,1,2))/np.linalg.det(f)[:,None,None]
        energy = .5*lam*trace**2+mu*np.einsum('nij,nij->n',e,e)
        # This specific conservative elastic law has known reference-volume
        # energy. Do not reuse this equality for a dissipative user material.
        return constitutive.ExplicitMaterialBatchOutput(stress,energy[:,None],schema,
            energy-request.state_old[:,0],stored_energy_density=energy)

    return constitutive.explicit_user_material(state_schema=schema,update=response,
        identity={'example_law':'saint_venant_kirchhoff','lambda':lam,'mu':mu,'revision':1},density=1.)


def build(*, checkpoint=None, output=None, mass_scaling=None):
    domain = dmesh.create_unit_cube(MPI.COMM_SELF,2,2,2,cell_type=dmesh.CellType.hexahedron)
    model = models.create(study=studies.dynamic_solid(dimension=3),mesh=domain)
    u = model.field(fields.displacement(domain))
    model.material(user_elastic_material())
    for side in (0,1):
        boundary = mesh.boundary(domain,lambda x,side=side:np.isclose(x[0],side),name=f'end_{side}')
        motion = None if side==0 else boundary_models.prescribed_rigid_motion_schedule(
            boundary_models.prescribed_rigid_motion(translation=(-.001,0,0)),end_time=.01)
        tool = boundary_models.rigid_body(boundary_models.rigid_plane(
            point=(side,0,0),normal=(1 if side==0 else -1,0,0)),motion_schedule=motion)
        model.add_boundary_model(boundary_models.rigid_contact_pair(boundary,tool,
            penalty=100.,name=f'tool_{side}'))
    return model.step(target=u,element_policy=elements.uniform_strain_hex8(
        kinematics='finite_strain',hourglass_modulus=3.,hourglass_scale=.1),dt=.001,steps=10,
        time_increment_ceiling=.001,stress_free_initial_state=True,
        material_storage='owned_arrays',history_every=1,progress=False,
        checkpoint=checkpoint, output=output, mass_scaling=mass_scaling)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    step=build(checkpoint=checkpointing.every(5, directory=args.output/'checkpoints', keep_last=2),
               output=args.output/'fields.xdmf')
    journal=results.AcceptedHistoryJournal(args.output/'history.jsonl')
    step.solve_result(progress=journal.reporter(step, every=2))
    record=step.accepted_material_fields()
    np.savez(args.output/'material.npz',stress=record['cauchy_stress'],state=record['state'],
             cell_keys=record['cell_keys'])
    report={'completed_steps':step.completed_steps,'physical_time':record['time'],
            'independently_validated':False,'energy':step.history_records[-1],
            'capability':step.summary()['execution_scope'], 'journal':journal.summary()}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    main()
