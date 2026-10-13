# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Classic free-bar longitudinal wave: ordinary Model, user law, core kernels.

No confidential mesh or extension is needed. A small-amplitude finite-strain
elastic calculation is compared with the linear analytical standing wave.
That comparison is a bounded linear-limit check, not a large-strain oracle.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from mpi4py import MPI
from dolfinx import mesh as dmesh
from agentfem.io import ResultWriter
from agentfem import constitutive, elements, fields, models, studies, checkpointing
from agentfem.constitutive.user_material import MaterialStateSchema, MaterialStateVariable


def elastic_material():
    # E=rho=1, nu=0: longitudinal wave speed c=1 with free lateral surfaces.
    schema = MaterialStateSchema('elastic-energy', (MaterialStateVariable('energy'),))
    def update(request):
        f = request.deformation_gradient_new
        strain = .5*(f.swapaxes(1,2)@f-np.eye(3))
        stress = (f@strain@f.swapaxes(1,2))/np.linalg.det(f)[:,None,None]
        energy = .5*np.sum(strain*strain,axis=(1,2))
        return constitutive.ExplicitMaterialBatchOutput(stress,energy[:,None],schema,
            energy-request.state_old[:,0],stored_energy_density=energy)
    return constitutive.explicit_user_material(state_schema=schema,update=update,
        identity={'law':'Saint-Venant-Kirchhoff','lambda':0.,'mu':.5,'revision':1},density=1.)


def run(output, *, cells=40, backend='auto'):
    if isinstance(cells, bool) or not isinstance(cells,int) or cells < 8:
        raise ValueError('At least eight axial cells are required')
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    started = perf_counter()
    mesh = dmesh.create_box(MPI.COMM_SELF,[[0.,0.,0.],[1.,.1,.1]],
                           [cells,1,1],cell_type=dmesh.CellType.hexahedron)
    model = models.create(study=studies.dynamic_solid(dimension=3),mesh=mesh)
    displacement = model.field(fields.displacement(mesh))
    model.material(elastic_material())
    duration, amplitude = .7, 1e-5
    steps = 16*cells
    step = model.step(target=displacement,
        element_policy=elements.uniform_strain_hex8(kinematics='finite_strain',
            hourglass_modulus=.5,hourglass_scale=.1),
        execution_kernel=elements.explicit_hex_kernel(backend=backend),
        material_storage='owned_arrays',dt=duration/steps,steps=steps,
        time_increment_ceiling=duration/steps,stress_free_initial_state=True,
        history_every=16,progress=False,
        checkpoint=checkpointing.every(steps,directory=output/'checkpoints'))
    x = displacement.value.function_space.tabulate_dof_coordinates()[:,0]
    step.state.v.value.x.array.reshape(-1,3)[:,0] = amplitude*np.pi*np.cos(np.pi*x)
    result = step.solve_result(metadata={'benchmark':'free-bar-standing-wave',
        'reference_scope':'small-amplitude linear limit', 'amplitude':amplitude})
    actual = step.state.u.value.x.array.reshape(-1,3)
    expected = amplitude*np.cos(np.pi*x)*np.sin(np.pi*duration)
    relative_error = float(np.linalg.norm(actual[:,0]-expected)/np.linalg.norm(expected))
    with ResultWriter(output/'fields.xdmf', mesh, (displacement,)) as archive:
        archive.write(duration)
    np.savez(output/'comparison.npz',x=x,displacement=actual,analytical=expected)
    result.add_artifact('fields',output/'fields.xdmf')
    result.write_manifest(output/'simulation-result.json')
    report = {'completed':step.completed_steps==steps,'cells':cells,'steps':steps,
        'physical_time':duration,'relative_displacement_error':relative_error,
        'linear_limit_check_passed':relative_error < .01,
        'end_to_end_seconds':perf_counter()-started,'energy':step.history_records[-1],
        'backend':backend, 'independent_large_strain_validation':False}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    if not report['linear_limit_check_passed']:
        raise RuntimeError('Standing-wave reference check failed; inspect report.json')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cells',type=int,default=40)
    parser.add_argument('--backend',choices=('auto','native','numpy'),default='auto')
    args = parser.parse_args()
    print(json.dumps(run(args.output,cells=args.cells,backend=args.backend)))
