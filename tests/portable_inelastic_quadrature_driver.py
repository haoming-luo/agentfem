"""Write inelastic state with two ranks and restore it with one rank."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from dolfinx import mesh as dolfinx_mesh
from mpi4py import MPI

from agentfem import constitutive, mesh


def _domain_and_maps():
    domain = dolfinx_mesh.create_unit_cube(MPI.COMM_WORLD, 2, 1, 1)
    regions = mesh.partition_cells(
        domain,
        left=lambda x: x[0] <= 0.5,
        right=lambda x: x[0] > 0.5,
    )
    j2_records = (
        SimpleNamespace(
            item=constitutive.J2LinearIsotropicHardening(
                young=1000.0,
                poisson=0.3,
                yield_stress=2.0,
                hardening_modulus=20.0,
            ),
            region=regions.left,
        ),
        SimpleNamespace(
            item=constitutive.J2LinearIsotropicHardening(
                young=1500.0,
                poisson=0.3,
                yield_stress=3.0,
                hardening_modulus=30.0,
            ),
            region=regions.right,
        ),
    )
    creep_records = (
        SimpleNamespace(
            item=constitutive.isotropic_power_law(
                young=1000.0,
                poisson=0.3,
                density=1.0,
                coefficient=1.0e-5,
                stress_exponent=2.0,
                reference_stress=1.0,
            ),
            region=regions.left,
        ),
        SimpleNamespace(
            item=constitutive.isotropic_power_law(
                young=1500.0,
                poisson=0.3,
                density=1.0,
                coefficient=2.0e-5,
                stress_exponent=2.0,
                reference_stress=1.0,
            ),
            region=regions.right,
        ),
    )
    return (
        domain,
        constitutive.QuadratureMaterialMap.from_assignments(
            domain,
            j2_records,
            material_type=constitutive.J2LinearIsotropicHardening,
        ),
        constitutive.QuadratureMaterialMap.from_assignments(
            domain,
            creep_records,
            material_type=constitutive.IsotropicPowerLawCreepMaterial,
        ),
    )


def _expected(state, *, scale: float):
    keys = np.asarray(state.domain.topology.original_cell_index, dtype=float)
    points = len(state.stress.points)
    base = np.repeat(keys, points) * scale
    base += np.tile(np.arange(points, dtype=float), len(keys)) * scale / 10.0
    tensor = np.zeros((len(base), 3, 3))
    tensor[:, 0, 0] = base
    tensor[:, 1, 1] = -0.5 * base
    tensor[:, 2, 2] = -0.5 * base
    return tensor, np.abs(base)


def _generic_state(domain):
    schema = constitutive.MaterialStateSchema(
        "portable_finite_strain_test",
        (
            constitutive.MaterialStateVariable(
                "damage",
                output_name="SDV_DAMAGE",
                unit="1",
            ),
            constitutive.MaterialStateVariable(
                "plastic_deformation_gradient",
                shape=(3, 3),
                initial_value=np.eye(3),
                output_name="SDV_FP",
                unit="1",
            ),
        ),
        version="1.0.0",
    )
    return constitutive.MaterialQuadratureState.create(domain, schema, degree=2)


def _expected_generic(state):
    keys = np.asarray(state.domain.topology.original_cell_index, dtype=float)
    points = len(state.reference_field.points)
    damage = np.repeat(keys, points) * 0.01
    damage += np.tile(np.arange(points, dtype=float), len(keys)) * 0.001
    plastic_gradient = np.broadcast_to(
        np.eye(3),
        (len(damage), 3, 3),
    ).copy()
    plastic_gradient[:, 0, 0] += damage
    plastic_gradient[:, 1, 1] -= 0.5 * damage
    return damage, plastic_gradient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("write", "read"))
    parser.add_argument("root", type=Path)
    arguments = parser.parse_args()
    domain, j2_materials, creep_materials = _domain_and_maps()

    j2 = constitutive.J2QuadratureState.create(domain, degree=2)
    creep = constitutive.CreepQuadratureState.create(domain, degree=2)
    generic = _generic_state(domain)
    j2_tensor, j2_scalar = _expected(j2, scale=0.01)
    creep_tensor, creep_scalar = _expected(creep, scale=0.001)
    generic_damage, generic_gradient = _expected_generic(generic)

    if arguments.action == "write":
        if MPI.COMM_WORLD.size != 2:
            raise RuntimeError("Portable write acceptance requires two ranks.")
        j2.plastic_strain.assign(j2_tensor)
        j2.equivalent_plastic_strain.assign(j2_scalar)
        creep.creep_strain.assign(creep_tensor)
        creep.equivalent_creep_strain.assign(creep_scalar)
        generic.committed["damage"].assign(generic_damage)
        generic.committed["plastic_deformation_gradient"].assign(
            generic_gradient
        )
        generic.rollback()
        j2.save(arguments.root.with_name(arguments.root.name + "-j2"), material=j2_materials)
        creep.save(
            arguments.root.with_name(arguments.root.name + "-creep"),
            material=creep_materials,
        )
        generic.save(arguments.root.with_name(arguments.root.name + "-generic"))
        return

    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("Portable read acceptance requires one rank.")
    j2.load(arguments.root.with_name(arguments.root.name + "-j2.npz"), material=j2_materials)
    creep.load(
        arguments.root.with_name(arguments.root.name + "-creep.npz"),
        material=creep_materials,
    )
    generic.load(arguments.root.with_name(arguments.root.name + "-generic.npz"))
    np.testing.assert_allclose(j2.plastic_strain.values, j2_tensor)
    np.testing.assert_allclose(j2.equivalent_plastic_strain.values, j2_scalar)
    np.testing.assert_allclose(creep.creep_strain.values, creep_tensor)
    np.testing.assert_allclose(creep.equivalent_creep_strain.values, creep_scalar)
    np.testing.assert_allclose(
        generic.committed["damage"].values,
        generic_damage,
    )
    np.testing.assert_allclose(
        generic.committed["plastic_deformation_gradient"].values,
        generic_gradient,
    )


if __name__ == "__main__":
    main()
