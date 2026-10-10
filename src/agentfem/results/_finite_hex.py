# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Unsmoothed views of accepted finite-Hex material evidence, without reintegration."""

import numpy as np

from ..provenance import collective_call


class FiniteHexCellFields:
    def __init__(self, residual, variables):
        from dolfinx import fem

        self.residual = residual
        internal = residual.internal
        response = internal.response
        domain = internal.displacement.function_space.mesh
        def source_contract():
            sources = {
                "S": response.cauchy_stress,
                "P": response.first_piola_stress,
                "SENER": response.strain_energy_density,
            }
            reserved = {*sources, "U", "F", "MISES"}
            state_descriptions = {}

            def add(name, source):
                # A component called S must never replace physical stress;
                # case aliases must not make a requested field ambiguous.
                if name.upper() in reserved:
                    raise ValueError(
                        "Material output name collides with a finite Hex8 field."
                    )
                reserved.add(name.upper())
                sources[name] = source

            for name, source in response.stored_energy_density_components.items():
                add(name, source)
            for variable in residual.material.state_schema.variables:
                if variable.output_name:
                    add(variable.output_name, response.state.committed[variable.name])
                    state_descriptions[variable.output_name] = variable.summary()
            return sources, state_descriptions

        self.sources, state_descriptions = collective_call(
            source_contract, comm=domain.comm, label="Finite Hex8 output source contract"
        )

        def selection():
            selected = (variables,) if isinstance(variables, str) else variables
            # Built-in abbreviations accept lowercase input. External state
            # fields retain their declared spelling instead of guessing it.
            names = tuple(
                dict.fromkeys(
                    str(name) if str(name) in self.sources else str(name).upper()
                    for name in selected
                )
            )
            unknown = set(names) - self.sources.keys() - {"U", "F", "MISES"}
            if unknown:
                raise ValueError(f"Unsupported finite Hex8 fields: {sorted(unknown)}.")
            return names

        names = collective_call(
            selection, comm=domain.comm, label="Finite Hex8 output selection"
        )
        if any(value != names for value in domain.comm.allgather(names)):
            raise ValueError("Finite Hex8 field selection differs across ranks.")
        shapes = {name: source.value_shape for name, source in self.sources.items()}
        shapes.update(F=(3, 3), MISES=())
        self.fields = tuple(
            fem.Function(
                fem.functionspace(
                    domain, ("DG", 0, shapes[name]) if shapes[name] else ("DG", 0)
                ),
                name=name,
            )
            for name in names
            if name != "U"
        )
        self.cell_dofs = tuple(
            np.asarray(
                [
                    value.function_space.dofmap.cell_dofs(k)[0]
                    for k in range(len(internal.cell_nodes))
                ],
                dtype=np.int32,
            )
            for value in self.fields
        )
        for value in self.fields:
            value._agentfem_processing = {
                "method": "direct_accepted_uniform_gradient_material_response",
                "formulation": "uniform_strain_hex8",
                "kinematics": "finite_strain",
                "source_position": "one_material_point_per_reference_cell",
                "representation": "cellwise_constant",
                "space_family": "DG",
                "space_degree": 0,
                "interelement_smoothing": False,
                "material_boundary_averaging": False,
                "artificial_hourglass_stress_included": False,
                "unit_system": "model_consistent_not_inferred",
            }
            if value.name in {"S", "MISES"}:
                value._agentfem_processing["stress_measure"] = "cauchy"
            if value.name == "P":
                value._agentfem_processing["stress_measure"] = "first_piola"
            if value.name in {"SENER", *response.stored_energy_density_components}:
                value._agentfem_processing["volume_measure"] = "reference"
            if value.name in state_descriptions:
                value._agentfem_processing["state_variable"] = state_descriptions[
                    value.name
                ]
                value._agentfem_processing["state_schema_identity"] = (
                    residual.material.state_schema.identity
                )

    def update(self):
        collective_call(
            self._update_local,
            comm=self.residual.comm,
            label="Accepted finite Hex8 fields",
        )
        for value in self.fields:
            value.x.scatter_forward()
        return self.fields

    def _update_local(self):
        self.residual.require_accepted_configuration()
        count = len(self.residual.internal.cell_nodes)
        for value, cell_dofs in zip(self.fields, self.cell_dofs):
            if value.name == "F":
                source = self.residual.accepted_gradient
            elif value.name == "MISES":
                stress = self.sources["S"].owned_values
                deviator = (
                    stress
                    - np.trace(stress, axis1=1, axis2=2)[:, None, None] * np.eye(3) / 3
                )
                source = np.sqrt(1.5 * np.einsum("cij,cij->c", deviator, deviator))
            else:
                source = self.sources[value.name].owned_values
            width = int(np.prod(value.ufl_shape)) if value.ufl_shape else 1
            value.x.array.reshape(-1, width)[cell_dofs] = source.reshape(count, width)
