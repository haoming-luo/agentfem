# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Live DG0 views of the declared uniform-gradient constitutive response."""

import numpy as np

from ..provenance import collective_call


class UniformHexCellFields:
    """Refresh constitutive cell fields, never re-evaluate full-gradient stress."""

    def __init__(self, operator, variables):
        from dolfinx import fem

        self.operator = operator
        domain = operator.displacement.function_space.mesh

        def selection():
            selected = (variables,) if isinstance(variables, str) else variables
            names = tuple(dict.fromkeys(str(name).upper() for name in selected))
            unknown = set(names) - {"U", "S", "E", "MISES", "SENER"}
            if unknown:
                raise ValueError(
                    f"Uniform Hex8 fields support U, S, E, MISES, SENER; got {sorted(unknown)}."
                )
            return names

        names = collective_call(
            selection, comm=domain.comm, label="Hex8 derived fields"
        )
        selections = domain.comm.allgather(names)
        if any(item != selections[0] for item in selections):
            raise ValueError("Uniform Hex8 field selection differs across MPI ranks.")
        self.fields = tuple(
            fem.Function(
                fem.functionspace(
                    domain, ("DG", 0, (3, 3)) if name in {"S", "E"} else ("DG", 0)
                ),
                name=name,
            )
            for name in names
            if name != "U"
        )
        count = len(operator.cell_nodes)
        self._cell_dofs = tuple(
            np.asarray(
                [field.function_space.dofmap.cell_dofs(k)[0] for k in range(count)],
                dtype=np.int32,
            )
            for field in self.fields
        )
        for field in self.fields:
            field._agentfem_processing = {
                "method": "direct_uniform_gradient_constitutive_response",
                "formulation": "uniform_strain_hex8",
                "kinematics": "small_strain",
                "source_position": "one_volume_average_strain_per_cell",
                "representation": "cellwise_constant",
                "space_family": "DG",
                "space_degree": 0,
                "interelement_smoothing": False,
                "material_boundary_averaging": False,
                "artificial_hourglass_stress_included": False,
                "unit_system": "model_consistent_not_inferred",
            }
            if field.name in {"S", "E"}:
                field._agentfem_processing["storage"] = "tensor_3x3"
                field._agentfem_processing["shear_convention"] = "tensor"

    def update(self):
        operator = self.operator
        operator.displacement.x.scatter_forward()
        collective_call(
            self._update_local, comm=operator.comm, label="Hex8 cell field response"
        )
        for field in self.fields:
            field.x.scatter_forward()
        return self.fields

    def _update_local(self):
        operator = self.operator
        if operator.cells is not None and self.fields:
            nodal = operator.displacement.x.array.reshape(-1, 3)
            for region, response in operator.cells.iter_responses(
                nodal, node_map=operator.cell_nodes
            ):
                stress = response.stress[:, [[0, 5, 4], [5, 1, 3], [4, 3, 2]]]
                for field, cell_dofs in zip(self.fields, self._cell_dofs):
                    if field.name == "S":
                        values = stress.reshape(-1, 9)
                    elif field.name == "E":
                        strain = response.strain.copy()
                        strain[:, 3:] *= 0.5
                        values = strain[:, [[0, 5, 4], [5, 1, 3], [4, 3, 2]]].reshape(
                            -1, 9
                        )
                    elif field.name == "MISES":
                        deviator = (
                            stress
                            - np.trace(stress, axis1=1, axis2=2)[:, None, None]
                            * np.eye(3)
                            / 3
                        )
                        values = np.sqrt(
                            1.5 * np.einsum("cij,cij->c", deviator, deviator)
                        )[:, None]
                    else:
                        values = (
                            response.physical_energy / operator.cells.volume[region]
                        )[:, None]
                    width = values.shape[1]
                    field.x.array.reshape(-1, width)[cell_dofs[region]] = values
