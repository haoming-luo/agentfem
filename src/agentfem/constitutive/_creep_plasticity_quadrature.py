# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Joint creep/plastic state on the existing implicit creep lifecycle."""

from dataclasses import dataclass, field, fields

import numpy as np

from ._creep_plasticity import CreepPlasticityMaterial, CreepPlasticityState
from .creep import ImplicitCreepState
from .plasticity import ChabocheCombinedHardening, ChabocheState, J2PlasticState
from .quadrature import CreepQuadratureState, QuadratureField, QuadratureTransaction


@dataclass
class CreepPlasticityQuadratureState(CreepQuadratureState):
    plastic_fields: dict = field(default_factory=dict)
    plastic_trial: dict = field(default_factory=dict)
    plastic_energy_increments: dict = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        self.transaction = QuadratureTransaction(
            committed={**self.transaction.committed, **self.plastic_fields},
            trial={**self.transaction.trial, **self.plastic_trial},
            schema="agentfem.coupled-creep-plasticity-small-strain-state",
            metadata={"integration": "simultaneous_backward_euler"},
        )

    @classmethod
    def create(cls, domain, *, material, degree=2, scheme="default"):
        if domain.comm.size != 1:
            raise NotImplementedError("Coupled creep/plasticity quadrature is serial experimental.")
        base = CreepQuadratureState.create(domain, degree=degree, scheme=scheme)
        layout = {"plastic_strain": ("PE", (3, 3)), "equivalent_plastic_strain": ("PEEQ", ())}
        if isinstance(material.plastic, ChabocheCombinedHardening):
            layout["backstresses"] = ("ALPHA", (material.plastic.backstress_count, 3, 3))
        def allocate(suffix):
            return {key: QuadratureField.create(domain, degree=degree, scheme=scheme,
                                                name=name + suffix, value_shape=shape)
                    for key, (name, shape) in layout.items()}
        return cls(**{item.name: getattr(base, item.name) for item in fields(CreepQuadratureState) if item.init},
                   plastic_fields=allocate(""), plastic_trial=allocate("_trial"))

    def _point_states(self):
        committed = self.snapshot()
        for index in range(len(committed["creep_strain"])):
            plastic = dict(plastic_strain=committed["plastic_strain"][index],
                           equivalent_plastic_strain=committed["equivalent_plastic_strain"][index])
            selected = (ChabocheState(**plastic, backstresses=committed["backstresses"][index])
                        if "backstresses" in committed else J2PlasticState(**plastic))
            yield CreepPlasticityState(
                ImplicitCreepState(committed["creep_strain"][index], committed["equivalent_creep_strain"][index]),
                selected,
            )

    def update(self, strain_values, material, *, time_start, time_end, temperature_values=None):
        if not isinstance(material, CreepPlasticityMaterial):
            raise TypeError("Coupled state requires one homogeneous coupled material.")
        strains = np.asarray(strain_values, dtype=float)
        if strains.shape != self.creep_strain.values.shape:
            raise ValueError("Strain and coupled-state layouts differ.")
        temperatures = self._temperatures(temperature_values, len(strains))
        updates = []
        for index, (strain, old) in enumerate(zip(strains, self._point_states(), strict=True)):
            try:
                updates.append(material.update(strain, state=old, time_start=time_start, time_end=time_end,
                                               temperature=None if temperatures is None else temperatures[index]))
            except Exception as exc:
                raise RuntimeError(f"Coupled material update failed at local point {index}: {exc}") from exc
        # No live field is assigned until every point has returned successfully.
        trial = {
            "creep_strain": [u.state.creep.creep_strain for u in updates],
            "equivalent_creep_strain": [u.state.creep.equivalent_creep_strain for u in updates],
            "plastic_strain": [u.state.plastic.plastic_strain for u in updates],
            "equivalent_plastic_strain": [u.state.plastic.equivalent_plastic_strain for u in updates],
        }
        if "backstresses" in self.plastic_trial:
            trial["backstresses"] = [u.state.plastic.backstresses for u in updates]
        for key, values in trial.items():
            self.transaction.trial[key].assign(values)
        self.stress.assign([u.stress for u in updates])
        self.tangent.assign([u.algorithmic_tangent for u in updates])
        self.plastic_energy_increments = {
            "plastic_work": np.array([u.plastic_energy.plastic_work for u in updates]),
            "plastic_stored_energy": np.array([u.plastic_energy.recoverable_storage_change for u in updates]),
            "plastic_dissipation": np.array([u.plastic_energy.modeled_irreversible_dissipation for u in updates]),
            "plastic_backward_euler_dissipation": np.array([u.plastic_energy.backward_euler_dissipation for u in updates]),
        }
        return dict(points=len(updates), creeping_points=sum(u.creep_increment > 0. for u in updates),
                    maximum_creep_increment=max((u.creep_increment for u in updates), default=0.),
                    maximum_plastic_increment=max((u.plastic_increment for u in updates), default=0.),
                    maximum_local_iterations=max((u.local_iterations for u in updates), default=0),
                    minimum_temperature=None if temperatures is None else float(np.min(temperatures)),
                    maximum_temperature=None if temperatures is None else float(np.max(temperatures)))

    @staticmethod
    def _temperatures(values, count):
        if values is None:
            return None
        selected = np.asarray(values, dtype=float).reshape(-1)
        if selected.shape != (count,) or not np.isfinite(selected).all() or np.any(selected <= 0.):
            raise ValueError("Coupled temperatures must match points and be positive kelvin.")
        return selected

    def refresh_response(self, strain_values, material, *, temperature_values=None):
        strains = np.asarray(strain_values, dtype=float)
        temperatures = self._temperatures(temperature_values, len(strains))
        stresses = [material.stress_from_state(strain, old,
                    temperature=None if temperatures is None else temperatures[i])
                    for i, (strain, old) in enumerate(zip(strains, self._point_states(), strict=True))]
        self.stress.assign(stresses)
        self.tangent.assign(np.broadcast_to(material.plastic.elastic_tangent(), self.tangent.values.shape))
        self.rollback()

    def output_fields(self):
        # Individual backstress branches are retained at integration points;
        # exporting a rank-three tensor as one VTK field is not implied.
        return (*super().output_fields(),
                self.plastic_fields["plastic_strain"].cell_average(name="PE"),
                self.plastic_fields["equivalent_plastic_strain"].cell_average(name="PEEQ"))

    def summary(self):
        result = super().summary()
        result.update(kind="coupled_creep_plasticity_quadrature_state",
                      state_variables=self.transaction.names)
        return result
