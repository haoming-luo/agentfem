# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Exclusive array owner for the bounded serial response-only Procedure.

Quadrature Functions are publication buffers, not simultaneous state owners.
Only explicit publication exposes accepted values; internal field access is
outside this opt-in contract. Arbitrary callback/observer graphs are not admitted.
"""
import numpy as np
from .._resident_field_records import _freeze


class OwnedExplicitState:
    def __init__(self, response):
        from .quadrature import QuadratureField
        if response.domain.comm.size != 1:
            raise NotImplementedError("Owned explicit arrays require serial execution")
        self.response = response
        for field in (*response.state.committed.values(), *response.state.trial.values(),
                      *response.response_transaction.committed.values(),
                      *response.response_transaction.trial.values()):
            if type(field) is not QuadratureField:
                raise TypeError("Owned explicit state needs standard quadrature fields")
            field._array_indices()
            if not field._identity_layout_cache:
                raise NotImplementedError("Owned explicit state requires identity quadrature layout")
        self.capture()

    def fields(self):
        r = self.response
        return {**r._storage_trial_fields(),
                **{'committed_state_'+k: v.function for k,v in r.state.committed.items()},
                **{'committed_response_'+k: v.function for k,v in r.response_transaction.committed.items()}}

    def capture(self):
        self.accepted = {k: _freeze(v.x.array) for k,v in self.fields().items()}
        self.history = _freeze(self.response.state.committed_state_vectors())
        self.trial = None
        self.dirty = False

    def assign(self, prepared):
        response = prepared.response
        records = {'P': _freeze(prepared.first_piola_stress).reshape(-1),
                   'S': _freeze(response.cauchy_stress).reshape(-1),
                   'WORK': _freeze(response.internal_work_density_increment).reshape(-1)}
        offset = 0
        history = _freeze(response.state_new)
        for variable in self.response.state.state_schema.variables:
            values = _freeze(history[:, offset:offset+variable.size]).reshape(-1)
            records['state_'+variable.name] = values
            records['committed_state_'+variable.name] = values
            offset += variable.size
        records.update({'committed_response_'+name: records[name] for name in ('P','S','WORK')})
        self.trial = (records, history)

    def commit(self):
        if self.trial is None:
            raise RuntimeError("No owned explicit trial to commit")
        self.accepted, self.history = self.trial
        self.trial = None
        self.dirty = True

    def rollback(self):
        self.trial = None

    def snapshot(self):
        if self.trial is not None:
            raise RuntimeError("Cannot snapshot unaccepted owned explicit state")
        return {k: _freeze(v) for k,v in self.accepted.items()}

    def publish(self):
        # Publication always uses the accepted owner, never a pending trial.
        if self.dirty:
            for name, field in self.fields().items():
                field.x.array[:] = self.accepted[name]
            self.dirty = False
