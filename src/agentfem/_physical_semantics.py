# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Physical definitions shared by load construction and installed discovery."""


def pressure_configuration(value):
    selected = str(value).lower().replace('-', '_')
    if selected in {'reference', 'dead', 'nominal'}:
        return 'reference'
    if selected in {'current', 'follower', 'spatial'}:
        return 'current'
    raise ValueError('pressure configuration must be reference or current.')


def load_semantics(kind, *, configuration='reference'):
    current = kind == 'pressure' and pressure_configuration(configuration) == 'current'
    return {
        'integration_configuration': 'reference',
        'magnitude_area': 'current' if current else 'reference',
        'direction': ('current inward normal' if current else 'reference inward normal')
                     if kind == 'pressure' else 'supplied vector components',
        'follower': current if kind == 'pressure' else None,
        'formula': ('-p J F^{-T} N' if current else '-p N') if kind == 'pressure' else 't dot v',
        'two_dimensional_convention': 'planar continuum resultant per unit reference thickness unless an explicit section is applied',
        'expression_dependence': 'explicit supplied expression; displacement-dependent traction is not automatically classified as dead',
    }


def callable_semantics(name):
    leaf = name.split('.')[-1]
    if name.startswith(('loads.', 'models.Model.')) and leaf in {'traction', 'pressure', 'surface_force'}:
        if leaf == 'pressure':
            return {key: load_semantics('pressure', configuration=key) for key in ('reference', 'current')}
        info = load_semantics('traction')
        if leaf == 'surface_force':
            info.update(formula='resultant / reference_measure', distribution='uniform')
        return info
    return None


def model_physics(model, *, options=None):
    from ._model_support import describe
    selected = dict(options or {})
    materials = ([selected['material']] if selected.get('material') is not None
                 else [getattr(record, "item", record) for record in getattr(model, "materials", ())])
    return {
        'materials': [describe(item) for item in materials],
        'loads': [describe(item) for item in getattr(model, "loads", ())],
        'displacement_reference': 'mesh supplied to this model; if imperfected, displacement excludes the imposed imperfection',
        'initial_imperfection_amplitude': {'status': 'not_checked', 'reason': 'requires original geometry and an explicit imperfection record'},
        'section': {'status': 'not_checked', 'reason': 'no universal section contract inferred; inspect selected provider',
                    'planar_default': 'unit reference thickness; do not interpret per-length resultants as total forces'},
        'units': describe(getattr(model, "unit_system", None)),
    }
