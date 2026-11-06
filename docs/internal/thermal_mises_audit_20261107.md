# Thermal assessment invariant correction

The older `creep_hot_wall_assessment.py` demo used a two-dimensional deviator
with trace divided by two for a plane-strain von Mises assessment. That is
not the three-dimensional J2 invariant: the constitutively constrained
out-of-plane stress must participate. The core standard result projection
already performs this reconstruction, including thermal eigenstrain.

The demo now consumes that existing `MISES` field and exports it alongside
displacement, temperature and stress. It uses the maximum DG0 cell-average
invariant, explicitly, rather than adding another hand-written stress formula.
This migration does not change equilibrium, thermal conduction or the K-R law.

## Independent checks

Four affine planar tests (plane stress/strain, with/without thermal expansion)
compare standard output with a full 3D Hooke stress and its principal-stress
invariant. They pass at 1e-12 relative tolerance. On the actual reduced hot-wall
mesh, an independent full-3D thermal strain reconstruction gives a maximum
cell-average value of 554397427.2293961 Pa; the maximum pointwise difference
between the two projected fields is 4.18e-7 Pa. Corrected degree-4 quadrature
sampling gives 554942269.3512055 Pa, showing separately the averaging effect.

## Versioned regression, not new physical validation

| Quantity | Old v1 | Corrected v2 |
| --- | ---: | ---: |
| Governing stress, Pa | 331321497.13443035 | 554397427.2293957 |
| Illustrative rupture estimate, h | 31308.280996458638 | 2386.718875046755 |
| Damage at 90% of illustrative rupture time | 0.4376507671 | 0.4376507671 |

The unchanged final damage follows the chosen normalized endpoint; it is not
independent validation of the rupture estimate. The old Golden rejected both
changed quantities before its reference was explicitly versioned to
`q1-6x10-heat4-kr40-plane-strain-mises-2`. Neither reference is an experimental
material fit, and a cell-average maximum is not hotspot mesh convergence.
This finding concerns the older illustrative hot-wall demo, not the separate
P92 research material or the composite pressing simulation.
