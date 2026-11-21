# Batched plane-stress Step timing

Run from the checkout in `fenicsx-env`:

```sh
PYTHONPATH=src python validation/plane_stress_batch/benchmark.py
```

This developer benchmark reuses the independently tested synthetic material
and model fixtures. It alternates three scalar/array pairs on an 8×8 planar
patch, 256 integration points and 12 accepted increments. Both routes use the
same material equations, loading, fixed-old-state reduction and tolerances.
It checks complete final F/P/S/state/energy/full-material-tangent arrays.

`timing.json` retains every timing, accepted/attempted increment count, platform
and relevant source digests. The boundary is model construction and completed
Step execution with existing local JIT cache. Imports, field-file output and
post-solve observation copies are excluded. No wall-clock threshold is imposed
in CI. This is not a customer performance result or a universal speedup.

Numerical regression lives in `tests/test_plane_stress_array.py` and
`tests/test_prescribed_material_fields.py`: loading/unloading/hold history,
fixed-state finite differences, scaled stress units, scalar fallback, chunk
failure rollback, 2D/3D thickness parity and independent output inspection.
