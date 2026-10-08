# Bounded thermoelastic installed-wheel acceptance

- Source commit: `9cd7e644` (clean working tree at build).
- Wheel: `agentfem-0.4.0-py3-none-any.whl`, 1,807,617 bytes.
- SHA-256: `0e3236444c49db5faeba3e407240840ed914e6f0cf5f48fff248666b9134df02`.
- This is a development acceptance artifact with the existing package version,
  not a replacement upload or a new release of 0.4.0.
- Runtime: local macOS ARM64 `fenicsx-env`; dependencies reused, package installed
  with `pip --no-deps --no-index --target` in a fresh temporary directory.
- Import location was asserted to be that target, not the development source.

## Executed checks

1. `examples/coupled_thermoelastic_3d.py`, serial: passed analytical temperature
   check and wrote final XDMF/HDF5 plus result manifest.
2. Same example from the same installed wheel, two MPICH ranks through
   `agentfem mpi-run -n 2 --timeout 120`: passed and wrote output.
3. `tools/benchmark_coupled_decay.py`, same installed wheel: all six cases and
   fixed spatial/time order thresholds passed. Temperature spatial orders were
   2.007765 and 2.128687; displacement 1.976102 and 2.070307. Temperature time
   orders were 0.985105 and 1.026396; displacement 1.002279 and 1.063052.

The examples and benchmark driver are repository files; neither inserts the
repository source into the import path. The working directory was outside
the checkout. This tests package contents against an existing numerical
runtime; it is not a fresh-machine dependency installation test, Windows
acceptance, or general multiphysics validation.

## Reproduction

Build from the stated commit using `pip wheel --no-deps --no-build-isolation`,
install that wheel into a fresh target, set `PYTHONPATH` to that target only,
and run the two files above from outside the checkout. Use the runtime's own
MPI launcher through `agentfem mpi-run`, with one OpenMP/BLAS thread per rank.
Never substitute a newly built wheel while retaining this artifact's hash.

The coupled capability remains experimental: shared 3D mesh, homogeneous
constant isotropic material, SI inputs, strong boundaries and fixed natural
loads. Final snapshots are not a transient field series; the linearized heat
and quadratic identities are not a general thermodynamic first law.
