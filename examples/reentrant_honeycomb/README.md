# Periodic cell and connected finite honeycomb

Run `python case.py` with optional Gmsh and dolfinx_mpc installed. Geometry
construction, static solving and strain measurement use separate public APIs.
The native solid constituent has E=1000 and positive nu=0.3.

Teaching prompt: Compute directional periodic properties, then stretch the
connected 3x3 array with transverse motion free. Explain why the finite-array
apparent Poisson ratio differs from the periodic-cell value. Check energy,
mesh size and strain amplitude before interpreting a negative ratio.

Expected answer: for this geometry periodic nu_xy is about -1.433 and the
finite-array gauge value about -1.303. Negative response comes from structure,
not a negative constituent Poisson ratio. The periodic cell is normalized by
full envelope volume; finite specimens remove detached boundary fragments.
These values are numerical examples, not universal honeycomb constants.

Outputs include periodic stiffness/compliance and consistency records,
finite-array XDMF/HDF5, gauge definitions and support reaction.
