# Linear solid-column buckling

Run `python case.py` in an installed AgentFEM/FEniCSx environment with SLEPc.
The case owns geometry, material and loading only; the public Step owns
geometric stiffness, constraint elimination and eigensolution.

Teaching prompt: Find the first three buckling modes of this clamped-free
solid column. Double the reference traction and explain the factor change.
Compare two lengths and show the physical critical stress, not just lambda.

Expected answer: the first critical stress is about 0.513145 in the declared
consistent units, within 1% of the slender Euler reference. Doubling reference
traction halves lambda and preserves critical stress. Doubling length
approximately quarters critical stress. Relative mode amplitude is not a
postbuckling displacement or a physical imperfection.

The standard run folder contains a result manifest and ParaView XDMF/HDF5.
