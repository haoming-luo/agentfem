# Thermal hold and reversal

Run `agentfem run --name demo` in this directory with a version exposing
`constitutive.CreepPlasticityMaterial`. The case uses only public modeling
interfaces; no project solver, generated adapter or AI runtime is required.

This small serial 3D example uses **synthetic** parameters, a prescribed accepted
temperature history (800 to 900 K), and axial loading, hold and reversal. It
demonstrates simultaneous creep and Chaboche plasticity through the existing
physical-time implicit Step. It does not solve a heat-transfer problem or
calibrate a real material. The output contains separate CE/CEEQ, PE/PEEQ,
backstress, stress and energy histories, a final visualization and checkpoint.

The mechanical ledger excludes prescribed thermal energy transfer. No fatigue
life, fracture, contact or MPI claim follows from a successful run. Refine the
physical increments before drawing quantitative conclusions. Do not reinterpret
a partial run or a small demonstration as an industrial validation.
