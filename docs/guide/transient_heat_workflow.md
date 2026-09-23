# Complete transient heat workflow

`examples/transient_heat_decay.py` is a runnable 2D/3D example using the public
Study, model, field, constraint, step and result APIs. It includes initialization,
time advancement, center-temperature history, field output and an analytical
comparison. The application does not manually copy time-step arrays or assemble
a new matrix in a Python loop; the selected platform procedure owns that work.

```sh
python examples/transient_heat_decay.py --dimension 2 --cells 12 --steps 72 --output heat_2d.xdmf
python examples/transient_heat_decay.py --dimension 3 --cells 12 --steps 72 --output heat_3d.xdmf
```

Use an installed development environment, or run from the source parent as
described in the repository workflow. The script does not modify Python paths.

The domain is the unit square/cube. Conductivity, density and heat capacity are
one. All faces are fixed at 300 K; the initial excess temperature is the product
of `sin(pi*x_i)`. The exact temperature is
`300 + exp(-dimension*pi**2*t) * product(sin(pi*x_i))`.
The reported relative L2 error is normalized by the **excess temperature**,
not by the 300 K background, which would conceal error in the decaying signal.

The example uses implicit Euler and first-order tensor-product elements.
Refine both mesh and time step to check combined convergence; this alone does
not measure the spatial or temporal convergence order separately.

`expressions.interpolate` initializes the field using a coordinate expression;
`pi` is built in, and must not be passed as a user parameter. The material's
reference temperature is absolute kelvin. In this thermal-only Study the
mechanical coefficients of the shared thermoelastic material are not used.

The XDMF/HDF5 pair contains the computed fields. The neighboring result JSON
contains the center probe history and the reported relative error. Failed or
non-finite runs exit unsuccessfully. A finite result is not by itself a claim
that a chosen grid meets an engineering tolerance: inspect the reported error.
