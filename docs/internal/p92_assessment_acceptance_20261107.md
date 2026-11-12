# P92 assessment composition acceptance

This is software integration evidence, not P92 engineering qualification or
a release. Local candidate `55daef4113baafd6841260c3a30c83c703e77157`
combines the exact heads of #170, #171, #173 and #176 with the reviewed source
notes. The only merge conflict was adjacent explanatory text in CONCEPTS;
both statements were retained. No numerical implementation was changed by
conflict resolution.

## Numerical and installed checks

- 462 targeted tests passed in 82.06 s: P92 references, structural paths,
  small-strain material outputs, selected-point histories, thermal Mises,
  release goldens, rupture curves and creep/fatigue assessments.
- An archived checkout of that commit produced a pure-Python wheel with native
  build disabled. SHA256:
  `c0866950de60d52252e4692a3e00675ced64f22a6649eee85f8b7c85ccee33ce`.
  It was installed into a temporary target with existing FEniCSx dependencies;
  the imported package path was checked. This is not a clean-OS installation
  or a native-accelerator benchmark. No user environment was replaced.
- The installed console command ran the ordinary small-strain relaxation
  project continuously and from its midpoint checkpoint. All three selected
  material histories and seven H5 datasets were exactly equal.
- VTK 9.6.2 read each XDMF as one 16-cell unstructured grid. Warp by `U`
  produced the expected coordinates and preserved `S_CELL` and `EV_CELL`.
- The installed package ran the four-cell P92 bending/hold research model.
  A **synthetic**, independent rupture relation `t=(200/sigma)^3 h` checked
  composition, not material life. Its reviewed table was explicitly converted
  from MPa/h into the recorded model stress unit and seconds. Assessment
  consumed co-located stress and temperature without modifying their arrays.
- For piecewise-linear sampled stress, the independently integrated cubic
  damage rate gives `0.007704128576210693`. Subdivisions 1/2/4/8 give
  `0.007706076877045165`, `0.007704615651419317`,
  `0.007704250345012856`, `0.007704159018411241`.
  Error ratios are four to rounding accuracy. Restarted and continuous
  assessment records are exactly equal. This checks quadrature on the sampled
  path, not source-time discretization or an empirical P92 rupture law.

Numerical processes were serial with TCP fabric and BLAS/OpenMP threads limited
to one. The initial attempted `python -m agentfem` invocation did not run a
case because the package has no module entry point; the installed `agentfem`
console command above is the supported entry point.

## Follow-up found during acceptance

The teaching material declared Pa/seconds but its example Model did not declare
units. #178 adds SI identity, leaving coefficients and geometry unchanged, and
tests the real case's 16 accepted history samples. Its physical-time suite
passed 34 tests; the dedicated two-rank admission test was skipped in serial.
Checkpoints from the old unit-undeclared example are not interchangeable.

The unified candidate still needs latest remote CI before integration. No
administrative bypass of branch protection is part of this acceptance.
