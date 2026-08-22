# AgentFEM PDEAgent-Bench report

- Cases: **645**
- Passed: **555 (86.0%)**

## Equation families

| Family | Passed | Total | Pass rate | Median error | Median time |
|---|---:|---:|---:|---:|---:|
| biharmonic | 57 | 57 | 100.0% | 9.140e-09 | 1.581 s |
| burgers | 41 | 43 | 95.3% | 7.162e-03 | 0.800 s |
| convection_diffusion | 68 | 84 | 81.0% | 1.959e-07 | 0.450 s |
| heat | 40 | 50 | 80.0% | 3.731e-04 | 0.332 s |
| helmholtz | 52 | 62 | 83.9% | 1.716e-07 | 0.925 s |
| linear_elasticity | 53 | 63 | 84.1% | 7.913e-07 | 0.367 s |
| navier_stokes | 24 | 28 | 85.7% | 6.874e-07 | 2.368 s |
| poisson | 77 | 91 | 84.6% | 6.007e-07 | 0.241 s |
| reaction_diffusion | 64 | 64 | 100.0% | 3.512e-04 | 1.725 s |
| stokes | 37 | 61 | 60.7% | 7.768e-06 | 1.105 s |
| wave | 42 | 42 | 100.0% | 3.083e-06 | 0.798 s |

## Spatial dimensions

| Dimension | Passed | Total | Pass rate |
|---|---:|---:|---:|
| 2D | 553 | 586 | 94.4% |
| 3D | 2 | 59 | 3.4% |

## Failure taxonomy

- ACCURACY_FAIL: 85
- TIME_FAIL: 5
