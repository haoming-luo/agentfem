# Imperfect elastic cantilever column

Run this standard project using `agentfem run` from its directory. The installed
package must include `constitutive.saint_venant_kirchhoff`.

The 200 × 10 × 10 mm specimen has E=210000 MPa, nu=0.3 and a stress-free 0.1 mm
first-mode transverse imperfection. The left end is fixed; a 10.7 kN reference
dead load acts on the right, with no constraint suppressing end rotation.

Outputs: actual displacement XDMF/HDF5, CSV of load, additional lateral tip
movement and support reaction, and the standard result manifest. The 2D reaction
is multiplied by the 10 mm thickness. Displacement is relative to the imperfect
stress-free geometry. Euler's 10.79488 kN value is a slender ideal-column
reference, not the nonlinear curve or an exact 2D-solid critical value.

Suggested prompt: “Run the geometrically nonlinear elastic column, plot its
load–sideways displacement curve, and compare mesh refinements. Preserve the
free rotation of its loaded end.”

See `docs/guide/geometric_elasticity.md` and
`validation/geometric_elasticity/REPORT.md` for equations and verification.

## Classroom prompt (中文)

用 AgentFEM 分析带初始弯曲的弹性悬臂压杆。长200 mm、截面10×10 mm，
弹性模量210000 MPa、泊松比0.3；左端固定，右端沿原轴线施加0至10.7 kN
的压力并保持自由转动。使用StVK几何非线性材料，初始横向缺陷分别取
0.1和0.2 mm。输出载荷—附加侧移曲线、真实比例变形和支反力，并检查
网格及加载步长收敛。参考实现为本项目case.py；不得使用小变形曲线替代。

## Consigne pédagogique (français)

Analyser une colonne encastrée-libre de 200×10×10 mm, E=210000 MPa et nu=0,3,
avec le modèle de Saint-Venant–Kirchhoff. Appliquer une charge morte axiale
jusqu'à 10,7 kN, sans bloquer la rotation de l'extrémité libre. Comparer deux
imperfections initiales de 0,1 et 0,2 mm. Fournir les courbes charge–déplacement
latéral additionnel, les réactions et les champs déformés à l'échelle réelle.
Vérifier la convergence du maillage et du pas de charge.
