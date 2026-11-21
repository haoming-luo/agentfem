# Engineering workflow completion — 2026-11-21

Branch `feat/explicit-dynamic-parity`, Draft PR #195. Existing execution,
model-review and engineering postprocessing increments are retained.

## Selected remaining work

1. Batch reflection-symmetric plane-stress reduction for existing array
   material providers; preserve full 3D responses, fixed-old-state Newton,
   Schur condensation, atomic publication and explicit scalar fallback.
2. Frozen reference-coordinate prescribed fields sampled in quadrature order,
   including physical-time updates and auditable identity. Connect spatial
   temperature to the finite-history Step without introducing a heat solve.
3. Explicit uniform reference thickness in the finite-history planar Step,
   consistently scaling virtual work, applied resultants and recorded work;
   retain the existing unit-thickness default.
4. Detached accepted material observations and ordinary two-point engineering
   gauge measurements, using existing state and sampling owners.

Automated fixtures exercise affine/reference fields, loading/unloading/history,
finite-difference tangents, failure rollback, checkpoint/restart and thickness
scaling. No customer simulation or external Abaqus parity claim is included.
Commit/push one coherent locally verified batch; retain Draft and do not merge.

## Design references

- DOLFINx Expression evaluation at integration points:
  https://docs.fenicsproject.org/dolfinx/main/python/generated/dolfinx.fem.html
- dolfinx_materials external-state mapping:
  https://bleyerj.github.io/dolfinx_materials/api/quadrature_map.html
- Existing platform plane-stress derivation, quadrature identity and
  trial/commit/restart contracts remain authoritative.

Timings will distinguish local material work from complete automated Step
execution; no fixed performance gain is assumed in advance.

## 近期开发汇总与本轮交付

| 主线 | 已交付的通用能力 |
| --- | --- |
| 显式动力学 | 节点集中力与来源身份、独立测点历史、seam 拓扑、正质量高阶单元；独立中心复合四面体已接入整网格装配、标准 Step、非线性材料更新、重启和结果输出。详细证据见 `validation/dynamic_parity/REPORT.md`。 |
| 建模与运行 | 已安装接口查询、模型审阅卡、空间变化牵引、项目缓存/目录预检、运行阶段记录与独立进程监督。 |
| 工程后处理 | 点、折线路径和截面采样，反力与合力矩，跨网格场比较，已保留收敛步提取。 |
| 本轮材料接口 | 批量平面应力局部 Newton、固定旧状态与分块原子性、逐点数值回退、完整三维材料切线与二维凝聚切线分别保留。 |
| 本轮空间输入 | `fields.prescribed` 统一标量、同网格 FE 场及参考坐标回调；冻结时间节点、线性插值、列顺序与单位、积分点身份、实际场输出和重启检查。 |
| 本轮厚度与测量 | 显式均匀参考厚度贯通虚功、反力、能量积分与单位；`accepted_fields` 暴露不可变副本；ROI 统计复用既有物理权重 reducer；`results.gauge` 区分客观标距应变与轴向投影。 |

### 退出问题的实证修复

一次只读接口查询已输出 JSON，但进程仍持续占用 CPU。原生调用栈
显示 `Py_FinalizeEx → PetscFinalize → MPI_Finalize → MPIDI_OFI_mpi_finalize_hook`
并在 libfabric socket progress 中循环。这是该次实例的直接证据，不能
泛化为所有机器、所有未退出问题的共同原因。

修复普通串行元数据输出意外初始化 MPI 的路径；常用简单接口签名读取
已安装源码，不执行默认表达式、装饰器或用户程序。动态声明和已加载模块
保留运行时查询，结果明确记录 `inspection`。MPI 作业和已启用的通信器
继续按真实 rank 输出。无网络规避参数的实际查询现可正常退出；独立测试
在禁止 MPI/PETSc/DOLFINx 导入时验证了常用查询及错误分支。

另外修复 `python -m agentfem.cli mpi-run --timeout ...` 被普通 `run`
监督选项误拦截的问题。没有更改用户全局 MPI 网络设置、禁用 MPI 正常
收尾或以强制退出冒充计算完成。

### 本地验证

- 材料、输出、状态、重启及接口的扩大回归：557 项通过。
- 后续入口/接口修正的相关回归：167 项通过；独立源码查询测试 13 项通过。
- 双 MPI 进程回归：每个进程各 70 项通过，包含真实 CLI、模型卡、新旧后处理及空间映射。
- 检查包括加载/卸载/保载、单位尺度 1e-200 至 1e200、差分切线、分批调用失败回滚、标距刚体转动、二维/三维薄片反力和变形对照，以及独立 XML/HDF5 温度场读取。
- 完整三维材料切线单独保存，凝聚切线仅供二维装配；默认单位厚度与显式物理厚度的力单位不混用。
- Ruff、生成文档/科学知识卡、严格文档构建通过。以上测试批次有重叠，不相加作为独立测试总数。
- 远端完整 CI 证据随同一 Draft PR #195 的当前提交记录；保持未合并状态。

### 实测性能与适用范围

`validation/plane_stress_batch/benchmark.py`
交替运行三对逐点/批量完整 Step，使用相同合成 Neo-Hookean 材料与历史时钟、
8×8 四边形网格、256 个积分点、12 个接受增量。已有本机 JIT 缓存；
不计 Python 进程导入、场文件写出和接受后观察复制。原始逐次计时、环境和
相关源码摘要在同目录 `timing.json`。

| 测量边界 | 逐点中位耗时 | 批量中位耗时 | 比值 |
| --- | ---: | ---: | ---: |
| 完成全部求解 | 4.782 s | 0.155 s | 约 31 倍 |
| 建模＋完成全部求解 | 4.833 s | 0.173 s | 约 28 倍 |

F、P、S、AGE、SENER、完整材料切线六类场在该记录中最大绝对差均为 0。
剖析还定位并移除了批量路径中每点重复的几何检查，没有改变其判据。
这是小型通用验证模型的实测，不能外推为客户材料、任意网格或含输出任务
的固定加速倍数；所有交替原始样本均保留。

### 保留的产品边界

本轮仅交付经验证的均匀厚度、参考坐标预设输入和串行隐式历史材料 Step。
当前坐标外部场、变厚度、热/扩散双向反馈以及该隐式材料 Step 的 MPI
求解未被静默启用。通用场映射、标距及已有后处理有各自的 MPI 证据。
C3D10M 专用公式等价性与外部 Abaqus/叶片对标仍与独立单元开发区分；
本轮不运行客户仿真，不用独立公式替代未取得的外部验证。
