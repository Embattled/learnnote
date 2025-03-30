TetSphere Splatting: Representing High-Quality Geometry with Lagrangian Volumetric Meshes

# 1. Introduction

将近年基于深度学习优化的, 几何的场景表示分成两大得
* Eulerian 欧拉表示
  * a set of pre-defined, fixed coordinates in 3D world space
  * 每一个 coordinate position 都关联一系列的属性
  * 即连续表达, 例如 volume or distance from the surface
  * NeRF 属于这一种, 此外用于重建 Mesh 的 NeuS 系列同理, 另外基于插值的 grid 表示在计算上
  * 欧拉表示是需要权衡 tade-off 计算量和重构质量
    * 高质量高精度的隐式几何表示需要 高容量的网络或者高密度/解析度的 Grid
    * 这种权衡往往导致对于细的, 薄的物体重构效果受限
* Lagrangian 拉格朗日表示
  * tracking the movement of a set of geometry primitives in 3D world space
  * 在3D空间中追踪一个几何元件的移动, 同时调整元件的几何属性
  * 当前最经典的拉格朗日方法就是 3D GS, 初次之外还有用于Surface 重构的 surface triangles
  * 这种方法通常从原理上的计算量少于 欧拉表示
    * 但是受限于 mesh 的表示质量
    * 因为对于每一个元件都是独立的追踪并计算的, 因此缺少全局的一致性约束
    * 例如 3D GS, 由于点云坐标的移动方向是完全3D自由的, 因此常常导致重构的表面非常噪
    * 而 surface triangles, 对于表面的连通性缺少约束, 因此常常生成非流体或者不规则(non-manifold surfaces or irregular)的表面
      * 这种问题对于下游任务是非常致命的 rendering simulation

本文的方案主要聚焦于生成高质量的 Mesh 几何

fact that existing Lagrangian primitives are too fine-grained to ensure high-quality meshes
作者的意见是当前的拉格朗日方法的基元都太 细碎, 难以生成高质量 Mesh
* 高质量的 Mesh 并不单单依赖于 独立的 基元
  * 例如, Surface triangles 需要互相对齐, 能够形成流体表面取决于它们的连接精度


TetSphere
* Tetsphere 的定义是一个 volumetric sphere 体积球
  * 初始化为一个单位球
  * 球是通过 tetrahedralization 四面体化 的规则通过一个点集来组成
* 对于 shape 的整体优化通过作用于 体积球上, 并传导到点集
  * (好像意思是一个点可以属于多个四面体)
* 将 TetShpere 的变形优化 解构为 几何能量优化问题 geometric energy optimization problem
  * 由  可微的渲染 Loss 
  * bi-harmonic energy of the deformation gradient field 变形梯度场的双谐波能量
  * and non-inversion constraints  非反演约束
  * all effectively solvable via gradient descent.

效果评价: 在两个任务上进行评价  多视角重构 和 单视角重构
* 在评价指标上提出了 3 个新指标, 专门用来评价网格的质量, 关注 3D 模型在后期的可用性
  * surface triangles uniformity        : 表面三角性均匀性
  * manifoldness                        : 流行性
  * structural integrity                : 结构完整性

结论:
TetSphere 针对 Mesh 的质量取得了非常领先的结果, 同时针对其他的传统评价指标也取得了不弱的成绩
同是展示了用 TetSphere 在下游任务的可用性


# 2. Related Work

Eulerian and Lagrangian geometry representations: 
* 介绍了这两个概念最初源自于  computational fluid dynamics 计算流体动力学 (WTF?)
* 欧拉-view 分析在空间中特定点的流体
  * NeRF, NeuS
  * 基于 Grid 的方法
  * 特点是输入坐标到神经网络, 在推理的时候可以获得无限的分辨率, 缺点是超慢的训练速度
* 拉格朗日方法 则关注特定流体粒子
  * 3DGS 的高斯微元
  * MeshAnything 以及 Dmesh 的 surface triangular


 