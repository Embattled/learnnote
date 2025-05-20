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
* 介绍了这两个概念最初源自于  computational fluid dynamics 计算流体动力学 (WTF?) 2002 年
* 欧拉-view 分析在空间中特定点的流体
  * NeRF, NeuS
  * 基于 Grid 的方法
  * 特点是输入坐标到神经网络, 在推理的时候可以获得`无限的分辨率`, 缺点是超慢的训练速度
* 拉格朗日方法 则关注特定流体粒子
  * 3DGS 的高斯微元
  * MeshAnything 以及 Dmesh 的 surface triangular


3D object reconstruction
* 早期的方法结合了 2D 图像编码 和 3D 解码器 的技术, 解码器本身是在有显示表示的3D 数据上进行的, 例如
  * voxels
  * meshes
  * point clouds
* 隐式表达
  * NeRF
  * occupancy networks
* 近期的新趋势是
  * 2d generative models for 3D reconstruction
    * 使用 SDS 以及 其他补充的 losses
  * single-view reconstruction
    * 使用生成式规则通过单张图片还原3D模型
* 拉格朗日方法通常使用 single surface sphere


# 3. TETSPHERE SPLATTING

四面体飞溅: 使用 tetrahedral spheres 作为 primitive
* 保持了几何完整性 Geometric integrity
* 在 Mesh 的内部施加了几何正则性 geometric regularization 来提高了 Surface 的质量
* 将重构任务通过 Tetsphere splatting 建模为 四面体球体的变形 deformation of tetrahedron spheres
* 同一组四面球体开始, 调整顶点的位置, 使得渲染图像与目标的多视图图像对级
  * 顶点的移动收到 几何约束的限制
  * regularization : 
    * 惩罚 non-smooth deformation (via bi-harmonic energy)
    * prevent the inversion of mesh lelments (via local jnjectivity) 通过局部单射防止网格元素的反转
      * 这个策略已经被证明在 生成的四面体网格上 能够实现 高质量与结构完整性


## 3.1. TETRAHEDRAL SPHERE PRIMITIVE

四面体球元素

tetrahedralized sphere (TetSphere)


四面体球体是一个大号的球体, 初始化的时候为球, 里面包含多个四面体

四面体球体有 N 个顶点和 T 个四面体, 通过应用有限元法 Finite Elementt Method (FEM)
* Mesh of each sphere 由四面体元素构成
* 每一个四面体构成一个 3D discrete piecewise linear volumetric entity 三维离散分段线性体积实体

第 i 个变形球体网格中的所有顶点的位置向量  
position vector of all vertices of i-th deformed  sphere mesh as $x_i \in \mathbb{R}^{3N}$

第i个球体中第 j 个四面体的变形梯度记作 3x3矩阵, 能够定量的描述四面体的形状变化
deformation gradient of the j-th tetrahedron in the i-th sphere is $F_x^{(i,j)}\in \mathbb{R}^{3\times 3}$


相比起单个 sphere, 四面体使用 多个 sphere 来高精度的表示 shapes
* Complete shape 是所有球体的 union 并集, 确保了 local region 是能够独立的描述细节, 实现高精度的表示
* 这种表示允许shapes with arbitrary topologies
  * 由 流行形状 的副属性所保证的  paracompactness property manifold shapes

同目前常见的表示相比, 优势有:
* 对比 NeRF 隐式神经表示: not rely on neural networks, 具有快速优化的能力
* 对比其他欧式表示 (e.g. DMTet), 避免了 iso-surface extraction : 由于预定义的分辨率不足导致的网格质量降低
* 对比其他拉格朗日方法 (e.g. 高斯点云, 三角形Mesh ), 四面体球表示能够提供 volumetric representation 的能力
  * 在四面体的顶点之间施加进一步的约束, 提高了网格质量


## 3.2. TETSPHERE SPLATTING AS SHAPE DEFORMATION

通过改变初始的 TetSphere 的顶点位置来进行变形, 实现重构, 这个过程受两个目标的指导
* 渲染精度: deformed TetSpheres align with the input multi-view images
* 几何约束: maintaining high mesh quality that adheres to necessary geometry constraints

使用 双调和能量 (bi-harmonic energy) 来量化 整个场中的平滑度能量 (energy quantifying smoothness)
* 这种 几何正则化能够确保在变形过程中 变形梯度场的平滑性, 防止出现不规则网格或者颠簸的表面    
* 这种双调和正则化并不会导致 最终梯度的过渡平滑  
* 该种能量针对的是变形梯度场, 该场测量的是顶点位置的相对变化, 而不是绝u地位之,  这种方法允许保留尖锐的局部几何细节, 类似于物理模拟中使用的技术  

引入了一个几何约束, 以确保所有变形元素的局部 单射性, 确保了元素在变形过程中能够保持其方向, 防止反转或者内部外部配置  
* `det(F_x(i,j))>0`

这两个方法可以应用于任何四面体网格


假设整个 shape 由 M 个四面体球体构成, 则整个 shape 的顶点数为  $x=[x_1,...,x_M]\in \mathbb{R}^{3\times N\times M}$  

所有四面体的形变记作 $F_x \in \mathbb{R}^{9MT}$

biharmonic energy, 拉普拉斯矩阵定义在 connectivity of the tetrahedron faces
在双谐能量中, 拉普拉斯矩阵是基于四面体之间的连通性定义的  

$$L \in \mathbb{R}^{9MT \times 9MT} $$

每一个 block 都是对称的 $L_{p,q} \in \mathbb{R}^{9\times 9}$
* 所有 $p \neq q$ 的矩阵
  * 如果 p, q 四面体有共享的 triangle, 则设置为 负的单位矩阵 $-I$
* 所有 $p = q$ 的矩阵, 设置为 $kI$, k 是 Number of neighbors of the p-th tetrahedron


最终的优化条件是  

$$\underset{x}{\text{min}} \Phi(R(x)) + ||LF_x||^2_2$$
* R(.) 是渲染, $\Phi$ 是渲染 Loss
* $LF_x$ 是 归一化 bi-harmonic energy across the deformation grident field



# 4. TETSPHERE INITIALIZATION AND TEXTURE OPTIMIZATION

**TetSphere initialization**

寻找四面体球体的最优初始化位置

















