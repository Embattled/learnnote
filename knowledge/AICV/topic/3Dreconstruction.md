# 1. Image-based 3D reconstruction

基于图像的3D重构

一个非常大的概念, 根据重构出来的内容有很多细分的领域
* 重构 3D 点云 (SfM系列)
* 重构新视角的 RGB (只重构可视化)
* 生成 Mesh 完全的 3D 模型 

基于 2D 图像的重构一般需要每张图像的相机位姿


无论如何, 3D 场景目前最终总是以 2D 图像可视化的, 因此 3D 的存储本身是一个中间表达, 除了表达方法以外, 最终的渲染方法也很重要
* 显式 Explicit representation : 例如 Mesh, Point Cloud, Voxel, Volume 等等, 对场景进行显式建模, 但是这些显式类型一般都是离散的, 有精度问题
* 隐式 Implicit representation : 用一个函数来描述几何场景, 一般是一个不可解释的 MLP 模型, 输入 3D 空间坐标, 输出对应的几何信息, 是一种连续的表示 (Neural Fields, 神经场)  


## 1.1. View Synthesis

视角合成任务, 通过输入对一系列对物体不同角度的图像, 来生成新的角度下的图像
* 对于一个训练好的模型
* 通常输入的数据是一系列图像, 并且带有对应的角度数据, e.g. 空间坐标 (x,y,z) 视角 (theta, phi)
* 输出是 and view-dependent emitted radiance at that spatial location, 可以直接理解成新视角下的图像


传统算法
* 根据目标相机位姿和视角 生成 reprojection function, 将所有输入图像的能用得上的点重投影到目标视角下
  * 因此需要将所有输入图像存储到 GPU 是早期方法的缺点
  * 对于 unreconstructed regions 无法表示 或者有 over-reconstruction 问题

基于 points 的 volumetric representation 的方法 (传统方法)
* 极度不连续的缺点
  * 容易导致 over or under reconstruction
* 基于 CNN 的算法, 求解空间中的 points 的特征


基于MLP神经网络的 volumetric representation 的方法
* 最早由 Soft3D 提出 [Penner and Zhang 2017]
* 体 ray-marching 在 19 年提出  [Henzler et al. 2019; Sitzmann et al. 2019] 
  * 加入了 连续的密度场, continusous differentiable density field
  * 查询体积需要大量采样, 渲染成本相当高
* NeRF 的优化点
  * 在既存方法上 加入了双层 inference 采样方法和 空间位置编码方法  极大的提高了质量




## 1.2. 3D shape/suface representations

* map xyz coordinates to signed distance functions or occupancy fields
* 这种方案最早的时候需要 GT 3D geomerty, 因此在研究中经常使用 synthetic 3d shape
* 后来有直接输出每个坐标对应的 feature vector 和 RGB function, 在通过复杂的 rendering function 得到2D img 再计算 Loss

# 2. Volumetric Rendering 体渲染

原本属于计算机科学的概念

尽管 NeRF 很火但是体渲染的概念是独立的, 很传统的, 且和其他 point-base 的重构方法是共通的
* 在渲染过程的权重计算上不太一样


## 2.1. 3D Gaussian Splatting

期刊文章, SIGGRAPH 2023, 视角合成任务的 里程碑手法

从大的方向上来说属于 **Differentiable point-based rendering**


全新的 3D 场景表达方法, 不算是神经隐式表达, 构造了显式表达并推算出了每一个项的导数
指出了 NeRF 表达的不足
* 虽然连续 MLP 的表达方法有助于拟合
* 但是在渲染的时候需要随机采样空间点, 这不仅耗时而且容易产生噪声

3D Gaussian Splatting 的属性
* 3D position
* opacity $\alpha$
* 各项异性协方差 anisotropic covariance
* 球面谐波 (spherical harmonic SH) 系数

学习过程
* adaptive density control steps 自适应密度控制, 用于在优化过程中直接添加或删除 3D Gaussian Splatting 实体

渲染过程
* tile-based rasterizer

### 2.1.1. vanilla

Kerbl B, Kopanas G, Leimkühler T, Drettakis G. 3D Gaussian Splatting for Real-Time Radiance Field Rendering. ACM Trans. Graph.. 2023 Aug 1;42(4):139-.


23年后半的文章 原始 Gaussian Splatting 可以实现
* mip-nerf360 质量的效果
* 学习速度等同于 instant-ngp 和 plenoxels
  * plenoxels  是基于 voxel 的
  * instant-ngp 是基于 hash grids 的

文章提供了 3个要点
* represent the scene with 3D Gaussians 
  * 即能同NeRF一样提供空间的连续体辐射
  * 避免对 empty space 进行不必要的学习
  * Gaussian Render 的本来特征
* interleaved optimization/density control
  * 优化 各向异性协方差
  * 特殊的优化方法, 剪枝, 复制等 (Splat)
* visibility-aware rendering algorithm



同一作者的基石论文
* (2021) Point‐Based Neural Rendering with Per‐View Optimization. (2D?)
* (2022) Neural Point Catacaustics for Novel-View Synthesis of Reflections
相关论文
* Pulsar: Efficient Sphere-based Neural Rendering
* 2D 思想的 3D 渲染
  * 假设空间中的 point representation 是一个 2D 的平面圆外加一个法线 (和三角渲染类似?)
  * 缺点: SfM 中获得的点云本社非常稀疏, 很难从中回复高精度的法线
  * 而本文的 3D 高斯则不需要法线
  * (2019)Differentiable surface splatting for point-based geometry processing
  * (2021)Point‐Based Neural Rendering with Per‐View Optimization.


高斯 Representation
* project our 3D Gaussians to 2D for rendering
  * 文章讨论了在之前的方法中, 使用了传统方法 `Zwicker et al. [2001a]` 来把 3D 高斯投影在 2D 中, 而 2D 高斯的可视化就简单得多了
  * 然而在优化的时候, 因为使用梯度下降算法, 所以很容易导致 高斯的协方差矩阵失去物理意义, 即协方差矩阵不正定

优化: 调整 Gaussian 的数量和大小位置
* 重建不足的区域 under-reconstructed 的小 Gaussian :
  * 比起直接调整 Gaussian 的大小
  * 克隆临近的 Gaussian, 维持相同大小, 只移动方向
  * 增加系统总体积和高斯数量
* 大 Gaussian, 分割成小的
  * 维持系统体积, 增加高斯数量


tile-base rasterizer
* radix sort at the beginning for each frame
  * 将所有 Gaussian 点按照 projected depth 排序
* screen 分割成 16x16 共计 256 tile
  * 对于每个 Gaussian, 统计该高斯所覆盖的 tile 的个数 (number of tiles they overlap)
  * 根据 view depth 和 tile ID 生成 key
  * 不存在 per-pixel ordering of points, 都是 per-tile的
  * 根据 key 排序所有高斯
    * 具体的 key 编码为 64 bit
      * lower 32 bit 编码 depth
      * higher 32 bit 编码该 Gaussian 所覆盖的 tile
      * Depth ordering is thus directly resolved for all splats in parallel with a single radix sort (???)
* for each tile, 有了一个根据 depth 的 sorted list
  * 根据该排序的结果进行 blending
  * 每一个 tile 启动一个 thread block
  * thread block 整体的读取对应的 Gaussian instance list 到共享的内存
  * 从前到后计算 tile 中每个像素对应的 alpha 和颜色, 从而最大化使用 GPU 的并行
  * 只要有一个 alpha 累加到了对应的 saturation, 则 thread 停止
    * 定期查询一个 tile 对应的 thread block 中的所有 thread, 如果所有像素都 saturated 则块停止
  * alpha 的饱和与否是线程停止的唯一准则 (这有可能是对于无限远空间的一个漏洞)
    * 作者明确表示不会限制参加梯度下降的 高斯数量, 这使得对于任意场景都不需要调整超参数
  * 对于 backward 过程, 从 loss 计算的梯度要反映到每一个高斯实体上, 这理论上需要 per-pixel full sequence of blended points in the forward pass
    * 作者(在该论文中最新的)解决方法是 复用在渲染时候的 forward pass, 在应用梯度的时候从后向前
    * 应用梯度的时候需要高斯的累计不透明度, 作者在这里只存储累计的不透明度, 然后因为是上述的从后往前, 累计除以当前即可得到在此之前的累计不透明度
    * 高斯的不透明度和渲染像素时候的 alpha 似乎不是同一个东西


### 2.1.2. 4D Gaussian Splatting for Real-Time Dynamic Scene Rendering (CVPR2024)

```latex
 @article{Wu_Yi_Fang_Xie_Zhang_Wei_Liu_Tian_Wang,  
 title={4D Gaussian Splatting for Real-Time Dynamic Scene Rendering}, 
 author={Wu, Guanjun and Yi, Taoran and Fang, Jiemin and Xie, Lingxi and Zhang, Xiaopeng and Wei, Wei and Liu, Wenyu and Tian, Qi and Wang, Xinggang}, 
 language={en-US} 
 }
```

4D 动态场景的建模的主要挑战点是 : 如何从稀疏的输入建模复杂的点的运动
* 简单的思路是为每一帧单独进行 3D-GS 重构, 缺点是超高的存储
  * Dynamic 3D Gaussians: Tracking by Persistent Dynamic View Synthesis
    * 内存消耗 O(tN)
  * REAL-TIME PHOTOREALISTIC DYNAMIC SCENE REP-RESENTATION AND RENDERING WITH 4D GAUSSIAN SPLATTING
    * 直接给 3D-GS 添加一个 时间-空间 的属性来直接升格为 4D-GS, 缺点是容易只关注局部空间的位移

作者提出了一个基于神经网络的 4D GS 表达
* 对于一个动态场景, 只维护 1组 GS 点
* GS 点的学习需要经过 deformation field network, 来完成同时对 空间位置和时间戳的拟合
* multi-resolution encoding 方法用来编码 GS 点的 空间-时间关联


* Gaussian Deformation Field Network $F(G,t)$
  * 用来编码高斯的变形和时间戳的关系
  * 首先可微高斯渲染 $S$ 本身不变, 针对高斯集合 $G$ 和 view matrix $M$, 执行渲染 $I=S(M,G')$
  * 变形后的高斯点 $G'=G+\Delta G$
  * 变形量 $\Delta G = F(G,t) = D( H(G,t) )$
  * 其中 $H,D$ 分别是 (spatial-temporal structure) encoder 和 (multi-head Gaussian deformation) decoder
* Spatial-Temporal Structure Encoder


* 实验数据集
  * 合成数据: D-NeRF
  * Realworld: HyperNeRF,  Neu3D

### 2.1.3. SuGaR: Surface-Aligned Gaussian Splatting for Efficient 3D Mesh Reconstruction and High-Quality Mesh Rendering

CVPR2024
3D-GS 用于 Mesh 生成

https://cvpr.thecvf.com/media/PosterPDFs/CVPR%202024/30910.png?t=1717366268.1976593

主要工作
1. 一个正则化方法， 使得高斯能够更加准确的捕捉几何形状
2. 高速的从高斯点中提取精确的网格的有效算法
3. 高斯绑定到网格， 由于 3D-GS 是可解释的点， 所以该论文就展示了在 Mesh 生成方面 3D-GS 的可编辑性


### 2.1.4. 2D Gaussian Splatting for Geometrically Accurate Radiance Fields


观点: 表面建模本身和 3D GS 不匹配, 因此使用 降维后的 2D GS 来进行该工作

并行工作PK:
* 对3D GS 添加 法线属性
  * 并行工作: 主要用于提高渲染的效果, 即 relightable 性能, 而不是表面建模
  * 本工作  : 2D GS 本身具有法线属性, 因此在表面建模上更加符合
* SuGaR :
  * 使用了 额外的 loss 正规化项目, 使得 3D GS 尽可能扁平
  * 本工作 : 2D GS 本来就是平面
* NeuSG : 
  * NeuS 的后续手法, 结合了 SDF 和 3D GS
  * 本工作 : 从原理上来说更简单, 从实际上来说学习速度更快

原理思路:
* 3DGS 与 薄 thin 的 Surface 是相冲突的
* 3DGS 本身没有 法线属性, 这对于表面重构是超级重要的
* NVS 任务不必要, 但是表面重构是超级重要的, view consistency 在原本的 3DGS 中没有


模型:
* 直接删去 3D 高斯方差的一个维度, 同时将对应的实际属性拆分
  * 中心点 $p_k$ 不变, 为 3维
  * 两个主切向向量 (principal trangential vectors) $t_u, t_v$, 沿着表面切线方向, 同时沿着 GS 椭圆的 长短轴
  * 一个 缩放向量 $S=(s_u, s_v)$, 控制 2D GS 在 UV 平面上的方差
  * 主法线通过 $t_u \times t_v = t_w$ 来定义
    * 同原本的 3D GS 相同的 旋转向量可以以此 获得 $R=[t_u, t_v, t_w]$
    * 对应的 33 scale 矩阵也是, 直接将 $s_u, s_v$ 放在对角线上即可
      * (t_u, t_v 沿着椭圆长短轴)因此 scale 矩阵是对角矩阵, 没有第三行列, 且前两行列没有协方差
  * 2DGS 的不透明度和 球面谐波同原本的 3D GS 相同
* 最终, 完整的 local tangent plane (局部切线平面) 在 world space 的表达为
  * $P(u,v) = p_k + s_ut_uu +s_vt_vv= H(u,v,1,1)^T$
  * H 是一个函数表达, 代表了 4x4 的 homogeneous transformation (齐次变换)
$$H = \begin{bmatrix}
  s_ut_u & s_vt_v& 0 &p_k \\
  0 &0&0&1
\end{bmatrix}=
\begin{bmatrix}
  RS & p_k \\
  0 & 1
\end{bmatrix}
$$


投影
* 直接将 2D GS 按照中心点 affine 投影到 图像平面上, 会导致只有中心点的距离是正确的
* 解决办法是, 将 2D splat 投影到 图像平面的这一过程描述为齐次坐标下的 general 2D-to-2D mapping
  * 从图像平面 (x,y) 坐标出发的 camera ray 射线可 slpat 的焦点的深度为 z, 其空间点的坐标为 
$$
x = (xz, yz, z, 1)^T=WP(u,v) = WH(u,v,1,1)^T
$$

Splatting
* 栅格化 rasterize : explicit ray-splat intersection
  * 能够快速的确定每一个像素点对应的高斯距离
* Degenerate Solutions : 
  * 在渲染的时候剔除与相机射线近乎平行的 2DGS
* rasterization : 与 3DGS 相同

Mesh 建模:
* TSDF 
  * 使用 median depth 效果最好
* Poisson surface reconstruction 也可以用, 但是效果不如上着


### RaDe-GS: Rasterizing Depth in Gaussian Splatting 

2D GS 的后继

对之前工作的评价
* GOF     : 计算量太大
* 2D-GS   : 扁平化的 GS, 直接使用低维度的表达会导致 优化困难, 对于复杂的形状导致重建困难

该工作的成果: 用原本 3D-GS 的速度实现了 NeuraLangelo (0.61)的近似精度 (0.69), 优于 GOF(0.74)  2D GS(0.80)
* novel rasterized method 用来高效计算 depth and normal maps




