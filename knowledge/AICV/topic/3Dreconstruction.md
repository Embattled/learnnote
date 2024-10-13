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

原本属于计算机图形科学的概念

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


### 2.1.5. RaDe-GS: Rasterizing Depth in Gaussian Splatting 

2D GS 的后继

对之前工作的评价
* GOF     : 计算量太大
* 2D-GS   : 扁平化的 GS, 直接使用低维度的表达会导致 优化困难, 对于复杂的形状导致重建困难

该工作的成果: 用原本 3D-GS 的速度实现了 NeuraLangelo (0.61)的近似精度 (0.69), 优于 GOF(0.74)  2D GS(0.80)
* novel rasterized method 用来高效计算 depth and normal maps


## 2.2. Neural Radiance Fields (NeRF) 基于(MLP)神经网络的 Radiance Fields


通过 coordinate-based multi-layer perceptrons (MLPs) 来将一个场景表示为一个隐式的函数  

根据所谓 函数的描述对象可以对手法进行分类   
即中间层的显式表达方法 Mesh, Point Cloud, Voxel, Volume 等

* occupancy fields
  * UNISURF: Unifying Neural Implicit Surfaces and Radiance Fields for Multi-View Reconstruction
* Signed Distance Functions (SDF)
  * Volume Rendering of Neural Implicit Surfaces
* Neural Radiance Fiedls


Volume 体数据  体渲染

从体数据渲染得到想要的 2D 图片  

体数据是一种数据存储格式, 例如 医疗中的 CT 和 MRI, 地址信息, 气象信息
* 是通过 : 追踪光线进入场景并对光线长度进行某种积分来生成图像或者视频   Ray Casting Ray Marching Ray Tracing
* 这种数据需要额外的渲染过程才能显示成 2D 图像并被人类理解  
* 对比于传统的 Mesh, Point 等方法, 更加适合模拟光照, 烟雾, 火焰等非刚体, 在图形学中也有应用   
* 体渲染是一种可微渲染  


Neural 3D shape representations
* 早期的基于深度学习的: 需要 3D model 的 GT
  * SDF-base : Local Implicit Grid Representations for 3D Scenes
  * occupancy field base: Local Deep Implicit Functions for 3D Shape
* 改进了渲染方法: 2D 图像即可学习
  * Differentiable volumetric rendering: Learning implicit 3D representations without 3D supervision.
  * Scene representation networks: Continuous 3D-structure-aware neural scene representations.




通过 神经网络表示 Radiance Fields

2019年开始兴起, 在 2020 年 ECCV 中得到 Best Paper Candidate  

NeRF 是一种隐式的 3D 中间表示, 但是却使用了 Volume 的规则, 即一个 隐式的 Volume, 实现了 神经场 Neural Field 与图形学组件 Volume Rendering 的有效结合  
* 本身的方法非常简洁, 且有效, 说明是合理的
* 对于启发 计算机视觉和图形学的交叉领域 有很大的功劳


Neural Fields  神经场:
* 场 Fields   : 是一个物理概念, 对所有 (连续)时间 或 空间 定义的量, 如电磁场, 重力场, 对 场的讨论一定是建立在目标是连续概念的前提上
* 神经场表示用神经网络来 全部或者部分参数化的场
* 在视觉领域, 场即空间, 视觉任务的神经场即 以 `空间或者其他维度 时间, 相机角度等` 作为输入, 通过一个神经网络, 获取目标的一个标量 (颜色, 深度 等) 的过程   


### 2.2.1. Vanilla NeRF

将一个 scene 表示成一个 5D vector-valued function.
* 输入 3D location X=(x,y,z) 和 viewing direction d=(theta, phi)
* 输出 emitted color $c=(r,g,b)$ 和 volume density $\sigma$
* volume density sigma(x) 可以解释为一个 ray 在空间中的无限微小点 X 终止的微分概率  


基于 NeRF 的 Volume Rendering
* 对于一个 camera ray  $r(t)=o+td$
* $t$ 是 camera ray 的远近距离 $t_n,t_f$
* camera ray 得到的颜色 C(r)可以写作  
$$C(r)=\int_{t_n}^{t_f}T(t)\sigma(r(t))c(r(t),d)dt$$
* $T(t)=exp(-\int_{t_n}^t\sigma(r(s))ds)$
  * 该公式代表了一个 accumulated transmittance, 
  * 对于一个距离 t 从 t_n 到 t, 光线最终没有被遮蔽的概率  
* 从一个 NeRF 模型中渲染出一个 view 需要
  * estimating this integral C(r) for a camera ray traced through each pixel of the desired virtual camera.
  * 从虚拟摄像头中, 对穿越每一个像素的 camera cay 计算 C(r)

NeRF 的新颖点:
* 将3D场景存储在神经网络中, 对比传统的存储 离散的 voxel grids, 节省了存储空间
* 体渲染用于场景重构, 同时相机的方向也作为网络输入, 提高了准确度
* 两个优化方法进一步一高精度
  * 高频位置编码
  * 分层空间位置采样


NeRF 本身的问题, 有如下:
* 速度慢  : 对于每个输出像素分别进行前向预测, 因此计算量很大  
* 只能应用于静态场景
* 泛化性差
* 需要大量的视角  : 需要数百张不同视角的图片来训练  

NeRF的训练:
* batch_size : 4096
* Nc=64, Nf=128
* Adam : 5e-4 -> 5e-5
* 100k~300k 收敛
* 1~2 day on V100


### 2.2.2. NeRF Conclusion 


优化训练和渲染的 3个研究方向
* 改进空间表达的存储方法: 连续的 MLP -> 离散的空间表示 -> 在渲染的时候直接插值特征
  * 比较有名的就是 instant-ngp 的 hash , 以及 Plenoxels 的 sparse voxel grid
  * FastNeRF 之类的
* different encodings
* MLP capacity



既存问题
* with scene segmentation, adding semantic information to the scenes
* Adapting NeRF to outdoor photo collections
* Real time render, 每一个光线 (对应一个像素点) 都需要采样一系列的空间点, 对于高分辨率图像的渲染简直恐怖
* 缺少对 empty space 的根源性 高效适应能力 (NeRF 通过学习到对应区域的不透明度为0 来实现)


#### 2.2.2.1. Topics


* 动态场景, Time-Space 的 NeRF 学习, 使之能够表示一段连续时间下的动态场景
* 预训练 MLP, 减少实际学习中的图片数量需求和学习时间
* 泛化 NeRF, 减少对学习数据的相机位姿依赖 (低精度位姿, 无位姿)

* mip-NeRF 360 consistently produces fewer artifacts and higher reconstruction quality. 
* low-dimensional generative latent optimization (GLO) vectors introduced in NeRF in the Wild, learned real-valued latent vectors that embed appearance information for each image. the model can capture phenomena such as lighting changes without resorting to cloudy geometry, a common artifact in casual NeRF captures. 
* exposure conditioning as introduced in Block-NeRF, 


* NeRF's baked representations


#### 2.2.2.2. Practical Concerns


* 输入数据 : a dense collection of photos from which 3D geometry and color can be derived, every surface should be observed from multiple different directions.
* For example, most of the camera’s properties, such as white balance and aperture, are assumed to be fixed throughout the capture.
* scene itself is assumed to be frozen in time: lighting changes and movement should be avoided. 
* As photos may inadvertently contain sensitive information, we automatically scan and blur personally identifiable content.




#### 2.2.2.3. Referance


Google Blog
Reconstructing indoor spaces with NeRF
Wednesday, June 14, 2023
https://ai.googleblog.com/2023/06/reconstructing-indoor-spaces-with-nerf.html


Mildenhall, Ben, et al. “NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis.” Computer Vision – ECCV 2020,Lecture Notes in Computer Science, 2020, pp. 405–21, https://doi.org/10.1007/978-3-030-58452-8_24.

## 2.3. Point-Based Radiance Fields

基于 points 的 volumetric representation 的方法 (传统方法)
* 极度不连续的缺点
  * 容易导致 over or under reconstruction
* 基于 CNN 的算法, 求解空间中的 points 的特征之类的

待看

Differentiable Point-Based Radiance Fields for Efficient View Synthesis

Point‐Based Neural Rendering with Per‐View Optimization


## 2.4. Neural Surface Reconstruction

基于 NeRF 的思想, 修改隐含表达的公式, 实现更容易对 3D 场景进行表面建模 (Surface Reconstruction)


### 2.4.1. NeuS 

该方向的开山作 : SDF 表达 (?), 基于 Radiance Fields 的思想进行魔改.
* 好像 SDF 的提出并不是 NeuS?  
  * 2020: Multiview Neural Surface Reconstruction by Disentangling Geometry and Appearance
  * NeuS 的主要理论工作是提出了基于 SDF 的 Render 方法, 使得 NeRF 的学习 pipeline 可以直接应用到 Surface Reconstruction 上
* 隐式表达的 MLP
  * 输入空间坐标 x, 输出到对应物体表面的 (最近?) 距离 distance
  * 输入 空间坐标 x 和 观看方向 v, 输出观察颜色
* surface 建模
  * 读取 MLP 网络内所有的 f(x) = 0 即可
* 基于 NeRF 学习方法
  * 定义了 probability density function $\phi_s(f(x))$ , 简称  S-density
  * $\phi(x)=se^{-sx}/1+e^{-sx}$ 的定义其实是 sigmoid 函数 $\Phi_s(x)=(1+e^{-sx})^-1$的导数
  * 从概念上, $\phi_s(x)$ 可以定义为任意 以0 为中心的单峰密度分布 (bell-shape)
    * bell-shape 钟形函数, 连续的数学曲线, 在 0 处取得最大值, 沿y轴对称
    * bell shape 函数的积分通常是 S 形函数
  * 作者在这里使用 该函数因为计算方便
  * $\phi_s(x)$ 可以理解为 logistic density distribution, 该分布的标准差为 1/s, s 也是一个可以学习的参数, 1/s 越接近于0, 即 s 越大, 代表训练越接近拟合
    * 该函数值代表了此处空间接近附近表面的程度
* 渲染方法 用以生成 RGB 和训练数据做 loss
  * 同 NeRF 类似, 也是定义从相机中心 emit 的 ray, 采样路径上的点, 获得最终图像的颜色
    * $C(o,v)=\int_0^{+\infin}w(t)c(p(t),v)dt$
  * 不同于体渲染, 基于SDF的渲染是本论文新提出来的, 因此渲染的时候采样点的权重计算方法也进行了一些讨论. 为了满足可学习性, 权重的计算函数应当满足
    * 无偏性: 在与表面相交的空间点, 理应取得最大的权重
    * 遮挡 aware : 如果ray方向的空间射线经过多次表面, 则理应返回最近的表面的颜色, 即最近的点取得尽可能大的权重
  * 推导过程
    * 使用 S-density 作为 NeRF 的体密度, 采用相同的积分方法计算权重 剩余光线量乘以当前位置的密度 -> occlusion-aware , 但是有偏差
    * 使用标准化的 S-density 直接作为权重, 分母为 0到无穷的S-density积分, 分子为当前位置的 S-density -> 无偏, 但是无法 occlusion-aware
  * NeuS方法 : 定义不透明度 opaque 函数 $\rho(t)$
    * 通过推导的出来, 首先将距离函数添加了角度项
      * $f(p(t))=-|cos(\theta)|(t-t^*)$
    * 通过推导, 得出 opaque 函数 如下可以同时满足两个要求
      * $\rho(t)=\frac{-\frac{d\Phi_s}{dt}(f(p(t)))}{\Phi_s(f(p(t)))}$
    * 此时权重计算的方法
      * $T(t)\rho(t)=|cos(\theta)|\phi_s(f(p(t)))$

## 2.5. occupancy grids

在 2020 年之前比较多  




# 3. Direct RGB-to-3D 

模型本身学习了 Strong 3D priors, 例如单目深度预测

此外, 建立一个可微分的 SfM Pipeline 并 E2E 的学习

# Pointmap Regression

通过向预训练完成的模型输入一系列的 无 Pose 无相机参数的图像, 直接推理出每一个点的空间坐标, 而其他的常见SfM相关的后继任务可以作为点图的后处理来实现

## 3R 系列


### DUSt3R: Geometric 3D Vision Made Easy

Dense and Unconstrained Stereo 3D Reconstruction (DUSt3R) 


密集 2D 输入作为 3D 点图, 2D 图像上的一个像素对应空间中的一点 $X\in \mathbb{R}^{W\times H\times 3}$

假定每一个像素的 camera ray 只会最终击中单个 3D point, 忽视所有半透明的表面

那么假设获取了某一个图像的 GT 深度图 D, 则可以根据 深度图和相机内参获取该 2D 图对应的 pointmap

$D\in \mathbb{R}^{W\times H}, X_{i,j}= K^{-1}[iD_{i,j},jD_{i,j},D_{i,j}]^T$

模型说明：
1. 整个模型由一个 Encoder 和两个 Decoder 序列构成
2. 输入数据为一个 image pair, 经过相同的 Encoder 获取特征量
3. 由于两个 Decoder 是由多个 block 构成的, 这里两个 Decoder 在最开始和每个 block 后面都会进行信息交换
4. 最后的两个 Head 会分别解码出两个 image 各自的 pointmap 和 confidence map, 其中 point map 只包含相对信息, 不包含针对特定相机模型的绝对坐标


伴随 confidence 的训练:
* 提出于  Confnet: Predict with Confidence
* 模型输出的 Confidence 的原始值 $\tilde{C}$,  应用 $C= 1+ \exp{\tilde{C}}$ 确保信用度为正且大于1
* 添加信用度的损失函数 $L = Cl - \alpha log{C}$ 迫使网络优先去拟合高难度区域



