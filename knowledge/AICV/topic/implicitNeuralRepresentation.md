# 1. Implicit Neural Representation

通过神经网络参数 来不同于传统方式的 隐式表示一些东西

在 图像方面有很高的成果

# 2. Nural 3D Representation 

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



## 2.1. Neural Radiance Fields (NeRF) 基于(MLP)神经网络的 Radiance Fields

通过 神经网络表示 Radiance Fields

2019年开始兴起, 在 2020 年 ECCV 中得到 Best Paper Candidate  

NeRF 是一种隐式的 3D 中间表示, 但是却使用了 Volume 的规则, 即一个 隐式的 Volume, 实现了 神经场 Neural Field 与图形学组件 Volume Rendering 的有效结合  
* 本身的方法非常简洁, 且有效, 说明是合理的
* 对于启发 计算机视觉和图形学的交叉领域 有很大的功劳


Neural Fields  神经场:
* 场 Fields   : 是一个物理概念, 对所有 (连续)时间 或 空间 定义的量, 如电磁场, 重力场, 对 场的讨论一定是建立在目标是连续概念的前提上
* 神经场表示用神经网络来 全部或者部分参数化的场
* 在视觉领域, 场即空间, 视觉任务的神经场即 以 `空间或者其他维度 时间, 相机角度等` 作为输入, 通过一个神经网络, 获取目标的一个标量 (颜色, 深度 等) 的过程   


### 2.1.1. Vanilla NeRF

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


### 2.1.2. NeRF Conclusion 


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


#### 2.1.2.1. Topics


* 动态场景, Time-Space 的 NeRF 学习, 使之能够表示一段连续时间下的动态场景
* 预训练 MLP, 减少实际学习中的图片数量需求和学习时间
* 泛化 NeRF, 减少对学习数据的相机位姿依赖 (低精度位姿, 无位姿)

* mip-NeRF 360 consistently produces fewer artifacts and higher reconstruction quality. 
* low-dimensional generative latent optimization (GLO) vectors introduced in NeRF in the Wild, learned real-valued latent vectors that embed appearance information for each image. the model can capture phenomena such as lighting changes without resorting to cloudy geometry, a common artifact in casual NeRF captures. 
* exposure conditioning as introduced in Block-NeRF, 


* NeRF's baked representations


#### 2.1.2.2. Practical Concerns


* 输入数据 : a dense collection of photos from which 3D geometry and color can be derived, every surface should be observed from multiple different directions.
* For example, most of the camera’s properties, such as white balance and aperture, are assumed to be fixed throughout the capture.
* scene itself is assumed to be frozen in time: lighting changes and movement should be avoided. 
* As photos may inadvertently contain sensitive information, we automatically scan and blur personally identifiable content.






#### 2.1.2.3. Referance


Google Blog
Reconstructing indoor spaces with NeRF
Wednesday, June 14, 2023
https://ai.googleblog.com/2023/06/reconstructing-indoor-spaces-with-nerf.html


Mildenhall, Ben, et al. “NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis.” Computer Vision – ECCV 2020,Lecture Notes in Computer Science, 2020, pp. 405–21, https://doi.org/10.1007/978-3-030-58452-8_24.

## 2.2. Point-Based Radiance Fields

基于 points 的 volumetric representation 的方法 (传统方法)
* 极度不连续的缺点
  * 容易导致 over or under reconstruction
* 基于 CNN 的算法, 求解空间中的 points 的特征之类的

待看

Differentiable Point-Based Radiance Fields for Efficient View Synthesis

Point‐Based Neural Rendering with Per‐View Optimization


## 2.3. Neural Surface Reconstruction

基于 NeRF 的思想, 修改隐含表达的公式, 实现更容易对 3D 场景进行表面建模 (Surface Reconstruction)


### 2.3.1. NeuS 

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

## 2.4. occupancy grids

在 2020 年之前比较多  



# 3. 2D - Implicit Image Function

通过 神经网络表达 2D 图像

