# 1. Implicit Neural Representation

通过神经网络参数 来不同于传统方式的 隐式表示一些东西

在 图像方面有很高的成果

# 2. Nural Surface Reconstruction

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


## 2.1. Neural Radiance Fields (NeRF)

通过 神经网络表示 Radiance Fields

2019年开始兴起, 在 2020 年 ECCV 中得到 Best Paper Candidate  

NeRF 是一种隐式的 3D 中间表示, 但是却使用了 Voluem 的规则, 即一个 隐式的 Volume, 实现了 神经场 Neural Field 与图形学组件 Volume Rendering 的有效结合  
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
* 对于一个 camera ray  $r(t)=o+td$  t 是 camera ray 的远近距离 t_n t_f
* camera ray 得到的颜色 C(r)可以写作  
$$C(r)=\int_{t_n}^{t_f}T(t)\sigma(r(t))c(r(t),d)dt$$
* $T(t)=exp(-\int_{t_n}^t\sigma(r(s))ds)$
  * 该公式代表了一个 accumulated transmittance, 
  * 对于一个距离 t 从 t_n 到 t, 光线最终没有被遮蔽的概率  
* 从一个 NeRF 模型中渲染出一个 view 需要
  * estimating this integral C(r) for a camera ray traced through each pixel of the desired virtual camera.
  * 从虚拟摄像头中, 对穿越每一个像素的 camera cay 计算 C(r)


NeRF 本身的问题, 有如下:
* 速度慢  : 对于每个输出像素分别进行前向预测, 因此计算量很大  
* 只能应用于静态场景
* 泛化性差
* 需要大量的视角  : 需要数百张不同视角的图片来训练  

### 2.1.2. Topics

* mip-NeRF 360 consistently produces fewer artifacts and higher reconstruction quality. 
* low-dimensional generative latent optimization (GLO) vectors introduced in NeRF in the Wild, learned real-valued latent vectors that embed appearance information for each image. the model can capture phenomena such as lighting changes without resorting to cloudy geometry, a common artifact in casual NeRF captures. 
* exposure conditioning as introduced in Block-NeRF, 


* NeRF's baked representations


### 2.1.3. Practical Concerns


* 输入数据 : a dense collection of photos from which 3D geometry and color can be derived, every surface should be observed from multiple different directions.
* For example, most of the camera’s properties, such as white balance and aperture, are assumed to be fixed throughout the capture.
* scene itself is assumed to be frozen in time: lighting changes and movement should be avoided. 
* As photos may inadvertently contain sensitive information, we automatically scan and blur personally identifiable content.


### 2.1.4. Rest questions


* with scene segmentation, adding semantic information to the scenes
* Adapting NeRF to outdoor photo collections
* Real time render

### 2.1.5. Referance


Google Blog
Reconstructing indoor spaces with NeRF
Wednesday, June 14, 2023
https://ai.googleblog.com/2023/06/reconstructing-indoor-spaces-with-nerf.html


# 3. 2D - Implicit Image Function

通过 神经网络表达 2D 图像
