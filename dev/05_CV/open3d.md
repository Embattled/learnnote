# Open3D: A Modern Library for 3D Data Processing

用于 高速开发处理 3D Data 的应用

用 C++ 和 Python 实现, 并仔细的进行了并行化

Core features of Open3D include:
* 3D data structures
* 3D data processing algorithms
* Scene reconstruction
* Surface alignment
* 3D visualization
* Physically based rendering (PBR)
* 3D machine learning support with PyTorch and TensorFlow
* GPU acceleration for core 3D operations
* Available in C++ and Python



# Tutorial 教程 (非 API)

## Core - 核心数据结构 


## Geometry - 几何体
### Point Cloud - 点云

Open3D 提供了相当多直接操作点云的函数, 甚至包括可视化



函数接口:
* 点云读取: Visualize Point Cloud
  * `read_point_cloud `       : 从支持的文件类型读取点云, 根据文件后缀名进行尝试解码
* 点云可视化:
  * `draw_geometries `        : 可视化对应的 点云结构体, 可视化窗口拥有交互
  * 在可视化的时候会把点云渲染为 面元 surfels, 可以使用键盘来更改 面元的大小
  * 按 `H` 会打印完整的键盘指令列表
* 体素降采样 : Voxel downsampling , 常常作为 点云的预处理
  * Points 会先 bucketed into voxels, 对于每一个 voxels 综合内部所有的 points 生成单个 points
  * `voxel_down_sample`
* 法线向量估计:
  * `estimate_normals`
  * 对于点云 (一般是稀疏或者降采样后的) 计算每一个点的法线, 该函数通过查找相邻的点, 并使用协方差来分析相邻点的主轴
  * 通过协方差分析出来的法线是无法得知 内外方向的, Open3D 的策略是与预先提供的法线尽可能一致, 否则就随机猜
    * 如果需要调整的话使用 `orient_normals_to_align_with_direction  orient_normals_towards_camera_location `
* 点云裁剪  : 获取对应多面体中的点云
* 绘制      : 手动为点云赋予颜色


### RGBD images

在 Open3D 中是非常常用的数据格式

`RGBDImage` 包括 2个独立的图, 分别是 `.depth` 和 `.color`, 两个图需要绑定相同的 camera frame 并且需要有相同的 resolution

Open3D 文档中提供了几个经典的数据集

### KDTree - K-Dimension Tree

K 维树, 用于在高位数据上实现快速最近邻检索  

在点云上建立 KDTree 用于后续的应用  


```py
pcd_tree = o3d.geometry.KDTreeFlann(pcd)

# 最近 200 邻查找, KNN
[k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[1500], 200)

# 半径小于阈值查找, RNN
[k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[1500], 0.2)

# 除此之外 Open3D 还实现了二者结合的 KRNN 方法
```

### Surface reconstruction

很多时候, 需要一个 dense 3D geometry, 例如 triangle mesh, 然而通过
* Multi-view stereo
* depth sensor
等方法只能获得稀疏的 unstructured point cloud 

Open3D 的库提供了多种重建方法
* Alpha shapes `[Edelsbrunner1983]`
  * `create_from_point_cloud_alpha_shape` 
* Ball pivoting `[Bernardini1999]`
* Poisson surface reconstruction `[Kazhdan2006]`



**Alpha shapes** 
* 是 凸包 convex hull 的推广
* 可以理解为
  * 冰激凌包裹着硬巧克力块
  * 使用球型冰激凌勺子, 在不碰到巧克力的情况下雕刻出冰激凌块的所有部分, 甚至在内部挖孔
  * 最终得到一个 (不一定凸的) 由 caps, arcs, points 构成的对象
  * 如果将所有圆面拉直为三角形和线段, 就可以得到一个形状的描述 -> alpha shape 

**Ball pivoting**
* 同 alpha shape 类似, 也是一个启发式算法
* 可以理解为
  * 想象一个具有给定半径的 3D 球, 落在 3D 点云上, 如果它击中任意 3 个点同时没有落入这三个点, 则通过这三个点创建三角形
  * 此后, 从现有三角形的边缘进行旋转, 每次碰到球不会落下的 3 个点时 都会创建三角形
* Open3D 的实现 `create_from_point_cloud_ball_pivoting`
  * 提供一个 list of radial 用于定义 重建 3D shape 的时候依次应用的球的半径


**Poisson surface reconstruction**
* 解决了 regularized optimization problem, 获得了平滑的表面, 比上述两个方法更加可取 (上述方法会产生非平滑结果)
* Poisson Surface Reconstrution 算法假设 PointCloud 拥有法线
* `create_from_point_cloud_poisson`
* 算法需要名为 `depth ` 的参数用于 表面重构的 八叉树的深度 (OCTree), 意味着 Mesh 的分辨率, 深度值越高代表网格的细节越好
* 还可以根据 mesh 的密度来进行低密度过滤


## Reconstruction system - Reconstruct a 3D scene from RGBD

The pipeline is based on `[Choi2015]`. 
Several ideas introduced in `[Park2017]` are adopted for better reconstruction results.


## Reconstruction system (Tensor)

两个目的
* volumetric RGB-D reconstruction
* online dense RGB-D SLAM

使用 Open3D 的 Tensor 和 Hash map 后端来实现  

该部分的基础需要参照论文工作
```latex
@article{Dong2021,
    author    = {Wei Dong, Yixing Lao, Michael Kaess, and Vladlen Koltun}
    title     = {{ASH}: A Modern Framework for Parallel Spatial Hashing in {3D} Perception},
    journal   = {arXiv:2110.00511},
    year      = {2021},
}
```

该章节的文档只描述了 
* online dense SLAM 
* offline integration with provided poses.

论文中的其他工作说明仍在编写(?), 可以参照上一章节, 即 legacy versions
* tensor-based offline reconstruction system
* Simultaneous localization and calibration (SLAC)
* shape from shading (SfS)


### Voxel Block Grid - 体素块网格


体素块网格 是一个 全局稀疏, 局部密集 的数据格式, 用于表达 3D scene
* globally spare    : Surfaces 通常是一个薄面, 因此仅仅占据空间中的很小一部分
* locally dense     : 表达 连续的 Surfaces 在某种层面上是一个密集表达

为了表达 surface, VBG 方法:
* 将 3D Space 分割为 block grids
* 对于 包含了 surface 的 blocks, 通过 3D 坐标 将它们组织在 hash map 中
  * 进一步可以生成可以通过数组索引访问的 dense voxels (dense locally)
  * 不直接通过 voxels 来生成 hash map 是有原因的:
  * 为了保证数据的局部集中性, 以防止相邻的数据分散到内存中

体素块网络是 3D 表示的一种, 也是该库中表面重构的基础  

```py
# colored
vbg = o3d.t.geometry.VoxelBlockGrid(
  attr_names=('tsdf', 'weight', 'color'),
  attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
  attr_channels=((1), (1), (3)),
  voxel_size=3.0 / 512,
  block_resolution=16,
  block_count=50000,
  device=device)

# gray
vbg = o3d.t.geometry.VoxelBlockGrid(
  attr_names=('tsdf', 'weight'),
  attr_dtypes=(o3c.float32,o3c.float32),
  attr_channels=((1), (1)),
  voxel_size=3.0 / 512,
  block_resolution=16,
  block_count=50000,
  device=device)
```

默认以 shape (3,) 的元素作为 hash 的 key (即 3D 坐标)
* attr 数据类型统一为 float32
* block_count = 50000 是作者对于 客厅大小的 scene 的经验之谈 (足够避免块的 rehashing)
* block_resolution : 决定单个 block 内部所具有的解析度, 在此处会导致:
  * truncated Signed Distance Function (TSDF) of element shape (16, 16, 16, 1)
  * Weight of element shape (16, 16, 16, 1)
  * RGB color of element shape (16, 16, 16, 3)
* `voxel_size=3.0 / 512` 决定了单个 voxel 的空间解析度
  * 此处的 3 代表米
  * 3 / 512 表示如果要表达完整的 3x3x3 空间, 会生成 512x512x512 的 voxel grid


### TSDF Integration - Truncated Signed Distance Function (TSDF) integration

是 dense volumetric scene reconstruction 的关键技术, 从相对 noisy 的 RGB-D Sensor 直出图 (例如 Kinect, RealScnse), 重建出 降低噪声并且相对平滑 的 surfaces


Activation : 激活阶段
* 从 camera 视锥体(frustum) 中查找当前 depth image 的活动块, 即包含对应空间点的块
* 对每一帧进行独立查找
* 在库内部, 通过无重复的 hash 查找来实现

Integration : 聚合
* 将所有的 voxels 投影到输入图像上, 以此来计算权重 (执行 weighted average)
* Open3D 在该步骤提供了优化函数


Surface extraction : 提取 Mesh
* extract_triangle_mesh applies marching cubes and generates mesh
* extract_point_cloud uses a similar algorithm, but skips the triangle face generation step.


实例代码
```py

# 获取激活的 blcoks
frustum_block_coords = vbg.compute_unique_block_coordinates(
    depth, depth_intrinsic, extrinsic, config.depth_scale,
    config.depth_max)

# 带颜色的 integrate
if config.integrate_color:
    color = o3d.t.io.read_image(color_file_names[i]).to(device)
    vbg.integrate(frustum_block_coords, depth, color, depth_intrinsic,
                  color_intrinsic, extrinsic, config.depth_scale,
                  config.depth_max)
else:
    vbg.integrate(frustum_block_coords, depth, depth_intrinsic,
                  extrinsic, config.depth_scale, config.depth_max)


# 直接通过 vbg 的类方法获取对应的 点云或者 Mesh
pcd = vbg.extract_point_cloud()
o3d.visualization.draw([pcd])

mesh = vbg.extract_triangle_mesh()
o3d.visualization.draw([mesh.to_legacy()])
```