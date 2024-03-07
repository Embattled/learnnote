# 1. Image-based 3D reconstruction

基于图像的3D重构

和 SLAM 技术有些重叠, 或者说某些 3D 重构会用到 SLAM 技术  

# 2. View Synthesis

视角合成任务, 通过输入对一系列对物体不同角度的图像, 来生成新的角度下的图像
* 对于一个训练好的模型
* 通常输入的数据是一系列图像, 并且带有对应的角度数据, e.g. 空间坐标 (x,y,z) 视角 (theta, phi)
* 输出是
  * the volume density 
  * and view-dependent emitted radiance at that spatial location, 可以直接理解成新视角下的图像



通常的手法使用一个中间的 3D 场景表征来作为中介, 并以此生成高质量的虚拟视角, 根据该中间表征的形式, 可分为:
* 显式 Explicit representation : 例如 Mesh, Point Cloud, Voxel, Volume 等等, 对场景进行显式建模, 但是这些显式类型一般都是离散的, 有精度问题
* 隐式 Implicit representation : 用一个函数来描述几何场景, 一般是一个不可解释的 MLP 模型, 输入 3D 空间坐标, 输出对应的几何信息, 是一种连续的表示 (Neural Fields, 神经场)   



## 2.1. 3D shape/suface representations

在 NeRF 提出之前的主流方案, 对于一个连续的 3D shape
* map xyz coordinates to signed distance functions or occupancy fields
* 这种方案最早的时候需要 GT 3D geomerty, 因此在研究中经常使用 synthetic 3d shape
* 后来有直接输出每个坐标对应的 feature vector 和 RGB function, 在通过复杂的 rendering function 得到2D img 再计算 Loss

