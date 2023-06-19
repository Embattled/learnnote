# 1. introduction - a Tour of Multiple View Geometry

给予本书的一些主要思想, 以及一些基本概念 


## 射影几何无处不在 - ubiquitous projective geometry

平面上的物体映射到图上即 射影变换的一个例子, 射影变换的特性有:
* 不保留形状
* 不保留长度
* 保留了 直线度 straightness : most general requirement on the mapping
* 平面上的射影变换为 平面上保持直线的任何点映射

欧氏几何 Euclidean Geometry : 是描述物体的角度和形状的几何
* 不足之处就是总是存在例外 : 
  * 2维中两条直线总是相交于一个点, 除了平行线, 或者可以说 平行线相交于无穷远
  * 无穷远只是一个虚构, 是一个理想点
* 通过向欧氏几何中添加一个无穷远, 可以拓展欧氏几何空间, 变成了 射影空间 Projective Space
  * 射影空间是一种很有用的思维方式
  * 和欧氏空间相比仍然保留了熟悉的概念, 距离, 点, 角度, 直线等
  * 唯一的扩展是 : 两条直线总是相交

坐标 Coordinates:
* 以 2 维举例, 在欧氏空间中, 一个点可以表示为有序的实数对 (x,y)
* 给该对加上一个额外的坐标 (x,y,1), 同时声称它们指代的是同一个点, 这样通过添加和删除最后一个坐标即可实现两空间表示的转换
* 此时 最后一个坐标 `1` 的意义是: 
  * 定义 : (kx, ky, k) 和 (x, y, 1) 指代的是同一个点, 或者更严谨的说 被表示为坐标三元组的 `等价类`
  * 即, 两个三元组相差一个公共倍数的时候, 它们是等价的
  * 此时, 三元组即为 点的 `齐次坐标 hemogeneous coordinates` : 通过除以 k ,即可获得原始坐标
  * 不存在二维上的点能够对应 (x, y, 0) , 因为 x/0 是无穷大的, 即通过这种方法来表示了无穷远的概念, 即齐次坐标中最后一个坐标值是 0 的点
* 把欧式空间中的点表示为 齐次向量 (hemogeneous vector) 即可实现把欧氏空间拓展到射影空间, 这对于任何维度都是有效的
  * 从欧氏空间 $IR^n$ 拓展到 射影空间 $IP^n$
  * 在2维射影空间中的无穷远点形成了一条 无穷远直线 line at infinity, 同样的, 在3维射影空间中成为无穷远平面 plane at infinity


齐次性 Homogeneity:
* 需要理解 欧氏空间中所有的点都是相同的, 不存在特殊的点
* 当用坐标来表示欧式空间时, 有一个点被选作为原点, 然而这完全是随机的
* 通过平移的旋转轴线, 空间中的任意一个点都可以作为原点, 这可以理解为欧式空间本身的位移和旋转, 这样的操作则成为 Euclidean Transform 欧式变换

仿射变换 Affine Transformation:
* 在应用一个 线性变换 linear transformation 后接着一个 欧式变换
* 这可以理解为整个欧氏空间的 移动, 旋转, 和线性拉伸 stretching 
* 仿射变换算是一个小拓展, 无穷远仍然保持在无穷远, 而空间内容上的点仍然以某种方式被保存

射影变换 projective transformation : 通过类比 欧式变换和仿射变换, 可以定义
* 射影空间 中的射影变换是一个 `表示点的齐次坐标 (n+1)维向量的映射`, 即本质上是一个 映射 mapping
* 射影变换中, 齐次坐标 被乘以一个 非奇异矩阵 non-singular matrix , 即齐次坐标的线性变换 
* 此时, 无穷远点不被保持
* $X'=H_{(n+1)\times (n+1)}X$

### Affine Geometry 仿射几何




## Camera Calibration 相机标定

也被叫做 camera resectioning

pinhole camera 是一种最简单的相机, 没有 lens, 只有一个光圈 aperture.  
光线通过光圈然后将图像的颠倒投影在相机里.  

### Distortion in Camera Calibration

在整个 Camera Calibration 算法中都不会考虑镜头畸变 (lens  distortion), 即相机是作为 pinhole 相机进行处理的, 而 pinhole 相机没有 Lens

需要考虑的畸变有 径向 radial 和 切向 tangential 畸变 distortion
* $x,y$ 指代未经过畸变的 pixel locations, 这里的 $x,y$ 都是在 normalized image coordinates 下的, 即转换到光学中心, 并且除以以像素为单位的焦距的.
* $r^2 = x^2+y^2$ 半径的平方?, 可以理解为点的到光学中心的距离平方  


Radial distortion : 镜头越小, 畸变越大. 光线在镜头边缘附近比其在光学中心弯曲得多
* radial distortion coefficients model radial distortion
* distorted points are denoted as $(x_{distorted}, y_{distorted})$
  * Normalized image coordinates are calculated from pixel coordinates by translating to the optical center and dividing by the focal length in pixels. 
  * Thus, x and y are dimensionless. 
* $k_1,k_2,k_3$ 具体的 radial distortion coefficients 
* $x_{distorted} = x(1+k_1r^2+k2r^4+k_3r^6)$
* $y_{distorted} = y(1+k_1r^2+k2r^4+k_3r^6)$
* 通常情况下, 两个系数 k1,k2 足够对径向畸变进行矫正, 对于严重畸变例如广角镜头, 则可以加入 k3


Tangential Distortion : 当镜头和成像平面不平行的时候会发生切向畸变
* $p_1, p_2$ , 切向畸变系数
* $x_{distorted} = x + [2p_1xy+p_2(r^2+2x^2)]$
* $y_{distorted} = y + [p_1(r^2+2y^2)+2p_2xy]$


### camera matrix
可以想象为:
* 真实的3-D 物体投影到一个 Virtual Image Plane 上, 然后颠倒后映射在相机传感器里
* Virtual image plane 到光圈和 传感器到光圈的距离相同, 都被称为  Focal length


pinhole 相机的参数可以通过一个 3-by-4 矩阵来表示, 也被称为相机矩阵 camera matrix, 通过该矩阵即可实现3-D 世界中的场景到 image plane 的转换, calibration 算法即为: 通过外部和内部的参数来计算 camera matrix  
* extrinsic parameters : location of the camera in the 3-D scene
* intrinsic parameters : optical center and focal length of the camera
* 在计算的含义上
  * 通过 extrinsic parameters 即可把 world points 转换为 camera coordinates
  * 通过 intrinsic parameters 把 camera coordinates map to image plane  
  * World coordinates $[X Y Z]$ --(Rigid 3-D to 3-D Extrinsic parameters)--> Camera Coordinates $X_c Y_c Z_c$ --(Projective 3-D to 2-D intrinsic parameters)--> Pixel coordinates $[x y]$
* 数学表达
  * P : camera matrix
  * K : intrinsics matrix
  * t : Extrinsics translation
  * R : Extrinsics Rotation
  * $P = K[Rt]$



Intrinsic Parameters 具体为:
$$
\begin{bmatrix}
f_x & s & c_x \\
0 & f_y & c_y \\
0 & 0 & 1 
\end{bmatrix}
$$
* $[c_x c_y]$ , 相机的光学中心, Optical center (the principal point), in pixels
* $[p_x, p_y]$ , Size of the pixel in world units, 真实世界下的每像素大小
* $F$ 相机的真实焦距, Focal length in world units, 通常会以 millimeters 单位表示
* $(f_x, f_y)$ 相机的 Focal length 焦距 $f_x=F/p_x, f_y=F/p_y$
* $s$ skew coefficient, 偏斜系数, 即如果图像的坐标轴垂直的话, 则该系数为 0 
  * $s = f_x \tan\alpha$ alpha 是坐标轴的偏斜角度
* 



