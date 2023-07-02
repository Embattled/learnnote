- [1. introduction - a Tour of Multiple View Geometry](#1-introduction---a-tour-of-multiple-view-geometry)
  - [1.1. 射影几何无处不在 - ubiquitous projective geometry](#11-射影几何无处不在---ubiquitous-projective-geometry)
    - [1.1.1. Affine Geometry 仿射和欧式几何](#111-affine-geometry-仿射和欧式几何)
  - [1.2. 摄像机投影 Camera Projections](#12-摄像机投影-camera-projections)
- [2. Projective Geometry and Transformations of 2D - 2D 下的射影几何与变化](#2-projective-geometry-and-transformations-of-2d---2d-下的射影几何与变化)
  - [2.1. Planar geometry - 平面几何](#21-planar-geometry---平面几何)
- [3. Projective Geometry and Transformations of 3D - 3D 下的射影几何与变化](#3-projective-geometry-and-transformations-of-3d---3d-下的射影几何与变化)
- [4. Estimation - 2D Projective Transformations - 估计2D射影变换](#4-estimation---2d-projective-transformations---估计2d射影变换)
- [5. Algorithm Evaluation - 算法评价和误差分析](#5-algorithm-evaluation---算法评价和误差分析)
- [6. Camera Models - 摄像机模型](#6-camera-models---摄像机模型)
  - [6.1. Finite cameras - 有限摄像机](#61-finite-cameras---有限摄像机)
- [\\end{pmatrix}](#endpmatrix)
  - [The projective camera - 射影摄像机](#the-projective-camera---射影摄像机)
- [7. Computation of the Camera Matrix P](#7-computation-of-the-camera-matrix-p)
    - [7.0.1. Distortion in Camera Calibration](#701-distortion-in-camera-calibration)
    - [7.0.2. camera matrix](#702-camera-matrix)
- [8. N-View Computational Methods - N 视图计算方法](#8-n-view-computational-methods---n-视图计算方法)


# 1. introduction - a Tour of Multiple View Geometry

给予本书的一些主要思想, 以及一些基本概念 


## 1.1. 射影几何无处不在 - ubiquitous projective geometry

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
* 把欧式空间中的点表示为 齐次向量 (hemogeneous vector) 即可实现把欧氏空间拓展到射影空间, 这对于任何维度都是有效- [1. introduction - a Tour of Multiple View Geometry](#1-introduction---a-tour-of-multiple-view-geometry)
  - [1.1. 射影几何无处不在 - ubiquitous projective geometry](#11-射影几何无处不在---ubiquitous-projective-geometry)
    - [1.1.1. Affine Geometry 仿射和欧式几何](#111-affine-geometry-仿射和欧式几何)
  - [1.2. 摄像机投影 Camera Projections](#12-摄像机投影-camera-projections)
- [1. introduction - a Tour of Multiple View Geometry](#1-introduction---a-tour-of-multiple-view-geometry)
  - [1.1. 射影几何无处不在 - ubiquitous projective geometry](#11-射影几何无处不在---ubiquitous-projective-geometry)
    - [1.1.1. Affine Geometry 仿射和欧式几何](#111-affine-geometry-仿射和欧式几何)
  - [1.2. 摄像机投影 Camera Projections](#12-摄像机投影-camera-projections)
- [2. Projective Geometry and Transformations of 2D - 2D 下的射影几何与变化](#2-projective-geometry-and-transformations-of-2d---2d-下的射影几何与变化)
  - [2.1. Planar geometry - 平面几何](#21-planar-geometry---平面几何)
- [3. Projective Geometry and Transformations of 3D - 3D 下的射影几何与变化](#3-projective-geometry-and-transformations-of-3d---3d-下的射影几何与变化)
- [4. Estimation - 2D Projective Transformations - 估计2D射影变换](#4-estimation---2d-projective-transformations---估计2d射影变换)
- [5. Algorithm Evaluation - 算法评价和误差分析](#5-algorithm-evaluation---算法评价和误差分析)
- [6. Camera Models - 摄像机模型](#6-camera-models---摄像机模型)
  - [6.1. Finite cameras - 有限摄像机](#61-finite-cameras---有限摄像机)
- [\\end{pmatrix}](#endpmatrix)
  - [The projective camera - 射影摄像机](#the-projective-camera---射影摄像机)
- [7. Computation of the Camera Matrix P](#7-computation-of-the-camera-matrix-p)
    - [7.0.1. Distortion in Camera Calibration](#701-distortion-in-camera-calibration)
    - [7.0.2. camera matrix](#702-camera-matrix)
- [8. N-View Computational Methods - N 视图计算方法](#8-n-view-computational-methods---n-视图计算方法)


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

### 1.1.1. Affine Geometry 仿射和欧式几何

总结下来, 通过在欧氏空间中添加无穷远直线(平面) 即可获得射影空间 (Projective Space) , 那么对于该过程的逆过程进行讨论  

仿射几何:  是射影几何的特殊化, 特殊在存在了无穷远直线(面)
* 首先, 再次重申射影空间的特性
  * 射影空间是保证齐次的, 即没有一开始既定的坐标系
  * 尽管在理解上可以说, 平行线在无穷远相交, 但是在射影空间中, 无穷远也是一个普通的点, 即所有点都是平等的
  * 总的来说, 平行度 (parallelism) 在射影几何下是一个无意义的话题   
* 对于仿射几何来说, 指定一条特殊的直线, 尽管所有点都是等价, 但是有一些却带了特殊性
  * 具体表现为, 在实际上的图像中, 铁轨的无穷远相交于地平线, 平坦地区的图片上, 无穷远显示为地平线
  * 因此, 现实中的无穷远是可以被对应到图像中的点的, 即地平线
  * 即, 射影平面和一条特殊直线所组成的几何称为`仿射几何`.
  * 即, 将一个空间中的特殊直线映射到另一个空间中的特殊直线的任何`射影变换`称为 `仿射变换`
* 通过辨别出一条直线为 无穷远直线, 使得平行性可以被定义. 
* 对于欧式几何来说, 通过在射影平面中区分出一条特殊的直线, 就可以获得平行的概念, 以及伴随它的 `仿射几何` .  


绝对二次曲线 (Absolute Conic) : 通过在无穷远上挑出某些特殊角色, 来让仿射几何转换成欧式几何
* 圆与椭圆
  * 在二维欧式几何中, 圆(Circles) 可以与椭圆(ellipses)区分开, 但在仿射几何空间中, 任何椭圆都可以被拉伸成圆.
  * 因此仿射空间中, 不会区分圆和椭圆
* 代数上, 椭圆由二次方程描述, 两个椭圆一般会相交于4个点, 然而圆只会相交于两个点
* 既然仿射空间中, 圆和椭圆是等价的, 那么圆一定具有其特殊性使得在平面上只会相交于两个点:
  * 两个圆还相交于复值上的两个点 (Complex Point)
  * 在齐次坐标$(x,y,w)$中表示圆 : $(x-aw)^2+(y-bw)^2=r^2w^2$
  * 它表示了圆心在齐次坐标 $(x_0,y_0,w_0)^T=(a,b,1)^T$ 的圆
  * 可以直接验证, 两个复值点 $(1,\pm i,0)$ 位于任何一个圆上, 即两个任意的圆都交于这两个点, 且由于这两个点的最后一维是 0 , 因此其位于无穷远直线上 
  * $(1,\pm i,0)$  被称为平面的 `虚圆点` (Circular Points of the plane), 虚圆点满足实方程 $x^2+y^2=0, w=0$
* 有了虚圆点, 就可以清晰的从射影几何中给出欧式几何:
  * 先挑出一条无穷远直线, 再在该直线上挑出两个点作为 虚圆点
  * 可以定义一个圆为通过这两个虚圆点的任意二次曲线
  * 在把一个欧式结构赋给一个射影平面的时候, 可以指定任何一条直线和该直线上任意两个点(复值) 作为无穷远直线和虚圆点  

```
知识补充:  二次曲线 / 圆锥曲线 / conic section

包括了 椭圆, 抛物线, 双曲线
到平面内一定点的距离 r 与 到定直线的距离 d , 之比是常熟 e = r/d 的点的轨迹叫圆锥曲线

e > 1 时为双曲线
e = 1 时为抛物线
0 < e < 1 时为椭圆 
```
* 观点的证明:
  * 一般情况下, 一条二次曲线 (conic, 圆锥曲线) 可以被平面下任意 5 个点来唯一定义
  * 而特殊的是圆只需要 3 个点
  * 因此通过虚圆点, 可以归一化的认识到, 所有二次曲线都需要 5 个点来唯一定义 ( 所有圆都经过两个 虚圆点 )


3D 的欧式几何与 绝对二次曲线:
* 代数下的定理: 两个一般的椭球面 (二次曲面) 相交于一条一般的 四次曲线  
* 然而将之前 2D 下的思考拓展到 3D 空间中, 两个球面相交于一个圆. 
* 书中没有给出一般下的球面齐次坐标方程, 只是说了
  * 其次坐标 $(X, Y, Z, T)$ 中所有的球面与无穷远平面相交于一条方程为 $X^2+Y^2+Z^2=0, T=0$的曲线, 这是无穷远平面上的仅由复值点构成的曲线
  * $X^2+Y^2+Z^2=0, T=0$ 即为 绝对二次曲线
  * 是该书中最关键的几何实体, 与相机标定关联.
* 空间下的垂直度是欧氏几何的概念但不是仿射几何的概念
  * 对于两条相交的直线, 直线的垂直度可根据绝对二次曲线定义如下
  * 延申两条直线直到它们与无穷远平面, 得到两个点, 并成这两个点为 两条直线各自的`方向`
  * 两条直线的垂直度根据 这两个`方向` 与绝对二次曲线的关系来定义
  * 如果这两个 `方向` 关于绝对二次曲线是共轭点 (conjugate) , 则这两条直线是垂直的  

## 1.2. 摄像机投影 Camera Projections

本书最重要的主题是 : 图像的形成过程 - 3维世界的一种 2维表示的形成  

<!-- omit from toc -->
# Part 0, The Background : Projective Geometry, Transformations and Estimation

基础知识, 射影几何, 变换, 估计

射影几何的思想和表示法是多视图几何分析的核心, 例如
* 使用齐次坐标就能够 用线性矩阵方程来表示非线性映射, 可以很自然的表示无穷远点
* 第二章介绍 2D 下的射影变换
* 第三章介绍 3D 下的射影几何和绝对二次曲线
* 第四章介绍 由图像测量进行几何估计
* 第五章介绍 如何评价估计算法的结果, 计算估计的协方差


# 2. Projective Geometry and Transformations of 2D - 2D 下的射影几何与变化

理解本书所必要的几何概念和记号

Projective transformations (射影变换)

Perspective camera (透视摄像机)
Perspective imaging (透视成像) 

Collinearity (保线性) : 直线被成像为直线  

射影几何对 透视成像的 各种性质进行建模并提供适用于计算的数学表达  

## 2.1. Planar geometry - 平面几何

平面几何是生活中的基本概念, 平面几何研究的是点和直线以及它们之间的关系

几何研究的发展:
* 传统论者上, 几何研究应该坚持 几何的(Geometric) viewpoint, 即与坐标无关的. 定理的叙述和证明应当仅使用几何的公理而不使用代数. 经典的欧式方法就是例子.
* 自笛卡尔(Descartes)后,  


# 3. Projective Geometry and Transformations of 3D - 3D 下的射影几何与变化

# 4. Estimation - 2D Projective Transformations - 估计2D射影变换

# 5. Algorithm Evaluation - 算法评价和误差分析 


<!-- omit from toc -->
# Part 1, Camera Geometry and Single View Geometry - 摄像机几何和单视图几何

single perspective camera 单个透视摄像机

3D 场景空间到 2D 图像平面的投影  

# 6. Camera Models - 摄像机模型

摄像机即为 3D 世界和 2D 图像之间的一种映射, 该书讨论的主要是 中心投影 (central projection)

该章节对几种摄像机模型进行了推导, 证明了摄像机映射模型可以通过特殊的矩阵来表示  

总的来说, 中心投影 central projection 的摄像机是 一般射影摄像机 (General Projective Camera) 的特殊情况.  

摄像机模型可以分成两类:
* 有限中心
* 无穷远中心 , 仿射摄像机 (affine camera) 是平行投影的自然推广, 因此非常重要

摄像机模型主要讨论点的投影

## 6.1. Finite cameras - 有限摄像机

从具体且最简单的摄像机模型即基本的针孔摄像机开始, 通过一系列的升级来将模型一般化  

推导基于最一般的 CCD (Charge-coupled Device, 电子耦合器件) 传感器相机, 同时也适用于其他相机, 例如 X-射线图, 扫描负片, 扫描放大负片等

基本针孔模型 (basic pinhole model) : 空间点到一张平面上的中心投影
* 投影中心位于一个欧式坐标的原点
* 考虑 图像平面(image plane)或者聚焦平面(focal plane) $Z=f$
* 空间坐标上的点 $X=(X,Y,Z)^T$ 被映射到图像平面上的一点
* 映射在图像平面上的点的坐标可根据相似三角形来计算
* $(X,Y,Z)^T \rightarrow (fX/Z, fY/Z)^T$

由此定义了一系列的概念
* camera centra (摄像机中心) : 即投影中心, 也称为光心 (optical centre)
* principal axis / principal point (主轴, 主射线) : 摄像机中心到图像平面的垂线
* principal point (主点) : 主轴与图像平面的交点
* principal plane (主平米) : 过摄像机中心且平行于图像平面的平面

用齐次坐标来表示中心投影, 首先数学表示上
* $diag$ 表示对角矩阵
* $[I|0]$ 的形式表示了一个 I 矩阵右侧加上一个 0 向量, 在中心投影里 I 是3x3的
* 中心投影的摄像机矩阵为 $P=diag(f,f,1)[I|0]$, P 具体的用于表示 `3x4齐次摄像机投影矩阵`
* 投影过程可以简化表示为 $x=PX$

$$
\begin{pmatrix}X\\Y\\Z\\1\end{pmatrix}
\rightarrowtail
\begin{pmatrix}
  fX\\fY\\Z
\end{pmatrix}
=
\begin{bmatrix}
  f &&&0\\
  &f&&0\\
  &&1&0\\
\end{bmatrix}
\begin{pmatrix}
  X\\Y\\Z\\1
\end{pmatrix}
$$

主点偏置 (Principal point offset) : 图像平面的坐标原点不在摄像机主点上的话, 则会产生 offset
* 此时投影可以表示为 $(X,Y,Z)^T \rightarrow (fX/Z+p_x, fY/Z+p_y)^T$
* $p_x, p_y$ 是主点在图像平面上的坐标
* 摄像机标定矩阵 (camera calibration matrix) : 带有主点偏置的中心投影的摄像机矩阵, 用 K 表示
* $X_{cam}$ 表示该点坐标是 摄像机坐标系 (camera coordinate frame), 其 Z 轴是沿着设想机的主轴的, 从坐标系上与图像平面坐标系区别开
$$
\begin{bmatrix}
    f &&p_x&0\\
  &f&p_y&0\\
  &&1&0\\
\end{bmatrix}
,K=
\begin{bmatrix}
    f &&p_x\\
  &f&p_y\\
  &&1\\
\end{bmatrix}, x=K[I|0]X_{cam}
$$


摄像机的旋转与平移 : 一般情况下, 空间里的点会以不同的欧式坐标系来表示, 这些坐标系称为 世界坐标系 (world coordinate frame)
* 两个坐标系可以通过旋转和平移相联系  
* 用 $\tilde{X}$ 来表示一般的 3维非齐次坐标, 表示直接坐标系中的一点, 而 $\tilde{X}_{cam}$ 是摄像机坐标系来表示的同一个点的坐标 
* 则, 可以联系两个坐标 $\tilde{X}_{cam} = R(\tilde{X}-\tilde{C})$
* R 是一个表示摄像机坐标系方向的 `3x3` 旋转矩阵
* 坐标转换可以表示为:
$$
X_{cam} = \begin{bmatrix}
 R & -R\tilde{C} \\ 0^T&1
\end{bmatrix}
\begin{pmatrix}
  X\\Y\\Z\\1
\end{pmatrix}
= \begin{bmatrix}
 R & -R\tilde{C} \\ 0^T&1
\end{bmatrix} X
$$

和带偏置中心投影的变换结合在一起, 有
$$x=KR[I|-\tilde{C}]X$$
* X 为点的世界坐标系表示
* 这是一个针孔摄像机的一般映射
* 对于一个一般的 针孔摄像机矩阵 $P=KR[I|-\tilde{C}]$, 简要的说
  * 一共有 9 个自由度
  * 摄像机标定矩阵 K 中的 3个为 $f,p_x,p_y$, 被称为内参数 / 内部校准 (internal camera parameters / internal orientation)
  * R 和 C 则各有 3 个自由度, 和摄像机在世界坐标系中的方位和位置有关, 称为外参数, 或外部校准 (external parameters / exterior orientation)
* 通常来说, 为了方便, 直接将 $\tilde{X}_{cam} = R(\tilde{X}-\tilde{C})$ 表示为 $\tilde{X}_{cam} = R\tilde{X}+t$, 其中 $t=-R\tilde{C}$


CCD 摄像机 : 实际中, 一个 CCD 传感器的像素可能不是正方形, 因此在 x,y 轴上 图像平面到现实世界中的尺度因子是不同的
* 如果定义 : 图像坐标中, 单位距离的像素数分别是 $m_x, m_y$
* 世界坐标到像素坐标的变换式需要左乘一个 $diag(m_x,m_y,1)$
* 增加了一个自由度
即 
$$K=\begin{bmatrix}
  a_x&&x_0\\&a_y&y_0\\&&1
\end{bmatrix}, a_x=fm_x, a_y=fm_y$$

有限射影摄像机 (finite projetive camera) : 增加了 s 扭曲参数, 尽管大多数相机中 s 都为 0, 但仍然有特殊情况, 又增加了一个自由度

最终, 一个有限射影摄像机的矩阵有 11 个自由度, 这与定义到 `相差一个任意尺度因子` 的 3x4 矩阵的自由度数目一样  
* 有限射影摄像机的摄像机矩阵的集合 等于
* 左边 3x3子矩阵为非奇异性的 3x4 齐次矩阵所构成的集合


一般射影摄像机 (General projective cameras) : 左边 3x3 子矩阵为任意秩为3的 3x4 其次矩阵

<!-- 完 -->

## The projective camera - 射影摄像机  


# 7. Computation of the Camera Matrix P
Camera Calibration 相机标定

也被叫做 camera resectioning

pinhole camera 是一种最简单的相机, 没有 lens, 只有一个光圈 aperture.  
光线通过光圈然后将图像的颠倒投影在相机里.  

### 7.0.1. Distortion in Camera Calibration

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


### 7.0.2. camera matrix
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



<!-- omit from toc -->
# Part 4, N-View Geometry - N 视图几何



# 8. N-View Computational Methods - N 视图计算方法