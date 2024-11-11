- [1. Introduction to Visual SLAM : From Theory to Practice](#1-introduction-to-visual-slam--from-theory-to-practice)
  - [1.1. Classical Visual SLAM Framework](#11-classical-visual-slam-framework)
  - [1.2. Mathematical Formulation of SLAM problems](#12-mathematical-formulation-of-slam-problems)
- [2. 3D Rigid Body Motion](#2-3d-rigid-body-motion)
  - [2.1. Rotation Matrix](#21-rotation-matrix)
    - [2.1.1. Points, Vectors, and Coordinate Systems](#211-points-vectors-and-coordinate-systems)
    - [2.1.2. Euclidean Transforms Between Coordinate Systems 坐标系之间的欧氏变换](#212-euclidean-transforms-between-coordinate-systems-坐标系之间的欧氏变换)
    - [2.1.3. Transform Matrix and Homogeneous Coordinates 变换矩阵与齐次坐标](#213-transform-matrix-and-homogeneous-coordinates-变换矩阵与齐次坐标)
  - [2.2. Rotation Vectors and the Euler Angles 旋转向量和欧拉角](#22-rotation-vectors-and-the-euler-angles-旋转向量和欧拉角)
    - [2.2.1. Rotation Vectors 旋转向量](#221-rotation-vectors-旋转向量)
    - [2.2.2. Euler Angles 欧拉角](#222-euler-angles-欧拉角)
  - [2.3. Quaternions 四元数](#23-quaternions-四元数)
    - [2.3.1. Definition 四元数的定义](#231-definition-四元数的定义)
    - [2.3.2. Quaternion Operations 四元数运算](#232-quaternion-operations-四元数运算)
    - [2.3.3. Use quaternion to Represent a Rotation 用四元数表示旋转](#233-use-quaternion-to-represent-a-rotation-用四元数表示旋转)
    - [2.3.4. Conversion of Quaternions to Other Rotation Representations 四元数到其他表示方法的转换](#234-conversion-of-quaternions-to-other-rotation-representations-四元数到其他表示方法的转换)
- [3. Lie Group and Lie Algebra 李群与李代数](#3-lie-group-and-lie-algebra-李群与李代数)
  - [3.1. Basic of Lie Group and Lie Algebra 李群与李代数](#31-basic-of-lie-group-and-lie-algebra-李群与李代数)
    - [3.1.1. Group, Lie Group - 群, 李群](#311-group-lie-group---群-李群)
    - [3.1.2. Introduction of the Lie Algebra - 李代数的引出](#312-introduction-of-the-lie-algebra---李代数的引出)
    - [3.1.3. The Definition of Lie Algebra - 李代数的定义](#313-the-definition-of-lie-algebra---李代数的定义)
    - [3.1.4. Lie Algebra 李代数 $\\mathfrak{so(3)}$](#314-lie-algebra-李代数-mathfrakso3)
    - [3.1.5. Lie Algebra 李代数 $\\mathfrak{se(3)}$](#315-lie-algebra-李代数-mathfrakse3)
  - [3.2. Exponential and Logarithmic Mapping - 李代数的指数与对数映射](#32-exponential-and-logarithmic-mapping---李代数的指数与对数映射)
- [4. Cameras and Images](#4-cameras-and-images)
  - [4.1. Pinhole Camera Models](#41-pinhole-camera-models)
    - [4.1.1. Pinhole Camera Germotry](#411-pinhole-camera-germotry)
    - [4.1.2. Distortion](#412-distortion)
    - [4.1.3. Stereo Cameras](#413-stereo-cameras)
    - [4.1.4. RGB-D camera](#414-rgb-d-camera)
  - [4.2. 图像 - Image](#42-图像---image)
- [5. Nonlinear Optimization - 非线性优化方法](#5-nonlinear-optimization---非线性优化方法)
- [6. Visual Odometry - 视觉里程计 Part 1](#6-visual-odometry---视觉里程计-part-1)
  - [6.1. Feature Method - 特征点法](#61-feature-method---特征点法)
    - [6.1.1. Feature Point](#611-feature-point)
    - [6.1.2. ORB Feature](#612-orb-feature)
    - [6.1.3. Feature Matching](#613-feature-matching)
  - [6.2. 2D-2D : Epiploar Geometry - 对极几何](#62-2d-2d--epiploar-geometry---对极几何)
    - [6.2.1. Epipolar Constraints - 对极约束](#621-epipolar-constraints---对极约束)
    - [6.2.2. Essential Matrix - 本质矩阵](#622-essential-matrix---本质矩阵)
    - [6.2.3. Homography Matrix - 单应矩阵](#623-homography-matrix---单应矩阵)
  - [6.3. Triangulation - 三角测量 (三角化)](#63-triangulation---三角测量-三角化)
  - [6.4. 3D-2D : PnP](#64-3d-2d--pnp)
    - [6.4.1. Direct Linear Transformation 直接线性变换](#641-direct-linear-transformation-直接线性变换)
    - [P3P](#p3p)
    - [Minimizing the Reprojection Error](#minimizing-the-reprojection-error)
  - [6.5. 3D-3D : ICP](#65-3d-3d--icp)
    - [6.5.1. SVD - Linear Algebra Method](#651-svd---linear-algebra-method)
    - [6.5.2. Non-linear Optimization Method](#652-non-linear-optimization-method)
- [7. Visual Odometry - 视觉里程计 Part 2](#7-visual-odometry---视觉里程计-part-2)
  - [7.1. The Motivation of the Direct Method - 直接法的引出](#71-the-motivation-of-the-direct-method---直接法的引出)
  - [7.2. 2D Optical Flow - 2D 光流](#72-2d-optical-flow---2d-光流)
    - [7.2.1. Lucas-Kanade 光流](#721-lucas-kanade-光流)
    - [7.2.2. 光流法实践](#722-光流法实践)
  - [7.3. Direct Method 直接法](#73-direct-method-直接法)
    - [7.3.1. Derivation of the Direct Method - 直接法的推导](#731-derivation-of-the-direct-method---直接法的推导)
    - [7.3.2. 直接法的讨论](#732-直接法的讨论)
  - [7.4. Practice Direct Method - 实践中的直接法](#74-practice-direct-method---实践中的直接法)
- [8. Filters and Optimization Approaches - 后端 Part1](#8-filters-and-optimization-approaches---后端-part1)
  - [8.1. Introduction of backend](#81-introduction-of-backend)
    - [8.1.1. State Estimation from Probabilistic Perspective - 状态估计的概率解释](#811-state-estimation-from-probabilistic-perspective---状态估计的概率解释)
    - [8.1.2. Linear Systems and the Kalman Filter - 线性系统与卡尔曼滤波KF](#812-linear-systems-and-the-kalman-filter---线性系统与卡尔曼滤波kf)
    - [8.1.3. Nonlinear Systems and the EKF - 非线性系统和EKF](#813-nonlinear-systems-and-the-ekf---非线性系统和ekf)
    - [8.1.4. Discussion About KF and EKF](#814-discussion-about-kf-and-ekf)
  - [8.2. Bundle Adjustment and Graph Optimization - BA 与图优化](#82-bundle-adjustment-and-graph-optimization---ba-与图优化)
    - [8.2.1. The Projection Model and COst Function - 投影模型和 BA 代价函数](#821-the-projection-model-and-cost-function---投影模型和-ba-代价函数)
    - [8.2.2. Solving Bundle Adjustment - BA的求解](#822-solving-bundle-adjustment---ba的求解)
    - [8.2.3. Sparsity - 稀疏性](#823-sparsity---稀疏性)
    - [8.2.4. Robust Kernels - 鲁棒核函数](#824-robust-kernels---鲁棒核函数)
- [9. Filters and Optimization Approaches - 后端 Part2](#9-filters-and-optimization-approaches---后端-part2)
  - [9.1. Sliding Window Filter and Optimization - 滑动窗口滤波和优化](#91-sliding-window-filter-and-optimization---滑动窗口滤波和优化)
    - [9.1.1. Controlling the Structure of BA - BA 的结构在实际环境下的控制](#911-controlling-the-structure-of-ba---ba-的结构在实际环境下的控制)
    - [9.1.2. Sliding Window - 滑动窗口法](#912-sliding-window---滑动窗口法)
  - [9.2. Pose Graph Optimization - 位姿图](#92-pose-graph-optimization---位姿图)
    - [9.2.1. Definition of Pose Graph - 位姿图的定义](#921-definition-of-pose-graph---位姿图的定义)
    - [9.2.2. Residuals and Jacobians - 位姿图的优化方法](#922-residuals-and-jacobians---位姿图的优化方法)
- [10. Loop Closure - 回环检测](#10-loop-closure---回环检测)
- [11. Dense Reconstruction - 建图](#11-dense-reconstruction---建图)
- [12. Discussions and Outlook - SLAM 的现在与未来](#12-discussions-and-outlook---slam-的现在与未来)

# 1. Introduction to Visual SLAM : From Theory to Practice
Xiang Gao and Tao Zhang


定位 Localization 可以理解为一个 输入的概念
map building 可以理解为一个输出的概念  

单目SLAM monocular
* 单张的图片没有任何深度信息
* 通过移动摄像机以及帧之间的视差 disparity 可以获得物体的相对深度
* 由于等量的放大场景中的所有物体不会影响单目摄像机的成像, 因此单目视频流获取的 depth 信息永远缺少一个 scale 缩放参数且根本上无法通过计算得到

双目摄像机
* 通过双目视差可以获取场景中的相对深度, 根据 stereo 相机的 `baseline` 参数则可以获取绝对深度  
* stereo 配置所可以获取的最远深度距离也是根据 baseline 来决定的
* 整体上, 深度信息的精度取决于 baseline length 和 成像解析度
* 标定和配置的复杂性是最大的缺点, 计算复杂性是目前最大的瓶颈


RGB-D 
* 通过不可见光的反射来计算深度, 计算流程一般被嵌入到相机本体中, 因此可以节省计算资源
* 可以通过单张成像获得点云
* 高噪声, 深度范围有限, 视野范围 FoV小, 受阳光干扰, 无法精准应对透明物体等
* 目前主要应用在室内SLAM, 不适合室外  


## 1.1. Classical Visual SLAM Framework


1. Sensor data acquisiton. 获取各种传感器数据, 图像, IMU, motor encoders 等所有.
2. Visual Odometry (VO) 前端视觉里程计. 也被称为 `frontend`, 通过较小范围的相邻帧, 获取该小范围内的相机移动, 生成一个 rough local map. 通过将 VO 得到的相机序列进行累加, 即可得到最简易的 SLAM 系统, 然而 accumulative drift 是这种方法最主要的问题, 也是 SLAM 系统为什么还需要一个全局的 backend 以及 loop closing的原因.
3. Backend filter/optimization. 从 VO 以及 loop closing的结果中 获取 camera poses at different time stamps. 根据以上结果来进行优化, 得到一个优化后的 追踪以及地图. 由于其是接在 VO 之后的, 因此常被称为 `backend`. 后端主要进行全局优化, 只接受移动信息, 并对移动信息进行 "降噪".
4. loop closing. 用于判断相机是否回到了之前待过的空间位置, 用于减少累加的定位漂移, 如果 loop 被检测到了, 则需要将该 loop 信息提供给 backend 用于进一步的优化之前的 map. 因此 loop closing 则属于一个视觉问题, 即 calculating similarities of images.  即使在 laser-based SLAM 系统中, loop closing 也很关键.
5. reconstruction. 根据 camera trajectory 建立一个 task-specific map. Map 的种类根据具体的需求不同有多种表现形式, 可以是 点云, 也可以是精美的 3D model, 或者整个场景的图片. 因此在 map recon 的领域中没有具体的算法. 通常, map 的种类可以大致被分为两种 : 
   1. metrical map : 公制地图? 会记录下具体的物体度量位置. 可以分为 sparse / dense. sparse 即只记录最关键的特征点, 可以用于定位. 而对于自动驾驶中更复杂的 导航 navigation 则需要记录场景中更多的信息, 即需要 dense map. 无论怎样, 记录这些特征点都是 storage expensive. 且会面临在大场景中, 相似特征场景多次出现的重叠问题.
   2. topological map : 更多的关注 map 中元素的相对关系. 由 nodes 和 edges 来组成map. 目前仍然是一个 open problem to be studied.  


总体上来看, 在 Visual SLAM 课题上
* frontend 可以说更加贴近于 Computer Vision 领域, 例如 image feature extraction and matching
* backend 由于不处理图像, 更加贴近于  state estimation 领域
* 在研究领域中, SLAM research 在很长一段时间里仅仅指代 backend 部分, 因为它更关键. 
  * 解决该问题需要使用到 state estimation theory 来表示 uncertainty of localization and map construction.
  * 使用 filters 方法或者 nonlinear optimization 来计算 state 的平均以及不确定性.  


## 1.2. Mathematical Formulation of SLAM problems

最简易的模型, 可以把 SLAM 概括为
* motion equation : pose 随时间的变换
* observation equation : landmark 随时间的观测情况的变化


在实际中根据两点
* motion and observation equations are linear?  (L/NL)
* noise is assumed to be Gaussian? (G/NG)

来将具体场景分为 4 大类  
* LG 系统是最简单的, 可以通过 Kalman Filter 来进行优化
* NLNG 是非常复杂的, 一般的对应方法可以分为
  * Extended Kalman Filter (EKF) 方法 : 到21世纪初都是主流. 但仍然有一些不足 (Linearization oerror, noise Gaussian distribution assumptions)
  * nonlinear optimization : 例如 graph optimization. 计算量大是其在应用上的瓶颈, 但在未来是更优的选择.  


# 2. 3D Rigid Body Motion

如何描述一个物体在 3D 空间下的移动

直观上, motion = rotation + translation

## 2.1. Rotation Matrix


### 2.1.1. Points, Vectors, and Coordinate Systems

在线性代数的视角下, 对于一个向量, 可以找到一组基 $(e_1,e_2,e_3)$  

任意向量 $\vec{a}$ 在该组基下就有一个坐标 $(a_1,a_2,a_3)^T$
$$
\vec{a}=[e_1,e_2,e_3]
\begin{bmatrix}
  a_1\\ a_2 \\ a_3
\end{bmatrix}
= a_1e_1+a_2e_2+a_3e_3
$$

定义向量的内积 inner product 为, 其中 $<a,b>$ 定义为两个向量的夹角, 内积可以描述向量间的投影关系  
$$
a\cdot b = a^Tb = \sum^{3}_{i=1}a_ib_i=|a||b|\cos<a,b>
$$

外积则可以定义为: $a\times b = a\^{}b$
$$
a \times b = 
\begin{Vmatrix}
  e_1 & e_2& e_3 \\
  a_1 & a_2& a_3 \\
  b_1 & b_2& b_3 
\end{Vmatrix} = 
\begin{bmatrix}
  a_2b_3-a_3b_2\\
  a_3b_1-a_1b_3\\
  a_1b_2-a_2b_1
\end{bmatrix} =
\begin{bmatrix}
  0 & -a_3 &a_2\\
  a_3 & 0 & -a_1 \\
  -a_2 & a_1 & 0
\end{bmatrix}b
$$

两个向量的外积结果是一个向量, 方向垂直 (perpendicular) 于两个向量构成的面, 且大小为 $|a||b|\sin<a,b>$  
是 **两个向量张成的四边形的有向面积** area of the quadrilateral  

定义 $\^{}$ 为把一个向量变成反对称矩阵的形式 skew-symmetric matric.  
反对称矩阵满足 $A^T = -A$  
这样就把向量的外积转换为矩阵向量乘法, 一个向量对应着唯一一个 反对称矩阵.  


要注意, 尽管计算上是根据向量的坐标来计算的, 但是向量之间的 加减法, 内外积运算其实是独立于坐标的, 即使在不谈论它们的坐标时也可以计算.  

### 2.1.2. Euclidean Transforms Between Coordinate Systems 坐标系之间的欧氏变换

定义 : 某个点在相机坐标系下的坐标为 $p_c$, 而在世界坐标系下的坐标为 $p_w$   

对于两个坐标系之间的移动, 是一种 刚体移动 rigid body motion. 即相机 camera 的运动是刚体的 (rigid).  在运动的过程中, length and angle of the vector will not change.    

这种刚体移动下, 可以描述坐标系之间相差的是一个 欧氏变换 (Euclidean transform)

旋转矩阵的推导:  假设某个向量在旋转前后的坐标分别为 $a$ 和 $a'$, 坐标系的基为 $\bold{e}$ 和 $\bold{e'}$

则由向量的不变性引出的  
$$
[e_1,e_2,e_3]
\begin{bmatrix}
  a_1 \\
  a_2 \\
  a_3 
\end{bmatrix} =
[e_1',e_2',e_3']
\begin{bmatrix}
  a_1'\\
  a_2'\\
  a_3'
\end{bmatrix}
$$

等式两边同时左乘 $[e^T_1,e^T_2,e^T_3]^T$ 则有左边的系数变为了单位矩阵, 即 
$$
\begin{bmatrix}
  a_1\\
  a_2\\
  a_3
\end{bmatrix} =
\begin{bmatrix}
  e^T_1e_1'& e^T_1e_2' & e^T_1e_3' \\
  e^T_2e_1'& e^T_2e_2' & e^T_2e_3' \\
  e^T_3e_1'& e^T_3e_2' & e^T_3e_3' \\
  
\end{bmatrix}
\begin{bmatrix}
  a_1'\\
  a_2'\\
  a_3'
\end{bmatrix}
= Ra'
$$

可以将中间的矩阵拿出来, 定义为旋转矩阵 R.  旋转矩阵是由两组基的内积所构成的, 刻画了旋转前后同一个向量的坐标变换关系.  

R 描述了旋转本身, 该矩阵的各个分量是两个坐标系基的内积, 由于基向量本身的长度为1 , 即实际上矩阵的各个分量是各个基向量夹角的余弦值, 所以该R矩阵也叫做 方向余弦矩阵 (direction cosine matrix)   

旋转矩阵有一些特殊的性质:
* 行列式 determinant  为 1 ( 1这个数字是人为定义的)
  * 行列式为 -1 的称为 瑕旋转 improper rotation, 即一次旋转加一次反射  
* 是正交 orthogonal 矩阵 ( 逆矩阵为自身的转置 )

因此可以定义 n 维旋转矩阵为一个集合:
$$
SO(n) = {R\in \mathbb{R}^{n\times n} | RR^T=I, \det(R)=1}
$$

称 SO(n) 为 特殊正交群 special orthogonal group.  
带入到原本的旋转变换中, 有 $a' = R^{-1}a = R^Ta$, 即 $R^T$ 定义了一个相反的旋转.  

在欧式变换中, 除了旋转还有平移, 加上平移后, 就有了完整的欧式变换
$$
a'=Ra+t
$$

在实际中, 对于坐标系12之间的变换, 有完整的写法. 这里 12 的意思是 由2变为1, 因为向量在这个矩阵的右边, 因此矩阵的下标读法是从右到左的.
$$
a_1=R_{12}a_2+t_{12}
$$


关于平移向量 t, 在拥有下标的时候, t12 对应的是坐标系1原点指向坐标系2原点的向量, 该平移向量是在坐标系1下取得的. 可以将该向量读作从 1 到 2 的向量.  
同理, 对于 t21, 即从坐标系2的原点指向坐标系1的原点, 由于还要考虑 旋转的因素, 因此 $t_{21} \not ={} -t_{12}$.  

在摄像机视角下, 摄像机坐标指的是世界坐标系下相机原点的坐标, 即 $t_{WC}$, 是在世界坐标系下求得的向量, 用于摄像机坐标转为世界坐标.  
而相反的  $t_{WC} = -R^T_{CW}t_{CW} \not = -t_{CW}$
* 从相机坐标系看向世界坐标系原点的向量 $t_{CW}$
* 世界坐标系旋转为相机坐标系的旋转矩阵 $R_{CW}$ 的转职-> 相机坐标系旋转为世界坐标系的方向
* 再取负数即可

### 2.1.3. Transform Matrix and Homogeneous Coordinates 变换矩阵与齐次坐标

引入齐次坐标, 使得旋转和平移能够写在同一个矩阵里, 称为变换矩阵 T (transformation matrices)  

$$
\begin{bmatrix}
  a'\\
  1 \\
\end{bmatrix}=
\begin{bmatrix}
  R& t\\
  0^T & 1
\end{bmatrix}
\begin{bmatrix}
  a\\
  1\\
\end{bmatrix}=
T
\begin{bmatrix}
  a\\
  1
\end{bmatrix}
$$

在公式的书写上, 从此以后默认使用变换矩阵 T 的时候对 a 进行了齐次坐标的变换以及反变换.  
即 $b= T_1a$

对于变换矩阵 T 的构成
* 左上角为旋转矩阵
* 右侧为平移向量
* 左下角为 0 向量
* 右下角为 1 

这种特殊的矩阵又称为 特殊欧式群 $SE(3)$ special Euclidean group.  
$$
SE(3)=\{
T=\begin{bmatrix}
  R & t\\
  0^T & 1
\end{bmatrix}  
\in \mathbb{R}^{4\times4}| R\in SO(3), t\in \mathbb{R}^3\}
$$

同 SO(3) 一样, 求解该矩阵的逆等同于求解一个变换的反向变换:
$$
T^{-1} = \begin{bmatrix}
  R^T& -R^Tt\\
  0^T & 1
\end{bmatrix}
$$

在数式中, 不刻意区分齐次坐标计算以及非齐次坐标计算, 默认使用符合运算规则的那一种, 例如直接书写为 $Ta, Ra$等.  在C++中, 也可以方便的通过运算符重载来实现程序的运算结构统一.  

## 2.2. Rotation Vectors and the Euler Angles 旋转向量和欧拉角

### 2.2.1. Rotation Vectors 旋转向量

矩阵表达方式有两个缺点
* 旋转只有3个自由度但SO(3)有9个量, 同理变换用了 16 个量表达了6自由度的变换
* 旋转矩阵由于有约束, 即正交矩阵且行列式为1, 在计算上求解非常困难  

一个旋转可以由 一个旋转轴 rotation axis 和一个旋转角 rotation angle 来表示, 可以使用一个向量, 方向表示旋转轴, 而长度等于旋转角, 以此来用 3 个量表达3个自由度的旋转过程. 该向量则被称为 旋转向量 或者 轴角 rotation vector (or angle-axis / axis-angle). 

用$n$ 来表达旋转轴, $\theta$ 表示旋转角度, 那么 旋转向量即为 $\theta n$

从旋转向量到旋转矩阵的转换过程可以由 罗德里格斯公式 (Rodrigues' formula) 来表明. 这里符号 $\^{}$ 仍然表示获取一个向量的反对称矩阵.
$$
R= \cos\theta I +(1-\cos\theta)nn^T+\sin\theta n\^{}.
$$

反之, 从一个旋转矩阵到旋转向量的转换公式推导过程为, 取 R 的 trace (对角线元素之和)

$$
\begin{align*}
tr(R) & = \cos\theta tr(I)+(1-\cos\theta)tr(nn^T)+\sin\theta tr(n\^{})\\
&=3\cos\theta + (1-\cos\theta)\\
&=1+2\cos\theta \\
\theta & =\arccos\frac{tr(R)-1}{2} 
\end{align*}
$$

对于旋转轴, 由于旋转轴上的向量在旋转前后不会发生改变, 因此有 $Rn=n$  
因此可以得知, 转轴 n 是旋转矩阵 R 的特征值 1 对应的特征向量, 求解方程再归一化后即可得到旋转轴.  

### 2.2.2. Euler Angles 欧拉角

无论是旋转矩阵还是旋转向量, 对于人类来说都是非常难以想象的  

欧拉角提供了一种对于人类来说直观的方法表述旋转, 即 通过三个分离的转角来将整个旋转分解为 3 次绕不同轴的旋转, 因为人类非常容易理解绕单个轴的旋转过程.  
然而在此之中, 仍然有容易混淆的部分:
* 如何分解 3次旋转的顺序, 先 XYZ?
* 旋转轴是按照全局固定的轴还是按照旋转之后的轴

由于这是一个应用上的问题, 因此数学上没有统一的定义, 在特定领域中, 有一些约定俗成的定义方式 : 偏航 - 俯仰 - 滚转 ( yaw - pitch - roll), 它等价于 ZYX 顺序的轴旋转 ( 按照旋转之后的轴 )

返回到 XYZ 上, 该旋转即可用一个 $[rpy]^T$ 的三维向量来描述 (尽管旋转顺序是 ZYX)
* z  yaw : 偏航, 与地面平行时候的左右
* y pitch : 俯仰, 上下
* x roll : 旋转

从欧拉角到旋转的定义可以由 简单的三角函数获取, 对于 Z-Y-X 旋转, 给定角度, 有旋转  

$$
R_{yaw} = \begin{bmatrix}
  \cos(yaw) & -\sin(yaw) & 0 \\
  \sin(yaw) & \cos(yaw)  & 0 \\
  0 & 0 & 1 \\
\end{bmatrix}
\\
R_{pitch} = \begin{bmatrix}
  \cos(pitch) & 0& \sin(pitch)  \\
  0 & 1 & 0\\
  -\sin(pitch)  & 0& \cos(pitch)  \\
\end{bmatrix}
\\
R_{roll} = \begin{bmatrix}
  1&0 & 0  \\
  0& \cos(roll) & -\sin(roll)\\
  0& \sin(roll) & \cos(roll) \\
\end{bmatrix}
$$
注意, 上述公式需要结合具体的欧拉角正方向定义来判断

数学上, 欧拉角和旋转向量的表述方法都存在 奇异性问题, 这使得 欧拉角和旋转向量在计算中不适用于 插值和迭代, 而只存在于人机交互上.  
欧拉角和旋转向量主要用于在系统中进行人为的查验错误与验证 (可视化)  

欧拉角的奇异性问题被单独称作 万向锁(Gimbal lock), 即在俯仰角为 +-90 度时, 第一次旋转 yaw 会和第三次旋转 roll 使用同一个轴, 这使得整个系统丢失了一个自由度.  

## 2.3. Quaternions 四元数

### 2.3.1. Definition 四元数的定义

到目前为止, 表述旋转的主要方法 旋转矩阵和旋转向量(欧拉角)都有问题
* 旋转矩阵用 9 个量描述 3 自由度的旋转, 具有冗余性
* 旋转向量 欧拉角是 紧凑的, 但是由奇异性
* 事实上, 不存在 不带奇异性的 三维向量描述方式
* 该事实可以通过降维来理解, 即通过 二维坐标来表述地球表面上的点, 一定会存在奇异性 (即 纬度为 +- 90度时经度无意义)

可以通过二维上的内容来理解四元数:
* 复数集 C 表示复平面上的向量, 复数间的乘法可以表示 复平面上的旋转
* 旋转复平面上的向量 theta 度时, 可以给这个向量乘以 $e^{i\theta}$
  * 欧拉公式 $e^{i\theta} = \cos i + i\sin i$
  * 逆时针旋转一个向量 90度时, 即给这个向量乘以 $i$

单位复数表示复平面上的旋转推广到 3维下, 即为 单位四元数描述 三维空间下的旋转

定义一个四元数 q, 拥有一个实部和三个虚部
$$ q = q_0 +q_1i+q_2j+q_3k$$

ijk分别为四元数的三个虚部, 由于是虚部, 且是单元四元数, 因此满足
* $i^2=j^2=k^2=-1$
* $ij=k, ji=-k$
* $jk=i, kj=-i$
* $ki=j, ik=-j$
这里在理解上, 可以理解为 ijk 分别是三个坐标轴, 相互之间的乘法是外积  

四元数还有一种表示方法 $q=[s,v]^T$, 这里 $s=q_0 \in \mathbb{R}, v=[q_1,q_2,q_3]^T \in \mathbb{R}^3$
* s 是四元数的实部, v 是虚部
* 如果虚部为0, 称为实四元数, 反之称为虚四元数  

单位四元数可以紧凑的, 不冗余的, 非奇异的表示一个旋转, 然而却是最抽象的
* 因为 ij=k 等性质, 乘以 i 代表着围绕 i 轴旋转 180 度
* 因为 i^2 =-1 等性质, 绕 i 轴旋转 360 度代表得到一个反转的东西, 需要绕两个 360 才能回到原本的位置  



### 2.3.2. Quaternion Operations 四元数运算

针对四元数的加减乘除  这里虚部分别用 xyz 表示 q1q2q3
* 加法: $q_a \plusmn q_b = [s_a \plusmn s_b, v_a \plusmn v_b]^T$ 实部虚部各个分量分别加减即可
* 乘法: 每项之间互相相乘最后相加, 注意充分理解 ij=k 
$$
\begin{align*}
q_aq_b = & s_as_b-x_ax_b-y_ay_b-z_az_b\\
& +(s_ax_b+x_as_b+y_az_b-z_ay_b)i \\
& +(s_ay_b+y_as_b-x_az_b+z_ax_b)j \\
& +(s_az_b+z_as_b+x_ay_b-y_ax_b)k
\end{align*}
$$

* 乘法通过向量形式, 并利用内外积的表示法, 可以更加简洁  
  * $q_aq_b=[s_as_b-v_a^Tv_b, s_av_b+s_bv_a+v_a\times v_b]^T$
  * 注意由于第四项外积的存在, 四元数乘法同矩阵一样是通常不可交换的, 除非外积项为0 , 此时 va 和 vb 是在 R3 中共线的存在

* 模长: 两个四元数乘积的模长等于 模长的乘积, 这满足了 单位四元数相乘后仍然是单位四元数
  * $||q_a||=\sqrt{s_a^2+x_a^2+y_a^2+z_a^2}$

* 共轭 : 四元数的共轭是把虚部写成相反数, 四元数的共轭乘以其本身, 会得到一个实四元数, 且实部(模长) 等于原本模长的平方
  * $q_a^*=[s_a, -v_a]^T$
  * $q^*q = qq^*=[s_a^2+v^Tv,0]$

* 逆: 四元数和自己逆的乘积为实四元数1, 如果 是单位四元数, 那么其逆等于其共轭
  * $q^{-1} = q^*/ ||q^2||$
  * $qq^{-1}=q^{-1}q=1$
  * 四元数的逆和矩阵的逆类似 $(q_aq_b)^{-1} = q_b^{-1}q_a^{-1}$

* 数乘: 和向量类似, 四元数可以与实数进行数乘
  * $kq=[ks, kv]^T$

### 2.3.3. Use quaternion to Represent a Rotation 用四元数表示旋转  

返回到传统的旋转表示  , 一个空间三位点 $p=[x,y,z]$, 一个四元数 q 表示的旋转
* 对于矩阵描述, 有 $p'=Rp$

如何把三位点表述成四元数 -> 三维空间上的点可以用一个虚四元数来表示
* $p=[0, x,y,z]^T = [0, v]^T$
* 此处终于体现了四元数的虚部与空间中三个轴的对应关系
* 旋转过程通过四元数即可表示为
* $p'=qpq^{-1}$

最终旋转的结果仍然是一个虚四元数  



### 2.3.4. Conversion of Quaternions to Other Rotation Representations 四元数到其他表示方法的转换   

四元数的乘法还可以写成一种矩阵的形式, 对于 $q=[s,v]^T$  

可以定义出 q 的两种矩阵表示  
$$
q^+=
\begin{bmatrix}
  s & -v^T \\
  v & sI+v\hat{} 
\end{bmatrix}, 
q^\oplus =
\begin{bmatrix}
  s & -v^T \\
  v & sI-v\hat{} 
\end{bmatrix}
$$

这种表示方法引申出两种旋转的计算  
$$q_1q_2 = q_1^{+}q_2 = q_2^\oplus q_1$$

从以上的结论进行反推, 有
* $p' = qpq^{-1}= q^+p^+q^{-1} = q^+q^{-1^\oplus}p$
* 那么则有 $p'= q^+q^{-1^\oplus}p = Rp$

$$
q^+(q^{-1})^\oplus=
\begin{bmatrix}
    s & -v^T \\
  v & sI+v\hat{} 
\end{bmatrix}
\begin{bmatrix}
    s & v^T \\
  -v & sI+v\hat{} 
3. \end{bmatrix}
=
\begin{bmatrix}
    1 & 0 \\
  0^T & vv^T+s^2I+2sv\hat{}+(v\hat{})^2
\end{bmatrix}
$$

由此, 即有了四元数到旋转矩阵的关系  
$$
R = vv^T+s^2I+2sv\hat{}+(v\hat{})^2
$$

再继续套用 旋转矩阵到旋转向量的转换公式  
* $tr(R) = 4s^2-1$
* $\theta = \arccos(\frac{tr(R)-1}{2})=\arccos (2s^2-1) = 2\arccos (s) = 2\arccos (q_0)$
* 对于旋转向量 $[n_x,n_y,n_z]^T =[q_1,q_2,q_3]^T/\sin\frac{\theta}{2}$

整理下, 即 四元数和 旋转向量的关系是非常近的
* 四元数的虚部代表的向量就是旋转矩阵的向量
* 而 四元数实部的值到欧拉角的转换即 $\theta = \arccos(q_0)$


# 3. Lie Group and Lie Algebra 李群与李代数  

在 SLAM 中, 除了对刚体的运动进行描述以外, 该需要进行估计和优化, 求解最优的 R,t  

在矩阵表述时, 由于旋转矩阵自身的约束 (正交且行列式为1), 作为优化变量时会引入额外的约束使得计算变得困难.  

通过李群-李代数之间的转换, 把位姿估计转变为无约束的优化问题, 简化求解方式  


## 3.1. Basic of Lie Group and Lie Algebra 李群与李代数


<!-- 章节完 -->
在旋转矩阵和变换矩阵的基础说明中
* 三维旋转矩阵构成了特殊正交群 special orthogonal group SO(3)
* 变换矩阵构成了 特殊欧式群 special Euclidean group SE(3)

完整的复习一遍定义为

$$
SO(3) = {R\in \mathbb{R}^{3\times 3} | RR^T=I, \det(R)=1}
\\
SE(3)={
T=\begin{bmatrix}
  R & t \\
  0^T & 1
\end{bmatrix} \in \mathbb{R}^{4\times 4} | R\in SO(3), t\in \mathbb{R}^3}
$$


对于群的解释:
* 首先定义 **封闭**
  * 旋转矩阵和变换矩阵对加法是不封闭的, 即两个旋转矩阵 R1, R2的加法, 结果不再是一个旋转矩阵
  * 可以称为这种矩阵的集合没有良好定义的加法, 或者称 矩阵加法对这两个集合不封闭
* 对应的:
  * 旋转矩阵和变换矩阵对于乘法是封闭的
  * 两个矩阵相乘代表进行了两次连续的 旋转/欧式变换
* 对于只有 一个运算的集合 , 称之为 群


### 3.1.1. Group, Lie Group - 群, 李群

群本质上被定义为 一种集合加上一种运算的 代数结构, 令 集合记为 A, 运算记为 点 $\cdot$, 那么群可以表示为 $G=(A,\cdot)$

群所定义的运算需要满足
* 封闭性 : $\forall a_1, a_2 \in A, a_1\cdot a_2 \in A$
* 结合律 : $\forall a_1,a_2,a_3 \in A, (a_1\cdot a_2) \cdot a_3 = a_1 \cdot(a_2\cdot a_3)$
* 幺元 : $\exist a_0 \in A, s.t. \forall a \in A, a_0\cdot a = a\cdot a_0 =a$
* 逆 : $\forall a \in A, \exist a^{-1}\in A, s.t. a\cdot a^{-1}=a_0$

四大规则可以记为  封结幺逆, 其他常见的群有:
* 整数加法群 $(\mathbb{Z},+)$
* 去掉 0 的有理数乘法
* 一般线性群 $GL(n)$, 指的是 $n\times n$ 的可逆矩阵
* 特殊正交群 $SO(n)$, 旋转矩阵群, 以 2 和 3 最为常见
* 特殊欧式群 $SE(n)$, n 维欧氏变换的群, 也是 2 和 3 最为常见

所谓的李群: 指的是具有连续(光滑)性质的群
* 整数群因为是离散的, 所以不是李群
* 空间中刚体的旋转很容易想象到是连续的, 所以SO 和 SE 是李群
* 每一个李群都会对应一个李代数, 例如 $\mathfrak{so}(3)$

### 3.1.2. Introduction of the Lie Algebra - 李代数的引出


李代数

首先对于旋转矩阵 R , 即 SO(3):
$$
RR^T=I
$$

现在对于某一个旋转的相机, 描述它 随时间的变换, 即为时间的函数 $R(t)$. 由于每一时刻都是旋转矩阵, 因此  
$$R(t)R(t)^T=I$$

在等式两边对时间求导, 得到 
$$
\dot{R}(t)R(t)^T + R(t)\dot{R}(t)^T = 0
$$

整理得, 右项移动到等式右边并提取一个转置    
$$
\dot{R}(t)R(t)^T = -(\dot{R}(t)R(t)^T)^T
$$

从如上等式可以看出来, 所谓的 $\dot{R}(t)R(t)^T$ 其实是一个 反对称矩阵.  
返回到原本对于符号 $\hat{}$ 的定义, 是把一个向量唯一的转换成一个反对称 矩阵, 同样的, 把一个反对称矩阵唯一的变换为向量的操作也存在, 称为 $\check{}$  

此时有 $a\hat{}=A, A\check{}= a$

结合上面的求导结果, 有 $\dot{R}(t)R(t)^T  = \phi(t)\hat{}$ , 这里用 $\phi(t)\hat{}$ 来标志旋转矩阵的 `一阶导乘以其转置`, 
等式两边右乘以 $R(t)$, 由于 R(t) 是正交矩阵, 因此有  

$$\dot{R}(t)=\phi(t)\hat{}R(t) = 
\begin{bmatrix}
  0 & -\phi_3 & \phi_2 \\
  \phi_3 & 0 & -\phi_1\\
  -\phi_2 & \phi_1 & 0
\end{bmatrix}R(t)$$

观察如上的结果, 可以发现, 对于旋转矩阵的求导, 只需要左乘一个 $\phi(t)\hat{}$ 矩阵即可.  

基于以上结果, 考虑 $t_0=0$ 的时候, 设此时旋转矩阵为 $R(0)=I$ 按照导数定义, 可以把 $R(t)$ 在 $t=0$ 附近进行一阶泰勒展开:
$$R(t)\approx R(t_0)+\dot{R}(t_0)(t-t_0)=I+\phi(t_0)\hat{}(t)$$

从上述公式可以发现 $\phi(t)\hat{}$ 有R的导数性质
* 在这里称 它在 SO(3) 的原点附近的正切空间 ( Tangent Space) 上
* 在 $t_0$ 的附近, 假设 $\phi$ 保持为常数, $\phi(t_0)=\phi_0$, 

可以列出微分方程 , 且有初始值 $R(0)=I$
$$\dot{R}(t)=\phi(t_0)\hat{}R(t)=\phi_0\hat{}R(t)$$

得, $R(t)=exp(\phi_0\hat{}t)$

到此, 仍然意义不明, 只是知道了
* 任意一个旋转矩阵 R 与另一个 反对称矩阵 $\phi_0\hat{}t$ 是通过指数关系联系起来的  
* 给定了旋转矩阵 R, 即可直到对应的 $\phi$, 描述了 R 在局部的导数关系

给出李代数的定义
* 称 $\phi$ 是对应了旋转矩阵 $SO(3)$ 上的李代数 $\mathfrak{so}(3)$
* 给定某个向量 $\phi$ , 对应的矩阵指数 $exp(\phi\hat{})$ 是如何计算的, 反之给定 R 来反计算 $\phi$, 这就是 李代数与李群之间的 指数/对数映射  


### 3.1.3. The Definition of Lie Algebra - 李代数的定义

**对于每个李群都有其对应的李代数, 李代数描述了李群的局部性质, 或者详细的说, 正切空间(tangent space)**

李代数由, 集合 $\mathbb{V}$, 数域 $\mathbb{F}$, 二元运算 $[,]$ 构成, 称李代数 $\mathfrak{g}$ 为 $(\mathbb{V,F,[,]})$ , 需要的性质有
* 封闭性 $\forall X,Y \in \mathbb{V}, [X,Y]\in \mathbb{V}$  二元运算后的结果仍然在集合内
* 双线性 $\forall X,Y,Z \in \mathbb{V}, a,b \in \mathbb{F}$ 满足
  * $[aX+bY,Z]=a[X,Z]+b[Y,Z]$
  * $[Z,aX+bY]=a[Z,X]+b[Z,Y]$
  * 有点类似于 某种乘法结合律
* 自反性 $\forall X \in \mathbb{V}, [X,X]=0$  自己与自己运算的结果为 0 
* 雅可比等价
  * $\forall X,Y,Z \in \mathbb{V}, [X,[Y,Z]]+[Z,[X,Y]]+[Y,[Z,X]]=0$

一个典型的例子是 三维向量 $\mathbb{R^3}$ 上定义的 叉积 $\times$ 是一个李括号, 构成了李代数 $\mathfrak{g} = (\mathbb{R^3,R,\times})$

### 3.1.4. Lie Algebra 李代数 $\mathfrak{so(3)}$

由上上一章提到的 $R(t)=exp(\phi_0\hat{}t)$  

这里 $\phi$ 即为李代数 $\mathfrak{so(3)}$ , 由于向量与反对称矩阵一一对应, 因此也会称 $\mathfrak{so(3)}$ 的元素是 三维反对称矩阵, 不加区别  

$$\mathfrak{so(3)}={\phi \in \mathbb{R^3}, \Phi=\phi\hat{} \in \mathbb{R^{3\times 3}}}$$

且定义 $\mathfrak{so(3)}$ 中, 两个向量的李括号为 
$$[\phi_1,\phi_2] = (\Phi_1\Phi_2 - \Phi_2\Phi_1)\check{}$$


### 3.1.5. Lie Algebra 李代数 $\mathfrak{se(3)}$

对于 $\mathbb{SE(3)}$ 也有对应的李代数.  其与 $\mathfrak{so(3)}$ 相似  

$$
\mathfrak{se(3)}=\begin{Bmatrix}
\xi=\begin{bmatrix}
  \rho \\ \phi
\end{bmatrix}
\in \mathbb{R^6}, \rho \in \mathbb{R^3}, \phi \in \mathfrak{se(3)}, \rho\hat{}=\begin{bmatrix}
  \phi\hat{} & \rho \\ 0^T& 0
\end{bmatrix}
\in \mathbb{R^{4\times 4}} 
\end{Bmatrix}
$$

拆分的讲:
* 李代数 $\xi$ 是一个 六维向量
* $\rho$ 为 $\xi$ 的前三维, 且含义与 变换矩阵的平移并不相同
* $\phi$ 即为 $\mathfrak{so(3)}$ 的元素
* 对于李代数 $\mathfrak{se(3)}$, 反对称矩阵的符号 $\hat{}$ 的意义也进行了拓展, 但仍然保留了 向量到矩阵的 意义, 且依旧保持一一对应
* 简单的理解为 平移加上 $\mathfrak{so(3)}$, 要留意这里是不同意义上的平移

定义对应的李括号为 
$$[\xi_1,\xi_2] = (\xi_1\hat{}\xi_2\hat{} - \xi_2\hat{}\xi_1\hat{})\check{}$$

## 3.2. Exponential and Logarithmic Mapping - 李代数的指数与对数映射

111


# 4. Cameras and Images

了解
* 针孔相机的模型, 内部参数, 径向畸变参数
* 空间点到成像平面的投影过程
* OpenCV的基础图像存储和表达方法
* 基本的摄像头标定

## 4.1. Pinhole Camera Models

针孔模型和畸变是描述一个投影过程最基本的模型, 二者共同构成了相机的基本 intrinsic 参数  

总结 单目相机的成像过程:
1. 世界坐标系下的点P, 坐标为 $P_w$
2. 相机在运动, 因此基于相机的外部参数 (R,t or $T\in \text{SE}(3)$) 将世界的点P转为相机坐标 $\tilde{P_c}=RP_w+t$
3. 相机坐标下的点 $\tilde{P_c}=[X,Y,Z]^T$, 将其投影到归一化平面上 normalized plane, 得到归一化坐标 $P_c=[X/Z,Y/Z,1]^T$
4. 根据畸变公式得到畸变后的归一化平面坐标
5. 最终将归一化坐标乘以 内参, 即可得到最终的成像坐标 $P_{uv}=KP_c$

### 4.1.1. Pinhole Camera Germotry

通常的, 以相机光心为坐标系原点, 建立 O-x-y-z 坐标系
* 习惯上 z 指向相机前方, x 向右, y 向下
* 成像平面物理上在相机的后方, 通过基本的相似三角形可以得到图像的倒影
* 实际上, 从相机输出的图像并不是倒像, 因此为了方便理解, 通常在建模时成像平面都已经对称的放到了相机前方, 这样在公式上就消去了 负号

基本的成像公式
$$\frac{Z}{f}=\frac{X}{X'}=\frac{Y}{Y'}$$  
整理后则有 $X'=f\frac{X}{Z}, Y'=f\frac{Y}{Z}$  

通常对相机进行物理描述的时候, 点的位置以及焦距的单位都是米 meter, 在量化到图像像素坐标的时候, 需要引入 缩放 zoom 和 原点平移 translation 的概念.

在软件建模中, 通常图像平面 u,v轴上的缩放倍数 $\alpha, \beta$ 会与焦距结合, 称为 $f_x =f\alpha, f_y = f\beta$. f 焦距的一般单位是 米, 因此 alpha, beta 的单位为 (像素/米).  
而 u,v 轴的原点是图像的左上角, 因此通常投影后需要对像素点进行平移, 将相机原点投影出的坐标移动到图像平面的中心, 需要对 u,v 各自加上 $[c_x,c_y]$


在最基本的投影中:  
$$
\begin{equation*}
  \begin{pmatrix}
    u \\ 
    v \\
    1 \\
  \end{pmatrix}

=
  \frac{1}{Z}
  \begin{pmatrix}
    f_x,0,c_x\\
    0, f_y, c_y\\
    0,0,1
  \end{pmatrix}
  \begin{pmatrix}
    X\\ Y\\ Z
  \end{pmatrix}
\end{equation*}
$$

习惯性会把 Z 移动到等式的左侧  

$$
Z\begin{pmatrix}u\\v\\1 \end{pmatrix}

=
\begin{pmatrix}
  f_x,0,c_x\\
  0, f_y, c_y\\
  0,0,1
\end{pmatrix}
\begin{pmatrix}
  X\\ Y \\ Z \end{pmatrix}
\underset{=}{\text{def}} KP $$

该最终式子中, K 称为相机内参数矩阵 K, K 在相机的使用中不会发生改变, 出厂即确定的.  
P 即要投影的点在相机坐标下的坐标, 它是由点在原本世界坐标系的位置 $P_w$ 利用相机本身在世界坐标系下的位置进行变换的来的.  
相机在世界坐标系下的位置和朝向(位姿)被称为外部参数, 位姿是由旋转矩阵 $R$ 和平移向量 $t$ 来描述的

综合下来, 完整的针孔相机投影过程应该描述为:   
$$
ZP_{uv} = Z
\begin{bmatrix}
  u\\v\\1
\end{bmatrix}

=
K(RP_w+t)=KTP_w
$$
在上述的式子中, 隐藏了一次齐次坐标 (homogeneous) 的转换 (计算 $TP_w$ 的时候需要将坐标转换成齐次坐标, 在转为非齐次坐标与 K 相乘)

此外, 在转化中需要对计算出的相机屏幕坐标进行归一化, 因此  
$(RP_w+t) = [X,Y,Z]^T -> [X/Z, Y/Z, 1]^T$  
并不会影响最终的投影坐标 (u,v), 即在实际计算中, 可以考虑先将世界坐标系投影到 归一化平面 `normalized plane`, 即 Z=1 的假想平面, 再直接乘以 内部内参数 K 就可直接得到成像坐标 (u,v)

同时也证明了单目相机无法获取图像的深度信息

### 4.1.2. Distortion

通常情况下, 为了提高相机的可视角度 (get a larger FoV, Field-of-View), 通常会在相机的前方加一个 len, 即透镜.
* 而由透镜对光线行动路线的影响
* 透镜本身在物理组装的时候无法与成像平面保持绝对的平行
导致最终成像效果的畸变 Distortion

由透镜形状导致的畸变, 会使得真实环境中的直线在图像中变成了曲线, 越靠边缘越明显, 这种畸变通常径向对称, 称为径向畸变 radial distortion, 细分为两类
* 桶形畸变 barrel-like distortion
* 枕形畸变 cushion-like distortion  
而由于物理上透镜和成像平面的非平行导致的畸变称为 切向畸变 `tangential distortion`  

对于畸变的数学建模使用 归一化平面 normalized plane 上的坐 标 $p$, 其坐标为 $[x,y]^T$.  
将其坐标转化为极坐标的形式 $[r,\theta]^T$, 这里 r 是点到 归一化平面原点的距离, 而 theta 表示该点与水平面的夹角. 有
* 径向畸变是改变了点到原点的距离, 即 $r$
* 切向畸变是改变了点与水平面的夹角, 即 $\theta$

一般上, 通过多项式来对径向畸变进行建模. 根据畸变的复杂度, 有时候不一定需要用到 k3  
$$
x_{distorted} = x(1+k_1r^2+k_2r^4+k_3r^6) \\
y_{distorted} = y(1+k_1r^2+k_2r^4+k_3r^6) 
$$

而切向畸变使用另外一组参数 $p_1, p_2$.  公式基本上是对称的
$$
x_{distorted} = x+2p_1xy + p_2(r^2+2x^2) \\
y_{distorted} = y+p_1(r^2+2y^2)+2p_2xy 
$$


统合两个畸变的公式, 同时应用, 将 切向畸变里的 x,y 替换成径向畸变的公式即可.  
$$
x_{distorted} = x(1+k_1r^2+k_2r^4+k_3r^6)+2p_1xy + p_2(r^2+2x^2) \\
y_{distorted} = y(1+k_1r^2+k_2r^4+k_3r^6)+p_1(r^2+2y^2)+2p_2xy 
$$

畸变矫正后的归一化平面上的点, 直接乘以内参矩阵就可以得到正确的投影位置.  

### 4.1.3. Stereo Cameras
<!-- 完 -->
单目摄像机无法获取深度, 这是因为 : 从光心到归一化平面连线上的所有的点都可以投影至该像素上.  
只有当某个像素点的深度确定时, 才能知道其空间位置  

主流双目摄像机都是由左右两个水平放置的相机构成, 理论上也可以做成上下.

由于是水平放置的, 因此在建模时会假设两个相机的光圈中心都位于 x 轴上.  两个相机光心的距离称为 双目摄像机的 baseline (记作b ), 是双目摄像机的一个非常重要的参数.  

假设一个空间点 P 投影到双目相机的图像平面, 并且假定这是一个理想的双目摄像机, 两眼的成像点坐标只在 x 轴上有差异.  那么, 记该点在左眼的 x 坐标为 $u_L$, 右眼为 $u_R$. 根据简单的相似三角形定理, 有 

* z 为空间点 P 的深度
* f 为焦距

$$\frac{z-f}{z}= \frac{b-u_L+u_R}{b}$$

整理有
* 定义视差 $d= u_L - u_R$
* $z=\frac{fb}{d}$

由此可以总结出一些双目相机的最基本的性质
* 距离和时差成反比, 距离越远视差越小
* 由于图像中时差最小为1个像素, 因此双目相机的测距存在理论上的最远距离. 即 $fb$
* 因此对于小型双目相机, 由于很难将基线做大, 因此很难实现远距离高精度测距


在该模型中, 省略了一个对人类容易, 但是对计算机难的问题, 视差计算中 : 如何确定左眼的某个像素出现在右眼的哪一个像素 (对应关系)


### 4.1.4. RGB-D camera
<!-- 完 -->
相对与 双目相机通过计算被动获取深度, RGB-D 相机的做法更加的主动, 目前根据原理对 RGB-D 相机进行分类有.  
* 红外结构光 (Structured Light)
* 飞行时间 (Time-of-Flight, ToF)
其实这两种方法都是需要相机主动的发射光线, 通常是红外光. 

限制
* 使用红外光的 RGB-D 相机, 容易受到日光或者其他红外线传感器的干扰, 不能在室外使用
* 没有调制的话, 同时使用多个 RGB-D 相机会互相干扰
* 对于有投射材质的物体, 因为无法反射光, 因此无法测量深度


## 4.2. 图像 - Image

二维数组表示图像的时候, 注意 y,x 坐标顺序

在表示深度的时候一般会用16进制数据表示, 换算成 mm 的话最多可以表示 65 米

# 5. Nonlinear Optimization - 非线性优化方法



# 6. Visual Odometry - 视觉里程计 Part 1

基于视觉的里程计构建, 两种主要方法
* 特征点法 feature methods: 
  * 特征点, 提取以及匹配
  * 根据配对的特征点估计相机运动
  * 是主流方法, 稳定, 对光照, 动态物体不敏感
* 光流法 direct methods

关键字:
* 对极几何, 对极几何的约束  `epipolar geometry` `epipolar constraints` : 主要用于单目相机的两组 2D 点估计运动
* PNP 问题, 利用已知的三维结构与图像的对应关系求解相机的三位运动  : 一组为 3D, 一组为 2D. 
* ICP 问题, 点云的匹配关系求解摄像机三位运动   : 主要用于 双目和 RGB-D, 即在某种方法得到了双目信息后, 根据两组 3D 点估计运动
* 通过三角化获得二维图像上对应点的三维结构


## 6.1. Feature Method - 特征点法

SLAM 系统分为前端和后端, 而前端指的就是 VO visual odometry 视觉里程计, 根据相邻图像的信息估算出粗略的相机移动, 给后端提供较为友好的初始值.  

从两帧之间的差异估计相机的运动和场景集合, 实现一个两帧的VO, 也成为 两视图几何 (Two-view geometry)  

### 6.1.1. Feature Point

在视觉 SLAM 中, 寻找多帧图像中比较有代表性的点, 这些点在相机视角发生改变后能够保持不变.  这些点在 SLAM 中称为 landmarks, 而在 CV 领域则称为 image feature.

图像像素值-灰度值 本身也可以理解为一种最原始的特征, 然而:
* 受光照, 形变, 物体材质的影响严重
* 不同图像之间变化大, 不稳定

2000年以前提出的特征大部分都是基于 角点 (corner point) 的算法, 例如 Harris, Fast, GFTT. 然而单纯的角点仍然不够稳定, 远处看上去是角点的地方离近了后就不再是角点.  

经过很多年的发展, CV 领域提出了很多新的人工设计的 局部图像特征算法, 例如 `SIFT, SURF, ORB`, 这些新颖的算法的优点有
* Repeatability 可重复性 : 能够在不同的图像中找到
* Distinctiveness 可区别性 : 不同的特征的表达是不同的
* Efficiency 高效性 : 特征点的数量应该远远小于像素的数量
* Locality 本地性 : 特征仅与一小片图像区域相关  


目前 SLAM 系统中, ORB 特征是应用最广的一种, 权衡了鲁棒性和计算速度. 特征的发展可以简述为.

SIFT, Scale-Invariant Feature Transform (尺度不变特征变换)  是最经典的一种图像特征:
* 充分考虑了图像变换中的 光照, 尺寸, 旋转等变化
* 计算量极大
* 截至 2016 年, 普通的计算机仍然无法实时的计算 SIFT 特征. 
* 对于目前的 SLAM 系统设计过于奢侈

ORB, Oriented FAST and Rotated BRIEF.  
* 基于 FAST 关键点检测 (原本的 FAST 算法只有关键点检测, 没有描述子 descriptor )
* 改进了 FAST 不具有方向性的问题
* 使用了计算速度极快的 BRIEF (Binary Robust Independent Elementary Feature) 二进制描述子
* 在同一幅图像中计算 1000个特征点的情况下 ORB (15.3ms) < SURF(217.3ms)  < SIFT (5228.7ms) 

### 6.1.2. ORB Feature

该章详细介绍了 ORB 特征的计算方法

### 6.1.3. Feature Matching
<!-- 完 -->

对于有大量相似纹理的场景, 基于局部的特征很难 真正有效的避免误匹配  


基础的匹配方法, 针对从两个图象提取的特征点 $x_t^m, m=1,2,...,M$ 和 $x_{t+1}^n,n=1,2,...,N$  寻找这两个集合元素的对应关系
* 最简单的方法就是暴力匹配 (Brute-Force Matcher), 所有点两两比较距离
* 另一种方法是快速最近邻 FLANN, Fast Approximate Nearest Neighbor 算法
* 对于距离的计算
  * 浮点数的特征描述, 可以利用欧氏距离来度量
  * 对于二进制的特征描述, 可以使用汉明距离 `Hamming distance`, 指的是两个二进制串之间的 不同位数的个数


<!-- Practice: Feature Extraction and Matching 实践章节跳过 -->

## 6.2. 2D-2D : Epiploar Geometry - 对极几何

主要针对两组 2D 点的运动估计

### 6.2.1. Epipolar Constraints - 对极约束
<!-- 完 -->
对于两张图片中完成匹配的特征点, 假设有若干个.  (具体的数目后面讨论)  

首先讨论特征点的几何关系  

对于两个时刻的相机影像 $I_1, I_2$, 定义相机$O_1, O_2$的位置关系 1->2 为 $R,t$  

对于正确匹配的在 $I_1, I_2$ 上的一对特征点 $p_1, p_2$, 连线 $O_1p_1, O_2p_2$在空间中相交于点 $P$, 则定义对极几何:
* Epipolar plane 极平面: $O_1,O_2,P$ 三点确定的平面 
* Epipolar 极点 : $O_1O_2$ 连线 与两个图像平面的交点  $e_1, e_2$ 
* Baseline 基线 : $O_1O_2$ 连线
* Epipolar line 极线 : $p_1e_1, p_2e_2$ 极平面与两个图像平面的交线, 也定义为 $l_1, l_2$

从想象上
* 从第一帧的视角 看 $O_1p_1$ 空间连线, P 的位置为该射线上的任意一点
* 从第二帧的视角看, P 的投影位置应该出现在 $\vec{e_2p_2}$ 射线上
* 由于 正确的特征匹配, 得知了 $p_2$ 的像素位置, 因此确定了 P 的空间位置以及相机的运动  
* 因此对于 对极几何来说, 正确的特征匹配的权重非常大

对极约束: $x_1, x_2$ 是两个像素点的归一化平面上的坐标 (返回之前的章节确认归一化平面的定义, 是一个三维点)
$$p_2^TK^{-T}t\hat{\space}RK^{-1}p_1 = 0$$  
$$x_2^Tt\hat{\space} Rx_1=0$$  

形式及其简洁, 对极约束本身包括了平移和旋转. 中间的部分可以分别基座两个矩阵
* Essential Matrix    本质矩阵  : $E = t\hat{\space} R$
* Fundamental Matrix  基础矩阵  : $F = K^{-T}EK^{-1}$
* 带入 E,F 整理更简洁的 式子有 
$$x_2^TEx_1= p_2^TFp_1=0$$

对极约束简单的描述了匹配点的空间位置关系, 因此相机的位姿估计问题变为了以下两步
1. 根据匹配点的像素位置算出 E 或者 F
2. 根据 E 或者 F 求出 R,t
由于 E 和 F 只相差了相机内参, 内参在 SLAM 任务中基本上是已知的 (在SfM任务中有可能是未知且有待估计的), 因此使用形式更加简洁的 $x_2^TEx_1=0$ 比较频繁

以下是推导过程:
* 对于空间点 $P=[X,Y,Z]^T$
* 根据针孔模型知道两个像素点 $p_1,p_2$ 的像素位置为 $s_1p_1=P, s_2p_2=K(RP+t)$
* s 为缩放尺度, 在使用齐次坐标的时候, 有无缩放的坐标都是等价的, 称为 equal up to a scale (尺度意义下相等) $sp\simeq p$
* 重写投影关系 $p_1\simeq P, p_2\simeq K(RP+t)$
* 反投影取得归一化平面坐标 $x_1=K^{-1}p_1 , x_2=K^{-1}p_2$
* 带入有 $x_2 \simeq Rx_1+t$
* 通过线性代数相关知识 (※需要复习)
  * 推导关键: 两边同时左乘 $t\hat{\space}$ 回忆 \hat 的定义, 这相当于两侧同时与 t 作外积 $t\hat{\space}x_2\simeq t\hat{\space}Rx_1$
  * 推导关键: 两边同时左乘 $x_2^T, x_2^Tt\hat{\space}x_2\simeq x_2^Tt\hat{\space}Rx_1$
  * 计算: 由于 $t\hat{\space}x_2$ 是一个与 t 和 x2 都垂直的向量, 因此它再和 x2 做内积的话, 结果为 0
* 等式左侧严格为 0, 则可以消去 尺度意义下相等, 记为普通等式, 就有了对极约束的式子 $x_2^Tt\hat{\space} Rx_1=0$

### 6.2.2. Essential Matrix - 本质矩阵

通过3X3矩阵本身的性质 : 8个自由度, 使用 8对匹配的点来算出 E

### 6.2.3. Homography Matrix - 单应矩阵

单目下无法对应纯旋转 (t=0 的运动), 考虑平面的 Homography 来解决

<!-- Practice: Solving Camera Motion with Epipolar Constraints -->



## 6.3. Triangulation - 三角测量 (三角化)

<!-- 完 -->
通过上一讲对极几何约束估计了相机运动之后, 需要用相机的运动 来估计特征点的空间位置, 此时对于单目相机来说, 仅通过单张图像无法获取深度信息, 因此需要通过三角测量 Triangulation 的方法来估计图像点的深度.  

三角测量: 通过不同位置 对同一个路标点进行观察, 从观察到的位置推断路标点的距离  

考虑图像 I1, I2. 以左图为参考, 且右图的变换矩阵为 T, 相机光心分别为 O1, O2.  
在 I1 中有特征点 p1, I2 中有特征点 p2.  

理论上, 直线 O1p1 与 O2p2 会在实际场景中相交于点 P. 
该点即两个特征点所对应的地图点在三维场景中的位置, 然而由于噪声的影响, 两条直线往往无法相交. 因此实际求解可以转为一个最小二乘法的问题.  

首先给出对极几何的定义: 
$$x_2 \simeq Rx_1+t$$
重新考虑尺度因子
$$s_2x_2 = s_1Rx_1+t$$

特征点的深度其实也就是 $s_1,s_2$ 那么通过对极几何约束得到了移动 $R,t$ 后求解  $s_1,s_2$, 可以:
* 在射线 $O_1p_1$ 上寻找 3D 点, 使得其投影位置接近 $p_2$
* 同理可以反过来寻找
* 甚至可以在两条线的中间找

以方法1, 在射线 $O_1p_1$ 上寻找 3D 点 为例, 希望计算 $s_1$, 那么对式子左乘 $x_2\hat{\space}$

$$x_2\hat{\space}s_2x_2 =0= x_2\hat{\space}s_1Rx_1+t$$
式子左侧为0, 则右侧可以看作方程, 即可直接求解 $s_2$ 并以此得到 $s_1$, 得到了两帧下的深度, 则可以确定空间坐标  
一般的, 由于噪声的存在, 不精确的 $R,t$ 会导致上述式子 不为0 , 因此常见的作法是使用最小二乘法求解  


## 6.4. 3D-2D : PnP 

Perspective-n-Point  求解 3D 到 2D 的点对运动的方法

当知道了 n 个 3D 空间点以及其对应投影位置的时候, 如何估计相机位姿  

在前面的章节中, 提到了 
* 2D-2D 的对极几何方法需要 8个 或者 8个以上的点对 (以 8 点法为例), 而且还需要考虑 初始化, 纯旋转和尺度的问题
* 如果 两张图象中的 1张, 其特征点的 3D 位置已知, 那么最少只需要 3 个点对 (以及一个验证点) 即可就可以估计相机运动, 这种情况下的点的 3D 位置通常由 三角化 或者 RGB-D 相机的深度图来确定
  * 因此在双目或者  RGB-D 的视觉里程计中, 可以直接使用 PnP 来估计相机运动
  * 在单目视觉里程计中, 必须先 初始化, 然后才能用 PnP
* 3D-2D 方法 不需要使用对极约束, 且可以在很少的匹配点中获得 较好的运动估计  是最重要的姿态估计方法

PnP 问题的解法其实有很多:  
* 3对点估计位姿:  P3P
* 直接线性变换  : DLT (Direct Linear Transformation)
* EPnP    : Efficient PnP
* UPnP
* 非线性优化  : 构建最小二乘问题并迭代求解, 也就是 Bundle Adjustment(BA) 光束平差法


### 6.4.1. Direct Linear Transformation 直接线性变换  

给定 3D 点的位置以及在某个相机中的投影位置, 求相机位姿  
如果 3D 点的位置是在另一个相机坐标系的话, 则求的是两个相机的相对运动  

考虑空间点 P, 有
* 齐次坐标 $P=(X,Y,Z,1)^T$
* 在图像的坐标 $x_1=(u_1, v_1, 1)^T$
* 此时相机位姿 R,t 是未知的, 此时展开形式满足  

$$
s\begin{pmatrix}
  u_1\\
  v_1\\
  1
\end{pmatrix}=
\begin{pmatrix}
  t_1  & t_2 & t_3  & t_4 \\
  t_5  & t_6 & t_7  & t_8 \\
  t_9  & t_{10} & t_{11}  & t_{12} \\
\end{pmatrix}
\begin{pmatrix}
  X\\
  Y\\
  Z\\
  1
\end{pmatrix}
$$

该等式可以通过前两行消去最后一行, 得到两个约束式  

$$
u_1 = \frac{t_1X+t_2Y+t_3Z+t_4}{t_9X+t_{10}Y+t_{11}Z+t_{12}} , v_1 = \frac{t_5X+t_6Y+t_7Z+t_8}{t_9X+t_{10}Y+t_{11}Z+t_{12}}
$$

此时可以定义 行向量并简化表达  
$$
\bold{t_1} = (t_1, t_2,t_3,t_4)^T, 
\bold{t_2} = (t_5 , t_6, t_7 , t_8)^T, 
\bold{t_3} = (t_9 , t_{10}, t_{11} , t_{12})^T \\

\bold{t_1^T}P-\bold{t_3^T}Pu_1=0, 
\bold{t_2^T}P-\bold{t_3^T}Pv_1=0, 
$$

3个t向量是待求的变量, 而每一个点提供了 两个关于 t 的线性约束, 假设一共有 N 个特征点, 则可以写成方程的形式  

$$
\begin{pmatrix}
  P^T_1 & 0 & -u_1P^T_1 \\ 
  0 & P^T_1 &  -u_1P^T_1 \\ 
  P^T_N & 0 & -u_NP^T_N \\ 
  0 & P^T_N &  -u_NP^T_N \\ 
\end{pmatrix}\begin{pmatrix}
  t_1 \\
  t_2 \\
  t_3 \\
\end{pmatrix}=0
$$

由于 t 的变量一共 12 维, 因此最少通过 6 对 匹配的特征点即可求解矩阵 T, 这种方法就叫做 DLT  

当匹配点多于 6 对的时候, 通过 SVD 等方法对超定方程求解最小二乘解   

这种解法是直接将 Rt 看作了 12 个未知数, 忽略了他们作为 SE3 的内在联系, 且 旋转矩阵属于 SO3, 而求解得到的旋转不一定满足该约束  

对于旋转矩阵 R, 需要针对 DLT 的结果寻找一个最好的 旋转矩阵对他进行近似, 可以由 QR分解完成, 或者  
$$R\larr (RR^T)^{-\frac{1}{2}} $$  

相当于把结果从 矩阵空间重新投影到 SE3 的形态, 转换成旋转和平移两部分   

在该章节中, x 直接使用了归一化的坐标, 在 SLAM 中内参通常都做假定为已知, 就算内参未知, 也可以用 PnP 直接去估计 K,R,t 三个量, 由于未知量增多, 效果会差一些  

### P3P


### Minimizing the Reprojection Error

可以把 PnP 问题构建成一个关于重投影误差的 `非线性最小二乘问题`  可以优化 PnP 或者 ICP

这一类把 相机和三维点 放在一起进行最小化的问题, 统称为 Bundle Adjustment


## 6.5. 3D-3D : ICP

对于一组已经匹配好的 3D 点, 例如通过 RGB-D 进行匹配, 希望求得 $R,t$  

设匹配点分别为 $P={p_1,\dots, p_n}, P'={p_1',\dots,p_n'}$
使得最终的 $R,t$ 满足
$$\forall i,P_i=Rp_i'+t$$

解决这个问题的方法就是 ICP (Iterative Closest Point) 迭代最近点, 在上述公式中并没有出现相机模型, 也就是说在 3D-3D 问题中和相机是没有关系的.  

PS: 在激光 Lidar 的 SLAM 问题中也会出现 ICP, 不过激光数据的特征不够丰富, 因此问题出现在匹配关系上. 而基于视觉的 SLAM 则不存在这种问题.  

ICP 的求解可以分为两种
* 基于线性代数的求解 SVD
* 基于非线性优化的求解 (类似于 BA)

### 6.5.1. SVD - Linear Algebra Method

### 6.5.2. Non-linear Optimization Method

<!-- 完 -->
使用非线性优化, 通过迭代来寻找最优解.  该方法和上述的 PnP 方法相似, 也使用到了 李代数.  

以李代数表达相机位姿的时候, 目标函数可以写成:  

$$\underset{\xi}{\min}=\frac{1}{2}\sum^n_{i=1}||(p_i-\exp(\xi\hat{\space})p'_i)||^2_2$$

带入 李代数扰动模型  (Lie algebra perturbation model): 
$$\frac{\partial e}{\partial \delta\xi}= -(\exp(\xi\hat{\space})p_i')^\odot$$

即通过迭代即可找到极小值
* 存在证明 : ICP 问题是 唯一解或者无穷多解的, 因此找到了极小值解相当于找到了全局最优解
* 因此 ICP 求解可以任意选定初始值

在匹配已知的情况下, ICP 问题实际上具有 `解析解`, 即无需通过迭代 而基于优化的 ICP 的存在意义是:
* 在某些场景, 例如 RGB-D SLAM, 一个像素的深度数据可能有, 也可能没有, 因此可以混合使用 PnP 和 ICP, 从而实现:
* 对深度已知的点, 3D-3D 的误差建模
* 对深度未知的特征点, 建模 3D-2D 的重投影误差
* 将所有的误差放在同一个问题中考虑, 使得求解更加方便

# 7. Visual Odometry - 视觉里程计 Part 2

* 理解 `光流法` 跟踪特征点的原理
* `直接法` 估计相机位姿的原理
* 多层直接计算法的实现

直接法是 视觉里程计的一个主流的分支, 与特征点法有很大的不同.  
是未来的潜力算法

## 7.1. The Motivation of the Direct Method - 直接法的引出
<!-- 完 -->
在 VO 的第一部分介绍了基于特征点来估计相机移动的方法, 尽管占据主流, 但研究者仍然认识到基于特征点算法的缺点:
* 关键点的提取以及关键点描述子的计算非常耗时, 截至书中截稿 SIFT 在CPU上无法实时计算, ORB 则需要 20ms. 如果整个系统是运行在 30ms/帧 约 30fps上, 则大部分时间都消耗在了计算特征点上
* 使用特征点时, 很可能会忽视掉其他有用的信息, 因为一幅图有上百万个像素, 然而特征点主流的只有几百个
* 当相机运行到特征缺失的地方时, 基于特征点的算法会因为特征点数量不足导致计算本深无法进行

从克服关键点的缺点的角度上考虑, 引出了两种思路
* Optical Flow (光流法) : 保留特征点, 但仅仅计算点的位置, 不进行特征描述. 使用光流法来跟踪特征点的运动, 这样可以跳过对描述子的计算和匹配的过程, 光流的计算复杂度本身要小于描述子的计算与匹配
  * 相当于把匹配描述子替换成了光流跟踪, 在计算相机运动的时候仍然使用 对极几何, PnP, ICP 算法
  * 仍然需要特区到的关键点具有可区别性, 需要角点
* Direct Method (直接法) : 但是使用直接法来计算特征点在下一时刻的运动, 连光流的计算都省去了
  * 根据图像的像素信息同时估计相机运动和点的投影

在特征法中, 主要根据特征点在三维中的不动性, 通过最小化 **重投影误差** (Reprojection error) 来优化相机运动.  
在直接法中, 不需要知道点与点之间的对应关系, 而是通过最小化 **光度误差** (Photometric error) 来求得  

该大章节关注 直接法, 根据像素的亮度信息直接求出相机运动, 只要场景中存在明暗变化. 根据使用的像素数量, 可以分为
* 稀疏
* 稠密
* 半稠密
与特征点法只能重构系数特征点 (稀疏地图) 相比, 直接法拥有构建稠密图的能力

使用直接法的近期的主流方法有 : SVO, LSD-SLAM, DSO 等

## 7.2. 2D Optical Flow - 2D 光流
<!-- 完 -->

直接法是从光流法演变而来的, 因此很相似, 在直接法之前先学习 光流法  
* 光流法描述了像素在图像中的运动
* 直接法在运动的基础上附带着相机运动模型  

光流 用于描述 pixels 在图像之间的运动, 即追踪像素点.
* 计算部分像素运动称为 Sparse Optical Flow (稀疏光流)
  * 在实际中, 主要用来追踪 特征点
  * 以 `Lucas-Kanade` (LK flow, LK光流) 为代表
* 计算全部像素则为 Dense Optical Flow (稠密光流)
  * 以 Horn-Schunck 光流 为代表



### 7.2.1. Lucas-Kanade 光流 

LK 光流中, 假设 相机的图像随时间变化, 图像看作时间的函数  $I(t)$ 具体到像素 则有 $I(x,y,t)$

光流估计的最基本的假设: 灰度不变假设  同一个空间点的像素灰度值, 在各个图像中是固定不变的.  
(这是一个很强的假设, 现实中往往不成立, 物体的材质, 高光和阴影, 相机的自动曝光参数等, 光流法就是基于如此的强假设的算法)

对于像素的移动, 有  $I(x+dx, y+dy, t+dt) = I(x,y,t)$, 左边进行一阶泰勒展开  
$\approx I(x,y,t) + \frac{\partial I}{\partial x}dx + \frac{\partial I}{\partial y}dy + \frac{\partial I}{\partial t}dt$

此时应用灰度不变假设, 从而 $\frac{\partial I}{\partial x}dx + \frac{\partial I}{\partial y}dy + \frac{\partial I}{\partial t}dt =0$

两边除以 dt, 得

$$\frac{\partial I}{\partial x}\frac{dx}{dt} + \frac{\partial I}{\partial y}\frac{dy}{dt} = - \frac{\partial I}{\partial t}$$

整理上式的各个元素
* $\frac{dx}{dt}, \frac{dx}{dt}$ 是像素在 x, y 轴上的运动速度, 可以记为 $u, v$
* $\frac{\partial I}{\partial x}, \frac{\partial I}{\partial x}$ 是像素在 x, y 方向上的梯度, 记为 $I_x, I_y$
* 图像灰度对时间的变化量为 $I_t$
写为矩阵乘法形式, 整理得 
$$
\begin{bmatrix}
  I_x & I_y
\end{bmatrix}
\begin{bmatrix}
  u \\ v
\end{bmatrix}
= -I_t
$$

这里要计算的是变量 u,v 然而目前的形式是 二元一次方程, 约束不够, 此时引入第二个约束

**假设某一个窗口内的像素具有相同的运动**  

此时假设计算窗口为 $\omega \times \omega$ 有像素下标 $k=i,...,\omega^2$
$$
A=\begin{bmatrix}
  [I_x,I_y]_1 \\
  ... \\
  [I_x,I_y]_{\omega^2}
\end{bmatrix},
b=
\begin{bmatrix}
  I_{t1} \\
  ... \\
  I_{t\omega^2}
\end{bmatrix},
A\begin{bmatrix}
  u \\
  v
\end{bmatrix}= -b 
$$

这是一个关于 u,v 的超定线性方程, 传统解法是求最小二乘解

$$
\begin{bmatrix}
  u \\ v
\end{bmatrix}^\ast = -(A^TA)^{-1} A^Tb
$$
注: 等式右边就是多元最小二乘法的通用解   
* 最小二乘法的优点: 简单易用
* 缺点: 对异常值敏感, 模型被假设为了线性


光流法小结:
* LK 光流能够直接得到特征点的对应关系, 类似于描述子的匹配
* 光流法对于图像的连续性和光照稳定性要求更高一些
* 可以避免计算和匹配 描述子的过程, 但要求相机运动平滑, 或者采集频率较高

### 7.2.2. 光流法实践

TODO




## 7.3. Direct Method 直接法
<!-- 完 -->

### 7.3.1. Derivation of the Direct Method - 直接法的推导
<!-- 完 -->
光流法的思路: 追踪特征点的位置 -> 根据位置确定相机的运动  (两部一走, 难以保证全局最优性 )

直接法 : 根据相机位置的运动 初始假设, 反过来调整光流的计算结果 

假设空间点 P $[X, Y, Z]$, 在两个时刻的相机上成像, 像素坐标为 $p_1, p_2$

求解第一个相机到第二个相机的相对位姿变换   R,t, 相机内参相同 为 K


特征点的方法中, 通过描述子和匹配知道了点 p 在两个成像上的坐标, 而在直接法中并没有这些情报

同样是基于灰度不变假设, 
要同时优化相机位姿以及 点p 的匹配信息, 可以通过求解一个优化问题来实现: 光度误差

同重投影误差不同, 光度误差测量的是两个像素的亮度误差

$$e=I_1(p_1)-I_2(p_2)$$

e是一个误差的标量, 取该误差的二范数, 不加权
$$\underset{T}{min}J(T) = ||e||^2$$  

假设有多个空间点 $P_i$ 的时候
$$\underset{T}{min}J(T) = \sum_{i=1}^N e^T_ie_i, e_i=I_1(p_{1,i}) - I_2(p_{2,i})$$  

注意, 直接法的优化对象是 相机位姿 T, 而不是特征点本身的运动, 因此在意的是误差 e 随相机 T 运动的变化关系, 即 e 和 T 的导数关系  


整理关系式
* $q=TP$ , q 为目标点在第二个相机下的坐标
* $u=\frac{1}{Z_2} Kq$ , u 为像素坐标

考虑李代数的左扰动模型 利用一阶泰勒展开
$$e(T) = I_1(p_1) - I_2(u)$$

$$
\frac{\partial e}{\partial T} = \frac{\partial I_2}{\partial u}\frac{\partial u}{\partial q} \frac{\partial q}{\partial \delta \xi} \delta \xi
$$

其中 $\delta \xi$为T 的左扰动, 一阶导数由于链式法则分成了三个部分, 而三个部分分别都是容易计算的 
* $\frac{\partial I_2}{\partial u}$ 为像素在 坐标 u 的梯度
* $\frac{\partial u}{\partial q}$ 投影方程关于相机坐标系下的 三维点 的导数, 记三维坐标为 $q=[X, Y, Z]^T$, 根据上一大章的推到方程, 此处的导数为  

$$
\frac{\partial u}{\partial q} = 
\begin{bmatrix}
  \frac{\partial u}{\partial X} & \frac{\partial u}{\partial Y} & \frac{\partial u}{\partial Z} \\
  \frac{\partial v}{\partial X} & \frac{\partial v}{\partial Y} & \frac{\partial v}{\partial Z}
\end{bmatrix}=
\begin{bmatrix}
  \frac{f_x}{Z} & 0 & -\frac{f_xX}{Z^2}\\
  0 & \frac{f_y}{Z} & -\frac{f_yY}{Z^2}\\
\end{bmatrix}
$$

* $\frac{\partial q}{\partial \delta \xi}$ 为 变换后的三位点对 变换本身 的导数, 这在 李代数的章节有推导 $\frac{\partial q}{\partial \delta \xi}=[I, -q\hat{\space}]$

可以看到后两项与图像本身没有关系, 因此可以合并

$$
\frac{\partial q}{\partial \delta \xi} = \begin{bmatrix}
    \frac{f_x}{Z} & 0 & -\frac{f_xX}{Z^2}& -\frac{f_xXY}{Z^2} & f_x+\frac{f_xX^2}{Z^2} & \frac{f_xY}{Z}\\
  0 & \frac{f_y}{Z} & -\frac{f_yY}{Z^2}  & -(f_y+\frac{f_yY^2}{Z^2}) & \frac{f_yXY}{Z^2} & \frac{f_yX}{Z}\\
\end{bmatrix}
$$

该 2 X 6 的矩阵在上一章也出现过, 最终 得到了 直接法的误差 相对于 李代数的雅可比矩阵  

$$J = -\frac{\partial I_2}{\partial u} \frac{\partial u}{\partial \delta \xi}$$

由此, 可以用这种方法计算优化 问题的 雅可比矩阵, 然后使用高斯牛顿法 (Gauss-Newton) 或者 列文伯格-马夸尔特方法 (Levenberg-Marquardt)  来计算增量, 迭代求导

### 7.3.2. 直接法的讨论
<!-- 完 -->

在直接法推导的一开始, 给定了 空间点 P 的坐标, 关于 P 的坐标生成
* 在 RGB-D 相机下, 可以直接将任意像素反投影到三维空间中
* 在双目相机系统中, 可以直接根据视差计算像素的深度  
* 在单目中, 该任务则变得尤为复杂, 因为存在深度不确定性


根据 P 的空间点的来源, 可以对直接法进行分类
* P 来自稀疏的关键点 : 稀疏直接法 (sparse direct method), 使用 数百个至上千个关键点, 并且像 L-K 光流那样, 假设周围像素也是不变的, 这种稀疏法不必直接计算描述子. 只使用数百个像素, 速度最快, 但是只能进行稀疏的重构
* P 来自部分像素 : 如果像素梯度为 0, 则整个雅可比矩阵就为0, 不会对计算有任何贡献, 因此在计算之初就过滤掉没有梯度的像素点, 称为半稠密直接法 (Semi-Dense) , 可以重构一个半稠密的结构
* P 为所有像素 : 计算量最大, 图片中的像素一般为几十万个, 多数不能在 CPU 上直接计算, 需要 GPU 加速. 尽管如此, 梯度为0 的区域仍然难以对优化有贡献

从稀疏到稠密重构, 都可以用直接法计算, 因此计算量也是可以控制逐渐增长的
* 稀疏方法可以快速地求解相机位姿
* 稠密方法可以建立完整地图


## 7.4. Practice Direct Method - 实践中的直接法


优缺点总结:
* 优点:
  * 省去了计算特征点, 描述子的时间
  * 只需要图像本身有像素梯度即可, 不需要特征点, 因此可以在特征缺失的场合下使用, 例如不存在角点, 只有渐变的场景
* 缺点
  * Non-convexity 非凸性 : 由于直接法完全依赖于梯度搜索, 降低目标函数来计算位姿. 因此优化算法很容易进入极小值, 只有在运动很小时直接法才能成功. 相对的, 引入 Pyramids 金字塔计算可以减少非凸性的影响
  * 单个像素很难有区分度 : 单个像素的区分度十分差, 因此往往以图像块为单位进行计算, 或者计算更加复杂的相关性. 直接法在选点较小的时候精度很差, 经验上使用 500+ 个点优先
  * Constant brightness is a strong assumption : 灰度不变是很强的假设, 在实际中由于相机自动曝光导致照度的改变, 图像整体的明暗度往往有很大差别导致算法失败, 实用中的直接法会同时估计相机的曝光参数.  



# 8. Filters and Optimization Approaches - 后端 Part1 


本章目标
* 理解够短的概念, 理解 后端优化的非线性性 
* 理解 use sparse structure in the bundle adjustment problem, 理解稀疏性是如何利用的
* 通过 g2o 和 ceres 来实际操作后端优化

通过建立一个 尺度 规模更大的优化问题, 解决 长时间内的最优轨迹和地图  

## 8.1. Introduction of backend
### 8.1.1. State Estimation from Probabilistic Perspective - 状态估计的概率解释  

<!-- 完 -->
SLAM 中通过最新的信息来更新较早以前的位置信息, 也可以理解为 当前的位置不仅仅由当前决定 还由未来决定  
* 当前的状态不仅仅由过去的信息更新, 也由未来的信息更新, 这成为 批量的 (Batch)
* 如果状态更新只由过去的时刻的信息决定, 则这称为 渐进的 (Incremental)

重新考虑运动和观测方程, 令 位姿为 $x$, 路标为 $y$, $N,M$ 分别为时间数和路标数    
$$
\begin{cases}
  x_k=f(x_{k-1},u_k)+w_k \\
  z_{k,j}=h(y_j,x_k)+v_{k,j}
\end{cases}
k=1,...,N,j=1,...,M
$$

$v,w$ 的意思是啥忘记了:  可能是噪声? 
* $u,z$ 分别表示运动数据(VO得到的运动差分量)和观测数据(路标观测到的在图像中的图像坐标)  
* $f$ 是运动方程
* $h$ 是观测方程, 即路标的三维点到图像像素坐标的投影过程

在SLAM问题中:
* 观测方程, 只有在位置 $x_k$ 看到了 路标 $y_j$ 的时候, 才会产生观测数据, 因此在一个位置上总是会只能看到一小部分路标. 且 SLAM 系统采集的特征点往往比较多, 因此特征方程的数量很巨大.  
* SfM 和 SLAM 问题的区别主要在对 图像和运动 的关系上
  * 假设在方程中没有运动方程, 则可以假设 相机不动或者匀速运动, 此时整个优化问题则只剩观测方程, 此时就很类似于 SfM 问题
  * SfM 即通过一组图像来恢复运动和结构, SfM 允许完全无时间关系的图像
  * SLAM 的图像有时间上的先后顺序  

概率解释: 实际中每个方程都受噪声影响, 一个普世的思想就是将所有噪声都考虑成高斯分布.  则在优化中只需要存储 均值和协方差矩阵即可, 均值可看作对变量最优值的估计, 协方差矩阵度量了变量的不确定性.  

在非线性优化的章节中介绍了 最大似然估计 (Maximum likelihood estimation), 提到了 基于批量的状态估计问题可以转化为最大似然估计问题, 并使用最小二乘法求解 (least-square method). 在SLAM的后端中, 将会讨论:
* 通过最小二乘法求解最大似然问题如何应用于 渐进式问题
* 在视觉 SLAM 中 最小二乘法的特殊结构

为了方便讨论, 在这里更改 运动观测方程的书写方法, 首先, 将 $x_k$ 的意义扩充, 表示 k 时刻的所有未知量, 包括 相机位置以及 m 个路标点(的位置?)
$$x_k \triangleq {x_k, y_1,...,y_m}$$

同时, 把 k 时刻的所有观测则记为 $z_k$, 则 运动与观测方程可以写成, 即消去了符号 $y$ .  

$$
\begin{cases}
  x_k=f(x_{k-1},u_k)+w_k \\
  z_k=h(x_k)+v_{k,j}
\end{cases}
k=1,...,N,j=1,...,M
$$

在最新的 k 时刻, 希望通过过去的数据来估计现在的状态分布: 
$$P(x_k|x_0,u_{1:k}, z_{1:k})$$  

此时按照贝叶斯法则, 交换 $z_k,x_k$ 的位置, 则有  
$$P(x_k|x_0, u_{1:k},z_{1:k})\propto P(z_k|x_k)P(x_k|x_0, u_{1:k},z_{1:k-1})$$

第一项为 似然 likelihood, 第二项为先验 prior 
* 似然可以由观测方程直接给定  
* 先验部分 则代表了 $x_k$ 会受过去所有的状态而影响. 而最起码, 会受到 $x_{k-1}$ 的影响  

则可以对先验部分 按照 $x_{k-1}$ 时刻为条件概率进行展开   
$$P(x_k|x_0, u_{1:k},z_{1:k-1})= \int  P(x_k|x_{k-1},x_0,u_{1:k},z_{1:k-1})P(x_{k-1}|x_0, u_{1:k},z_{1:k-1})dx_{k-1}$$

以上就是贝叶斯的估计的一次展开, 如果持续考虑更久之前的状态, 则持续对上式进行展开即可  

针对后续是否持续展开, 在 SLAM 问题中 分成了两个流派
* 1: SLAM 的主流为 非线性优化方法, 考虑 k 时刻状态与之前所有状态的关系  
* 2: 假设了马尔可夫性, 一阶马氏性认为, k时刻状态只与 k-1 时刻状态有关, 以 扩展卡尔曼滤波 (EKF) 为代表的滤波器方法, 只考虑 k-1 时刻的状态  

### 8.1.2. Linear Systems and the Kalman Filter - 线性系统与卡尔曼滤波KF

补充一些背景知识, 线性系统和卡尔曼滤波  
* 卡尔曼滤波有多种推导方式, 本节说明从概率角度出发的最大后验概率估计
* 卡尔曼滤波构成了该系统中的最大后验概率估计
* 卡尔曼滤波构成了线性系统的最优无偏估计


当假设了一个系统具有马尔可夫性的时候, 当前时刻的状态只与 上一个时刻有关, 因此在上一章节的最终数式可以进一步简化  

$$P(x_k|x_{k-1},x_0,u_{1:k},z_{1:k-1})=P(x|x_{k-1}, u_k)$$

即删除了所有比 k-1 更早之前的项目
此外, 上一节的第二部分 也可以进行简化

$$P(x_{k-1}|x_0, u_{1:k},z_{1:k-1})=P(x_{k-1}|x_0,u_{1:k-1},z_{1:k-1})$$

即 k 时刻的状态与 k-1 之前的无关, 简化成只与 $x_{k-1}, u_{k}$ 有关的形式  , 此外 k 时刻的输入量与 $k-1$ 时刻的状态无关.  

总的来说, 实际上在做的事就是 把 k-1 时刻的状态分布推导致 k 时刻, 程序运行期间, 只需要维护一个状态量并对其进行 迭代和更新即可.  
假设状态量服从高斯分布, 则只需要维护 状态量的均值和协方差即可.  

线性系统的定义: 运动方程和 观测方程可以用线性方程来描述  
$$
\begin{cases}
  x_k = A_kx_{k-1} + u_k+w_k \\
  z_k = C_kx_k+v_k
\end{cases} k = 1,...,N
$$

假设所有的状态 和噪声均满足高斯分布, 即 $w_k\sim N(0,R), v_k \sim N(0,Q)$

用 $\hat{x}$ 表示后验, 用 $\check{x}$ 表示先验分布  
假设已知k-1时刻的状态估计和协方差 $\hat{x_{k-1}}, \hat{P_{k-1}}$现在要根据 k 时刻的输入和观测数据, 确定k时刻的状态后验分布 $\hat{x_k}$ 

卡尔曼滤波的第一步, 先明确 运动方程确定 $x_k$ 的先验分布. 已经假设运动方程是线性的, 而高斯分布的线性变换仍然是高斯分布  

$$P(x_k|x_0,u_{1:k},z_{1:k-1})=N(A_k\hat{x_{k-1}}+u_k, A_k\hat{P_{k-1}}A^T_k+R)$$

这一步称为预测 predict, 显示了从上一个时刻的状态, 根据有噪声的输入推断当前时刻的状态分布, 这个分布也就是先验, 此时记: 可以明确随着状态的迁移噪声逐渐变大  

$$\check{x_k}=A_k\hat{x_{k-1}}+u_k, \check{P_k}=A_k\hat{P_{k-1}}A^T_k+R$$

根据观测方程, 可以计算在某个状态下的观测数据是啥样的 

$$P(z_k|x_k)=N(C_kx_k,Q)$$

此时为了计算后验概率, 希望得到乘积, 已知最终的结果是关于 $x_k$ 的高斯分布, 假设结果为 $x_k\sim N(\hat{x_k},\hat{P_k})$, 那么有  
$$N(\hat{x_k},\hat{P_k}) = \eta N(C_kx_k,Q)\cdot{ }N(\check{x_k},\check{P_k})$$


TODO: 推导太复杂了


卡尔曼滤波的最终表达, 预测和更新两个步骤
1. 预测 得到先验分布  

$$\check{x_k}=A_k\hat{x_{k-1}}+u_k, \check{P_k}=A_k\hat{P_{k-1}}A^T_k+R$$

2. 更新, 先计算 K, 又称为 卡尔曼增益, 然后计算后验概率的分布  

$$
K=\check{P_k}C^T_k(C_k\check{P_k}C^T_k+Q_k)^{-1} \\
\hat{x_k}=\check{x_k} +K(z_k-C_k\check{x_k})\\
\hat{P_k} = (I-KC_k)\check{P_k}
$$

### 8.1.3. Nonlinear Systems and the EKF - 非线性系统和EKF

尽管学习了 卡尔曼滤波 在线性系统的运用, 仍然需要明确 SLAM 中的运动方程和观测方程经常是 非线性函数, 尤其是 SLAM 中的相机位姿, 需要用相机内参数以及李代数.  
一个高斯分布, 经过非线性变换以后往往就不再是高斯分布, 因此在非线性系统中, 必须要取一定的近似.  

把卡尔曼滤波的结果拓展到非线性系统中, 称为 拓展卡尔曼滤波器.  

### 8.1.4. Discussion About KF and EKF

<!-- 完 -->

EKF 以形式简洁, 应用广泛著称.  当想要在某个时间段内估计某个不确定量时, EKF 是首选.  

非线性优化从精度上比 滤波器方法有明显的优势, 但在计算资源受限, 时间敏感, 或者待估计量比较简单的场景, EKF 仍然是有效的方法.  

EKF 的局限性:
* 滤波器方法假设了 状态变更的 马尔可夫性, k 时刻的状态只与 k-1 时刻相关, 与更早之前的观测和状态都无关.  而非线性优化则倾向于使用所有的历史数据, 称为 Full-SLAM, 使用了更多信息, 因此也需要更多的计算
* 与非线性优化相比, EKF 只在 $\hat{x_{k-1}}$ 处做了一次线性化, 就直接根据线性化的结果算出了 后验概率. 根据运动模型和 观测模型 的非线性情况, EKF 的非线性误差也会随之改变. EKF 相比与优化方法, 使用的范围较小, 可以粗略的认为 EKF 是优化中的一次迭代.  
* 程序实现上来说, EKF 需要存储状态量的 均值和 方差, 并进行更新和维护.  如果把路标也放进状态中, 则 当 SLAM 中路标数量非常庞大的时候, 这个存储的量是很恐怖的, 与 状态量成平方增长 (存储协方差矩阵). 普遍认为 EKF SLAM 不适合大场景下的 SLAM
* EKF 滤波器没有异常值检测的机制, 因此 在视觉 SLAM 这种由于特征匹配或者光流法导致的 错误匹配的点 会让整个系统的预测发散, 在实用中非常的不稳定.

综上所述, 笔者认为, 同等计算量的情况下, 非线性优化能够取得 精度和鲁棒性 综合起来更好的效果.  

本书中主要讨论 图优化 的方法

## 8.2. Bundle Adjustment and Graph Optimization - BA 与图优化   

在 3D 视觉重建 (visual 3D reconstruction) 任务中, BA 与图优化都是常见概念.
* Bundle Adjustment : 从视觉图像中提炼出最优的 3D 模型和相机参数 (内外参数). 
  * 考虑从任意特征点发出的几束光线(`bundles` of light rays), 会在几个相机的成像平面上变成像素或者检测到的特征点.  
  * `Adjustment` 调整相机位姿和各个特征点的空间位置, 使得这些光束最终能够收束到相机的光心.  

BA 本身在第 4, 6 章节有学习.  在该章节主要研究 BA 对应的图模型结构的特点, 以及在图模型上的通用快速求解的方法.  

### 8.2.1. The Projection Model and COst Function - 投影模型和 BA 代价函数  
<!-- 完 -->
复习投影模型:
1. 拥有 相机外参数$(R,t)$, 将已知的 地标 的世界坐标转换到相机坐标
  * $P'=Rp+t = [X',Y',Z']^T$
2. 投影到 归一化平面, 得到 归一化坐标
  * $P_c=[u_c,v_c,1]^T=[X'/Z',Y'/Z',1]$
3. 附加畸变, 公式略, 得到 $u_c',v_c'$
4. 计算在图像上的坐标
  * $u_s=f_xu_c'+c_x, v_s=f_yv_c'+c_y$

在 SLAM 中, 其实投影过程就是 观测方程. 在之前的学习中将它抽象的记为 $z=h(x,y)$  
* $x$ 指代 相机的当前位姿, 即外参数 $R,t$. 对应的李群为 $T$, 李代数为 $\xi$
* $y$ 指代路标 的三维点, 观测得到的数据就是像素坐标 $z\triangleq [u_s,v_s]^T$

则, 可以列出此次观测的误差为 相机位姿和 路标空间坐标 的关系式, 即 $z$ 看到的和 $h(T,p)$ 算出来的差分  
$$e=z-h(T,p)$$

整合其他时刻, 其他路标, 给误差添上坐标, 就有了整体系统 overall 的代价函数  

$$ \frac{1}{2}\sum_{i=1}^m\sum_{j=1}^n ||e_{i,j}||^2=\frac{1}{2}\sum_{i=1}^m\sum_{j=1}^n ||z_{i,j}-h(T_i,p_j)||^2= $$
对该方程进行最小二乘求解, 即所谓的 BA, 对位姿和路标进行同时调整.  

### 8.2.2. Solving Bundle Adjustment - BA的求解  

### 8.2.3. Sparsity - 稀疏性

### 8.2.4. Robust Kernels - 鲁棒核函数  

# 9. Filters and Optimization Approaches - 后端 Part2

<!-- 头部完 -->

* 理解滑动窗口优化
* 理解 pose graph optimization 位姿图优化
* 理解带 IMU 紧耦合的优化
* 掌握 g2o 的位姿图优化方法  

上一张介绍了以 BA 为主的图优化, 尽管 BA 能够精确地优化每个相机位姿与特征点的位置, 然而在更大的场景中, 大量的特征点的存在会严重降低计算效率, 导致计算量随着时间持续增长 从而无法实现 SLAM 的实时性

本章节介绍一种简化的 BA : 位姿图 pose graph  

## 9.1. Sliding Window Filter and Optimization - 滑动窗口滤波和优化

### 9.1.1. Controlling the Structure of BA - BA 的结构在实际环境下的控制  
<!-- 完 -->

带有相机位姿和空间点的图优化称为 BA , 在大规模建图以及 SfM 问题中十分有用, 然而在 SLAM 过程中需要控制 BA 的规模, 确保计算的实时性.  
* SLAM 系统中必须限制 后端的计算时间  e.g. BA规模不能超过 1万个路标点, 迭代不超过 20次等
* SfM 任务中可能会有 用一周时间 重建一个 城市地图的算法, 在 SLAM 中不现实

控制规模的问题可以是理论上的, 也可以是工程上的
* 从连续的视频中抽出一部分作为 关键帧, 关键帧参与和路标点的 BA, 非关键帧只用于定位, 不参与建图
* 仅保留离当前时刻最近的 N 个关键帧, 去掉时间上更早的. BA 将被固定在一个时间窗口内.  
* 可以考虑在空间上有一定展开的帧, 这样在相机停止运动的时候保证关键帧不会缩成一团引起精度退化
* 考虑共视图 (Co-visibility graph), 即与当前相机存在共同观测路标的关键帧, 构建共视图, 这样能在长时间内保持最优

工程上, 还需要考虑 不参与计算的变量的存留方式
* 固定: 例如共视图之外的变量, 保持不变即可
* 丢弃: 彻底删除变量 

### 9.1.2. Sliding Window - 滑动窗口法


## 9.2. Pose Graph Optimization - 位姿图
### 9.2.1. Definition of Pose Graph - 位姿图的定义 
<!-- 完 -->
相比于 位姿的优化, 特征点在优化过程中 占据了计算的绝大部分. 经过多次较好的观测之后, 已经收敛的特征点的位置变化已经很小了,  而发散的 outliers 也已经被剔除. 因此每次都对 收敛点进行计算优化 是一种冗余的计算.  

一种思想是 在优化几次后就将 特征点固定住, 将其看作 相机位姿估计的 约束  

而另一种则是 位姿图的基础思想, 完全不优化路标的位置. -> 一旦出池化完成, 就不再优化路标点的位置, 而只关心所有的相机位姿之间的联系.  

事实上, BA 中特征点的数量的确远大于位姿节点, 关键帧也往往包含了 数百个关键点. 工程上:
* 要么采用 滑动窗口法
* 要么采用 位姿图, 而且 位姿图的计算和 额外的 Pose 传感器的耦合也相对容易

### 9.2.2. Residuals and Jacobians - 位姿图的优化方法  



# 10. Loop Closure - 回环检测

# 11. Dense Reconstruction - 建图

# 12. Discussions and Outlook - SLAM 的现在与未来

