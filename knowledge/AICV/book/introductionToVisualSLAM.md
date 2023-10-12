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
\end{Vmatrix}
= 
\begin{bmatrix}
  a_2b_3-a_3b_2\\
  a_3b_1-a_1b_3\\
  a_1b_2-a_2b_1
\end{bmatrix}
=
\begin{bmatrix}
  0 & -a_3 &a_2\\
  a_3 & 0 & -a_1 \\
  -a_2 & a_1 & 0
\end{bmatrix}b
$$

两个向量的外积结果是一个向量, 方向垂直 (perpendicular) 于两个向量构成的面, 且大小为 $|a||b|\sin<a,b>$  
是 **两个向量张成的四边形的有向面积** area of the quadrilateral  

定义 $\^{}$ 为把一个矩阵变成反对称矩阵的形式 skew-symmetric matric.  
反对称矩阵满足 $A^T = -A$  
这样就把矩阵的外积转换为矩阵向量乘法, 一个向量对应着唯一一个 反对称矩阵.  


要注意, 尽管计算上是根据向量的坐标来计算的, 但是向量之间的 加减法, 内外积运算其实是独立于坐标的, 即使在不谈论它们的坐标时也可以计算.  

### Euclidean Transforms Between Coordinate Systems 坐标系之间的欧氏变换

定义 : 某个点在相机坐标系下的坐标为 $p_c$, 而在世界坐标系下的坐标为 $p_w$   

对于两个坐标系之间的移动, 是一种 刚体移动 rigid body motion. 即相机 camera 的运动是刚体的 (rigid).  在运动的过程中, length and angle of the vector will not change.    

这种刚体移动下, 可以描述坐标系之间相差的是一个 欧氏变换 (Euclidean transform)

旋转矩阵的推导:  假设某个向量在旋转前后的坐标分别为 $a$ 和 $a'$, 坐标系的基为 $\bold{e}$ 和 $\bold{e'}$

则由向量的不变性引出的  
$$
[e_1,e_2,e_3]
\begin{bmatrix}
  a_1\\
  a_2\\
  a_3
\end{bmatrix}
=
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
\end{bmatrix}
=
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
* 是正交 orthogonal 矩阵 ( 逆矩阵为自身的专职 )

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

### Transform Matrix and Homogeneous Coordinates 变换矩阵与齐次坐标

引入齐次坐标, 使得旋转和平移能够写在同一个矩阵里, 称为变换矩阵 T (transformation matrices)  

$$
\begin{bmatrix}
  a'\\
  1
\end{bmatrix}
=
\begin{bmatrix}
  R& t\\
  0^T & 1
\end{bmatrix}
\begin{bmatrix}
  a\\
  1
\end{bmatrix}
=
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

## Rotation Vectors and the Euler Angles 旋转向量和欧拉角

### Rotation Vectors 旋转向量

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

### Euler Angles 欧拉角




<!-- omit from toc -->

