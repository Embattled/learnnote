# Introduction to Visual SLAM : From Theory to Practice
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


## Classical Visual SLAM Framework


1. Sensor data acquisiton. 获取各种传感器数据, 图像, IMU, motor encoders 等所有.
2. Visual Odometry (VO). 也被称为 `frontend`, 通过较小范围的相邻帧, 获取该小范围内的相机移动, 生成一个 rough local map. 通过将 VO 得到的相机序列进行累加, 即可得到最简易的 SLAM 系统, 然而 accumulative drift 是这种方法最主要的问题, 也是 SLAM 系统为什么还需要一个全局的 backend 以及 loop closing的原因.
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


## Mathematical Formulation of SLAM problems

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


# 3D Rigid Body Motion

如何描述一个物体在 3D 空间下的移动

## Rotation Matrix


### Points, Vectors, and Coordinate Systems



<!-- omit from toc -->

