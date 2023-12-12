# Simultaneous Localization And Mapping (SLAM)

## Glossary

* visual(-inertial) odometry        (VO/VIO)
* Maximum a Posteriori              (MAP)
* Bundle Adjustment                 (BA)
* Pose-Graph                        (PG)



* feature based method
* direct method
## Introduction

当前业界对于 VO/VIO 和 SLAM 的边界越来越模糊 (diffuse)
* SLAM : 根据搭载在移动设备上的传感器来试试绘制地图, 同时确定移动设备在地图中的位置
* VO   : 更多的聚焦在 agent 的 ego-motion (自我移动), 而不是地图的绘制上
* SLAM 相比于 VO, 从思想上能够更多的使用 previous observations
  * Short-term data association : 使用 last few seconds, 在曾经的 VO 上被主要使用, 丢弃走出视野外的数据, 会导致 drift
  * Mid-term data association : 匹配地图元素中 距离相机较近 同时 drift 较小的元素, 能够实现在已绘制地图的区域中实现 0 drift
  * Long-term data association : 主要用于在大型环境中保证 SLAM 的精度, 基于 place recognition, 允许重设 drift 为 0, 以及矫正既存地图

### Stereo SLAM

* 过远的点由于立体图形的时差过小, 会导致 depth 不能可靠的测量, 需要一些 trick

## Algorithms 



keyframe-based
initialization: obtain the poses of the first two keyframes and the initial map
* monocular systems : homography and 5-point relative pose algorithm
* stereo cameras    : can recover the depth information and initialize the map directly
* multi-camera setups : there could be various ways of combining different cameras depending on the extrinsic parameters.

Keyframe Selection: maintain a fixed number of keyframes in the front-end as the local map, against which new frames are localized. 
* crucial for the performance of SLAM systems.
* heuristic-based methods:  combination of different heuristics usually relies on certain assumptions of the sensor configurations and scenes, makes parameter tuning as well as the application to general multi-camera setups complicated.
  * Camera motion : whether the current frame is a certain number of frames away from the last keyframe.
  * Number of tracked features : A new keyframe is selected if the number or the percentage of tracked features in the current frame falls below a certain threshold
  * Optical flow  :  The Euclidean norm between the corresponding features from the current frame and the last keyframe.
  * Brightness change : relative brightness support direct methods
* information-theoretic methods:
  * 

Map Management and Query : estimate the pose of newly coming frames


# Paper Memo

## Survey



## Single


ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial and Multi-Map SLAM.  
Campos, C., Elvira, R., Rodriguez, J. J. G., M. Montiel, J. M., & D. Tardos, J. (2021).  
IEEE Transactions on Robotics, 1874–1890. https://doi.org/10.1109/tro.2021.3075644  
* IMU initialization technique
* multi-session map-merging functions







Redesigning SLAM for Arbitrary Multi-Camera Systems 2020.03.04  
Juichung Kuo, Manasi Muglikar, Zichao Zhang, Davide Scaramuzza  
IEEE Conference on Robotics and Automation (ICRA), Paris, 2020  

Focus on : an adaptive design for general multi-camera VO/VIO/SLAM systems
Proposed:
* adaptive initialization scheme  : analyzes the geometric relation among all cameras and selects the most suitable initialization method online.
* a sensor-agnostic, information-theoretic keyframe selection algorithm.
* a scalable, voxel-based map management method.




Significance of omnidirectional fisheye cameras for feature-based visual SLAM
Author(s)
Chang, Raphael,M. Eng.Massachusetts Institute of Technology. 
* MIT 的硕士论文