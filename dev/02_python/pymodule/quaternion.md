# Quaternions in numpy 

Numpy 的四元数附加模组  

This Python module adds a quaternion dtype to NumPy.  
要安装的话, 使用:
`conda install -c conda-forge quaternion` or
`python -m pip install --upgrade --force-reinstall numpy-quaternion`  

除此之外还有一个不基于 Numpy 的纯 四元数库 : `quaternionic` , 对比于底层是 C 写的 Numpy, 在平台兼容性方面可能会更好

以及一个停止维护的很旧的库 `pyquaternion` 




# numpy-quaternion


## quaternion

* `from_rotation_matrix(rot, nonorthogonal=True)`   
  * 从一个 3x3 的旋转矩阵获取四元数
  * 使之满足四元数定义 : 
    * 对于输入的旋转矩阵 `rot`, 获取到的 四元数 `q` 应用于表示空间点的 纯向量四元数 `v` 有
      * `rot @ v.vec == q*v*q.conjugate()` 
      *    


## quaternion_time_series

时间序列

* `slerp(R1, R2, t1, t2, t_out)`
  * 球面线性插值
  * 输入两个 四元数代表的旋转, 其对应的时间轴, 插值出来 `t_out` 时刻的旋转矩阵
  * 参数
    * R1,R2 : quaternion
    * t1, t2: float
    * t_out : array of floats, float


# Quaternionic

沿用了 numpy-quaternion 的思想, 是一个比较新的库

