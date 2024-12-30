# RoMa: A lightweight library to deal with 3D rotations in PyTorch

https://naver.github.io/roma/




## class


`class RigidUnitQuat(linear, translation)`
* 刚体变换, 通过传入 四元数定义的旋转和 平移来构建
  * `linear` : (…x4 tensor): batch of unit quaternions defining the rotation.
  * `translation` : (…x3 tensor): batch of matrices specifying the translation part.
* 在计算的时候, quaternions 会被假定为 单位范数
  * 因此在不能保证输入是 单位范数的情况下先调用 normalize() 在进行别的操作


`RigidUnitQuat` 的转换接口:
* `normalize()`   : 复制该变换, 并归一化旋转为单位范数
* `to_homogeneous(output=None)` :
  * `output` (...x4x4 tensor or None) : 输出的地方可以通过参数来指定, 也可以直接用返回值
  * `Returns`: A …x4x4 tensor of homogeneous matrices representing the transformation.
  * 能够确保得到的齐次矩阵的最后一行 是 `0,0,0,1`



`rigid_points_registration(x, y, weights=None, compute_scaling=False)`
* 给定起始点集和目标点级, 计算 R,t,s, 使得对 x 应用 Rts 后距离 y 点级的距离最小
