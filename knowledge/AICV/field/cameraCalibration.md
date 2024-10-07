# Camera Calibration



# Paper

## Rethinking Generic Camera Models for Deep Single Image Camera Calibration to Recover Rotation and Fisheye Distortion



Deep Single Image Camera Calibration With Radial Distortion : 深度学习方法同时标定内外参数 
* 所使用的相机模型不适用于鱼眼, 参数过少


Deep Single Fisheye Image Camera Calibration for Over 180-degree Projection of Field of View
* 面向语言, 同时标定焦距和外参, 缺少内参


基于深度学习的标定的 loss 分量的权重问题:
* 联合 joint loss 用的比较多, 而各自的权重都是超参数, 基于实验得出或者直接是相同权重


论文成果:
1. 新的鱼眼相机模型
2. 基于深度学习的相机标定方法
3. 新的 Loss function


通用相机模型:
* 畸变参数 $\gamma = \tilde{k_1}\eta+\tilde{k_2}\eta^3...$
* 此处 $\eta$ 是相机的入射角

新相机模型
* $\gamma = f(\eta+\tilde{k_1}\eta^3)$
* 此处 $f$ 是相机焦距, k1 是畸变系数, 即模型显示定义了焦距
* 计算梯度也很容易

