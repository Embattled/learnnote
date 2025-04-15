Wang, X., Xie, L., Dong, C., & Shan, Y. (2021).
Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data. 
2021 IEEE/CVF International Conference on Computer Vision Workshops (ICCVW). 
Presented at the 2021 IEEE/CVF International Conference on Computer Vision Workshops (ICCVW), Montreal, BC, Canada. https://doi.org/10.1109/iccvw54120.2021.00217


# 1. Introduction

早期的基于深度学习的 SR 都是对双线性降采样进行建模, 这和实际上的画质退化相差很远

后来的 Blind Super-resolution 着重于建模未知的,复杂的实际退化. 现有的方法可以分为两类, 隐式建模和显式建模.

*   显式退化模型包括, blur, downsampling, noise, JPEG 压缩,
*   隐式的退化模型着重于用 GAN 来拟合实际的劣化图片, 但是这会受限于训练集的domain

本论文的工作是

1.  通过重复多次传统的退化模型, 来实现从 1阶退化到高阶退化的转变, 得以更加逼近真实情况的退化. 实验中, 作者使用了 2阶退化来平衡效果和复杂度.  此外, 在退化中引入了 sinc 滤波器用以模拟 ringing 鬼影和 overshoot 鬼影
2.  使用了 U-Net 作为识别器替代了原本ESRGAN工作中的VGG识别器, 此外使用了Spectral Normalization(SN) 正则化来提高学习的稳定性

# 2. Related Work


3.  基于简单的双线性降采样已经证实在实际应用中的效果不理想
4.  通过显式建模退化, 将模型分为两个部分, 退化预测和条件Restoration使得结果可控, 但是这导致最终效果受退化预测模型的精度影响, 同时退化预测也只能实现简单的退化模型
3.  另一种是获取大量的 pairs图像, 来跳过对退化建模, 直接拟合优化过程, 但通常需要限定输入相机, 并且需要大量的数据以及对齐标定.
4.  或者通过 cycle consistency loss 来直接学习 unpaired data, 但是学习很有挑战性, 而且结果不理想

# 3. Methodology

传统退化模型, blur, 降采样，噪声， jpeg压缩

1.  blur: 各向异性的高斯模糊，用2维协方差矩阵表示模糊核的方向。在此基础上取 beta 次方，得到 generalized gaussian blur，作者声称这会在某些real samples中取得更好的效果
2.  noise: 高斯灰噪和彩噪，泊松噪声
3.  resize: 截取，双线性和双三次插值，考虑到升降采样，直接用resize函数
4.  jpeg: 随机质量参数， 使用了可微分的 Pytorch 实现的 jpeg实现

高次退化： 按照 blur, resize, noise, jpeg 的顺序执行第二遍 （顺序不进行打乱？）

Ringing和 Overshoot：通常一起出现， 主要的由来是 sharping 和 JPEG压缩。使用 sinc 滤波模拟。模拟的效果很接近，在整个流程中的最初和最终应用两次 sinc滤波

网络结构和训练方法

1.  ESRGAN Generator, 用 Pixel shuffle 降低 GPU 压力
2.  U-Net 识别器，能够更好的获取局部准确度梯度信息，给每一个像素都返回 realness值。在此处应用了 Spectral Normalization Regularization稳定训练，同时还能减轻过度锐化和伪影。
3.  训练过程，**先用 L1正常训练，再添加 perceptual Loss和 GAN loss**

# 4. Experiments

1.  DIV2K Flickr2K OutdoorSceneTraining
2.  Patch size 256
3.  V100  x4
4.  ESRNet 2x10-4, ESRGAN 1x10-4。 GAN网络的学习率没有特别大的下降
5.  GAN Loss的权重只有 0.1，其他两个都是 1.0
6.  perceptual loss 使用了各层的特征图，权重为 0.1 0.1 1 1 1
7.  Exponential moving average (EMA)
8.  second-order 退化使用与第一层几乎完全相同的参数，太多了不做笔记
9.  batch pool 用于提高 batch 内的数据多样性（我觉得可以通过更好的实现方法跳过这一步）
10. 对 GT 执行 sharpen 提高锐化效果
