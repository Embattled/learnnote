# 1. Image Transformation

最广义的图像处理

input image is tranformed into an output image:
* image denoising
* super-resolution
* colorization
* style transfer

输入 image 通常被称为 degraded image: noisy, low-resolution, grayscale  


## Loss 损失函数

* pixel-loss 
* perceptual loss   : 直接使用预训练的特征提取网络作为 loss
* conditional random field (CRF) : CRF loss


这一思路与一致性正则化（Consistency Regularization）、对比学习（Contrastive Learning）、以及特征对齐（Feature Alignment） 有很大关系。已有相关的研究和方法包括：
* Feature Matching
* Consistency Regularization
* Contrastive Learning


# 2. Image Restoration and Enhancement

图像的复原和增强是广义上的图像处理应用  

相对来说 具体的 
* 图像去噪,去模糊,exposure adjustment,超分辨率
算是低级的 imagery applications, 或者说图像恢复的子任务






# 3. (NTIRE) New Trends in Image Restoration and Enhancement




# 4. Network

### 4.0.1. NAFNet


2022/04/10  Simple Baselines for Image Restoration

旷视科技的文章, 化繁为简, 设计了一个及其简单的网络并达成了 NR 和 Deblur 任务的 SOTA

Inter-block Complixity : 块间复杂度, 设计多了级联的 U-Net 来提高性能
* Multi-Stage Progressive Image Restoration(2021)
* HINet: Half Instance Normalization Network for Image Restoration (2021)

Intra-block Complexity : 块内复杂度, 引入 Attention 进行降噪  


### 4.0.2. CUGAN - Toward Interactive Modulation for Photo-Realistic Image Restoration

controllable Unet

背景: 图像复原可以分为两个方向
* PSNR-Oriented     : 底层是 MSE, 会导致忽视 mild degradation 而只重视 severe degradation, 导致不平衡学习
* GAN-based         : 对于 servere degradation, 会直接被 discriminator 判定为 fake, (导致梯度消失?)


要想提出一个 GAN 方法能够处理多种程度的退化, 需要让 discriminator 能够区别从各种不同的退化程度中恢复的图像. 
根据输入图像的退化来调整其对退化程度判定的基准.  


