# 1. Image Restoration and Enhancement

图像的复原和增强是广义上的图像处理应用  

相对来说 具体的 
* 图像去噪,去模糊,exposure adjustment,超分辨率
算是低级的 imagery applications, 或者说图像恢复的子任务






# 2. (NTIRE) New Trends in Image Restoration and Enhancement




# 3. Network

### 3.0.1. NAFNet


2022/04/10  Simple Baselines for Image Restoration

旷视科技的文章, 化繁为简, 设计了一个及其简单的网络并达成了 NR 和 Deblur 任务的 SOTA

Inter-block Complixity : 块间复杂度, 设计多了级联的 U-Net 来提高性能
* Multi-Stage Progressive Image Restoration(2021)
* HINet: Half Instance Normalization Network for Image Restoration (2021)

Intra-block Complexity : 块内复杂度, 引入 Attention 进行降噪  


### CUGAN - Toward Interactive Modulation for Photo-Realistic Image Restoration

controllable Unet

背景: 图像复原可以分为两个方向
* PSNR-Oriented     : 底层是 MSE, 会导致忽视 mild degradation 而只重视 severe degradation, 导致不平衡学习
* GAN-based         : 对于 servere degradation, 会直接被 discriminator 判定为 fake, (导致梯度消失?)


要想提出一个 GAN 方法能够处理多种程度的退化, 需要让 discriminator 能够区别从各种不同的退化程度中恢复的图像. 
根据输入图像的退化来调整其对退化程度判定的基准.  

