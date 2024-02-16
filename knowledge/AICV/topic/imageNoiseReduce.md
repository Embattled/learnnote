# 1. Image Noise Reduce


# 2. Ground


# 3. Paper Memo



## 3.1. (CVPR 2020) A Physics-Based Noise Formation Model for Extreme Low-Light Raw Denoising 

简称 ELD (Extreme Low-Light Denoising)

主要工作
* 一种新的噪声合成方法, 其对于低照度图像的噪声模拟效果更好
* 一种噪声标定方法
* 一个数据集  

$N_p$  表示光子散粒噪声, 最遵循泊松分布的物理本质噪声  

read noise 一般被认为是高斯模型, 但在论文中阐述 实际中数据具有长尾性质, 被建模为 : Tukey lambda distribution (TL)  
$N_{read}=N_b+N_t+N_s \sim TL(\lambda;0,\sigma_{TL})$
* $N_t$ thermal noise 热噪声
* $N_s$ source followr noise 
* $N_b$ banding pattern noise

* $N_q \sim U(-1/2q, 1/2q)$ 量化噪声, 模拟信号到数字信号的截断噪声  


噪声的标定
* 平场照片获取 散粒噪声参数 K
* 暗场照片获取 行噪声,  减去行噪声后在对 $N_{read}$ 进行拟合

## 3.2. (ICCV2021) Rethinking Noise Synthesis and Modeling in Raw Denoising

简称 SFRN (Sample From Real Noise)  

