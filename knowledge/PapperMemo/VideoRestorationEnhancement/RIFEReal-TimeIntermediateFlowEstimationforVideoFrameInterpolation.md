
RIFE: Real-Time Intermediate Flow Estimation for Video Frame Interpolation

ECCV2022

    @inproceedings{huang2022rife,
    title={Real-Time Intermediate Flow Estimation for Video Frame Interpolation},
    author={Huang, Zhewei and Zhang, Tianyuan and Heng, Wen and Shi, Boxin and Zhou, Shuchang},
    booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
    year={2022}
    }
Video Frame Interpolation (VFI)  


提出网络结构 IFNet, 预测 内部的 flows  


特权蒸馏方案  
privileged distillation scheme 用于 稳定的训练  

不依赖 预训练的 optical flow, 支持 任意 timestep frame interpolation  

与流行的 SuperSlomo 和 DAIN 对比, 更快且效果更好  

# Introduction

插针生成的应用
* slow-mothion generation
* video compression
* video frame predition
* 降低高分辨率的 streaming 的网络带宽要求
* 为低性能计算设备 提供/提高 视频编辑能力  
* 为高刷新率屏幕提供内容支持


通常基于深度学习的方法都是分两步:
1. 将输入帧 warp 为 optical flow
2. 通过 CNN 网络 fuse warped frames 
非常依赖 optical flow 的准确度, 而且往往插帧需要预测双向的 flow


本论文提出了
* 不再需要其他的组件, 例如用于补偿中间流预测的缺陷: 图像深度, flow refinement model, flow reversal layer, 
  * 消除对 SOTA optical flow 模型的依赖
* e2e 训练的 motion estimation: CNN 相比去建立一些不准确的 motion modeling, 直接去学习 e2e 的 flow 更好
* 对中间中间流进行直接监督训练 : 大多数 VFI 模型只对最终的重建结果计算损失
  * 然而, pixel-wise loss在梯度传播中, 由于经过了 warping operator, 最终对 flow estimation 的优化效果并不高效
  * 在此之前的方法缺乏 专门设计用于 flow estimation 的 监督, 会导致最终 VFI 的性能下降

本文提出 IFNet: 直接预测 intermediate flow from adjacent, 并暂时的编码输入  
* 计算高效, 应用了一些 高效网络的思路
* 采用 coarse-to-fine 策略, 逐步提高分辨率 
* 通过 连续的 IFBlocks 迭代性的更新 intermediate flows 和 soft fusion mask
  * 根据 flow fields, 可以 将两个输入帧中 相应的像素 直接地洞到潜在中间帧中的相同位置 ??

IFNet 直接使用 reconstruction loss 并达不到 SOTA 的精度  
* optical flow estimation 的精度不达标
* 当应用了 privileged distillation scheme 后 效果有了大幅度的改善  

最终实现了 RIFE  Real-time Intermediate Flow Estimation
* 不需要 pre-trained models or datasets with optical flow labels
* 文章的主要贡献为
  * IFNet， privileged distillation scheme
  * 在 arbitrary-time frame interpolation 也实现了 SOTA
  * RIFE 在 depth map interpolation 和 dynamic scene stitching  任务中也可以实现


# Related Works

## Optical Flow Estimation


## Video Frame Interpolation


## Knowledge Distillation



# Method

## Pipeline Overview

输入 两个图片 I0 I1, 以及中间时间戳 t, 输出中间帧 It  

模型输出
* 两个预测的流 $F_{t\rightarrow 0}, F_{\rightarrow 1}$
* 一个 fusion map `M`

$$
\hat{I_{t \leftarrow 0}} =  W(I_0, F_{t\rightarrow 0}), \hat{I_{t \leftarrow 1}} =  W(I_1, F_{t\rightarrow 1})\\
\hat{I_t} = M \odot \hat{I_{t \leftarrow 0}} + (1-M) \odot \hat{I_{t \leftarrow 1}}
$$

拿到最终的 hat(I_t) 后, 还需要再传入一个输出残差的 RefineNet 用于消除 artifacts 以及提高高频区域的效果

RefineNet 不属于文章的核心内容但是影响最终结果  


## Intermediate Flow Estimation

## Priveleged Distillation for Intermediate Flow

## Implementation Details

