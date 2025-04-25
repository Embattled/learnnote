
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

# 1. Introduction

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


# 2. Related Works

## 2.1. Optical Flow Estimation

预测 per-pixel motion

FlowNet: Learning Optical Flow with Convolutional Networks (2015) 为深度学习的光流方法 milestone  
RAFT: Recurrent All-Pairs Field Transforms for Optical Flow 循环迭代的 pair 获取光流  

还有的研究方向就是无监督光流估计  

## 2.2. Video Frame Interpolation

光流是 视频插值 VFI 任务中 最新方案中的重要组成部分  

SuperSlomo:
* linear combination of the two bi-directional flows as an initial approximation of the intermediate flows and then refining them using U-Net

DAIN: Depth-Aware Video Frame Interpolation 
* estimate the intermediate flow as a weighted combination of bidirectional flow

SoftSplat:
* forward-warp frames and their feature map using softmax splatting

QVI:
* exploit four consecutive frames and flow reversal filter to get the intermediate flows

EQVI:
* extend QVI with rectified quadratic flow prediction


其他的方法则基于隐式的推理 帧之间的运动  
* Deep Animation Video Interpolation in the Wild


## 2.3. Knowledge Distillation

本文的 privileged distillation for intermediate flow 属于知识蒸馏领域  

本文的方法属于 codistillation:
* Large scale distributed neural network training through online distillation


# 3. Method

## 3.1. Pipeline Overview

输入 两个图片 I0 I1, 以及中间时间戳 t, 输出中间帧 It  

模型输出
* 两个预测的流 $F_{t\rightarrow 0}, F_{\rightarrow 1}$
* 一个 fusion map `M`

$$
\hat{I}_{t \leftarrow 0} =  W(I_0, F_{t\rightarrow 0}), \hat{I}_{t \leftarrow 1} =  W(I_1, F_{t\rightarrow 1})\\
\hat{I_t} = M \odot \hat{I}_{t \leftarrow 0} + (1-M) \odot \hat{I}_{t \leftarrow 1}
$$

拿到最终的 hat(I_t) 后, 还需要再传入一个输出残差的 RefineNet 用于消除 artifacts 以及提高高频区域的效果

RefineNet 不属于文章的核心内容但是影响最终结果  


## 3.2. Intermediate Flow Estimation


早期 VFI 方法会先预测 双向 optical flow  bi-directional flows
但是这种方法很难处理物体的运动  

早期的方法会在 optical flow field 上执行空间插值, 由于 物体的平移 object shift 问题并不容易实现  

IFNet 通过卷积网络直接输出  $F_{t\rightarrow 0}, F_{\rightarrow 1}, M$
当且仅当, t=0 or t=1 的时候, IFNet 的表现与传统 optical flow 相同  

网络结构为 coarse-to-fine, 用于 提高训练效率  

对比传统光流方法, 推理速度非常快  


## 3.3. Priveleged Distillation for Intermediate Flow

直接预测两个帧的 intermediate 流是一个很困难的任务, 相比起来直接预测两幅图之间的 流容易的多  
如果能够获取 intermediate 的一定程度的信息, 那么预测插值流则会变得简单很多  

privileged distillation loss:
* 在正式的模型基础上再添加一个 Teacher block
* 只有 teacher block 能够访问到 GT 的插值流
* Teacher block 拿到最高精度的预测 flow
* 网络的最终输出 Flow 和 Teacher block 的输出流之间计算 `L_dis = L2Loss(F_teacher, F_final)`

L_dis 不会被反向传播到 教师模块 , 训练结束后 Teacher block 会被丢弃  


## 3.4. Implementation Details

完整的 Loss为  L_total = L_rec + L_rec_teacher + lamda L_dis  
* lamda = 0.01 用于平衡 Loss 的 scale
* L_rec 为金字塔每层分别计算出来的, d 为分层 pixel-wised loss, 具体用的是 L1
  * L_rec = d(I_t, I_gt)
  * L_rec_teacher = d(I_teacher, I_gt)


训练数据集为: `Vimeo90K`, 具有 51312 个三元组  
Data Aug:
* vert/hori flip
* 时域反转
* rotating 90 degrees

训练策略:
* 固定 `t=0.5`
* AdamW
* LR: 10e-4 -> 10e-5
* Cosine Annealing
* 8 TiTAN X for 300epochs in 10 hours

延申训练
* 使用 `Vimeo90K-Septuplet`
* 7 帧为一个单位
* 


# 4. Experiments - 实验

## 4.1. Benchmarks and Evaluation Metrics

在标准 评测集上 取得了迄今最好的效果

## 4.2. Comparisons with Previous Methods

实现了两个重要方向的能力
* Interpolating Arbitrary-timestep Frame 任意时间戳帧生成
* Model Scaling 输入解析度扩张
* Middle Timestep Interpolation 在标准的中间帧生成任务中实现了最好的性能(速度)

## 4.3. General Temporal Encode

提供给模型双目输入

给模型提供不同的 t
图像 pair 的每列使用不同的 t 进行推理, 可以生成假想的宽 FOV 图片  


## 4.4. Image Representation Interpolation

图像特征表示的插值

除了普通的 图片插值, 对于其他的任务输出, 例如 单目深度估计 也可以进行帧插值  



