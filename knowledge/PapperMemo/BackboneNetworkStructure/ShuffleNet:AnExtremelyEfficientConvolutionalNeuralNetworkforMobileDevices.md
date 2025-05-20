ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices

2017年
Shuffle Net 第一篇论文

Zhang, Xiangyu, Xinyu Zhou, Mengxiao Lin, and Jian Sun. 2018.
“ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices.” In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition. doi:10.1109/cvpr.2018.00716.

# 1. Introduction

追求在维持精度的同时降低深度学习的计算量以便于在低性能设备上运行  

本文着重于通过构造高速的 网络结构 来实现对应的低计算量网络  

通过分析出 1x1 卷积的冗余性, 将其 group 化, 并通过 channel shuffle 操作来降低 group 化的性能损失, 提高特征量的表现力  

对比 MobileNet 提高了非常多的性能  

# 2. Related Work

**Efficient Model Designs**
基本上高效网络的改进对象都是VGG, 有 `GoogLeNet`, `SqueezeNet`, `SENet`.  `ResNet` 也作为高效率网络的一种
还有 `NASNet`

**Group Convolution**

该方法最早是 AlexNet 用来在 2 个GPU 上训练的 tirck  

在 `ResNeXt` 上 证明了有效性  

并于 `Xception` 上被拓展为了 separable convolutions   

`MobileNet` 进一步提出了 depthwise separable convolutions, 并成为了 SOTA 的轻量模型  

本文进一步提出了新形式的 group convolution and depthwise separable convolution 


**Channel Shuffle Operation**

Channel Shuffle 作为精度提升的方案在另一篇论文中被提及, 但没有验证 channel shuffle 的高效性  
Interleaved Group Convolutions for Deep Neural Networks

相比起来本论文着重描述高速性能

**Model Acceleration**
模型后处理加速的方案也有提及，包括剪枝类和量化类方法，除此之外傅里叶类的层以及蒸馏也被提及

# 3. Approach

**Channel Shuffle for Group Convolutions**

作者提及了 1x1 Conv 的计算量问题，占据了整个网络的9成。
一个简单的解决方案是将1x1也进行 Group 推理，但是这样的话中间层的特征量只从一部分的输入通道得来，几个group是从始至终隔离的，会导致模型表征能力下降，

通道混淆则可以解决这个问题

**ShuffleNet Unit**

将 1x1 Conv 转化成 Group 1x1 后接 Channel Shuffle 运算, 然后再接 3x3 Depth Wise Conv

在这之后还有一个 1x1 Conv 用于匹配 block 的 Shortcut 维度

参照其他的论文, 在 Depthwith Conv 后不使用ReLU 激活函数

Fig2 C 还提供了一个带 Stride 的 block 实现方法
*   shortcut 使用 3x3 stride=2 AVG pool
*   3x3 Depthwise Conv 使用 Stride=2
*   与shortcut合并使用 Concat 而非 Add

ShuffleNet 的思想可以在计算量受限的时候尽可能的提高通道数  

尽管如此, 理论计算量并不能反映在实际的计算效率上, 由于 Channel Shuffle 会降低内存命中率, 该模块并不适合应用在所有网络模块  

论文 `Xception` 也有提到这一点, 基于 2017年的 TensorFlow  

在该文章中, ShuffleNet unit 值会替换那些 作为 `瓶颈` 的模块  
这里的瓶颈似乎指的是  Res 残差模块中, 位于模块中间的通道数相比输出通道更少的 网络层  


**Network Architecture**

网络为 识别网络

分为 3个 stages, 每一个 stage 的一一开始会进行下采样, 同时加倍通道数  

而被 Shuffle unit 替换的 bottleneck 的部分则通道数为输出通道数的 1/4 
number of bottleneck channels to 1/4 of the output channels for each ShuffleNet


ShuffleNet unit 中的分组数是可控制的参数, 会影响计算量以及 连接的离散度, 在试验阶段论文尝试了不同的 分组数 g, 同时相应的微调输出通道数, 用来平衡最终的网络计算量  
分组数越高, 在计算量固定的时候能够采用的通道数就越多, 相应的特征信息量越多, 但是同时 由于每一个卷积层的输入视野变窄, 其特征表现力相应也可能下降  
详细的调查在实验章节  


# 4. Experiments

在经典 ImageNet 2012 图像分类上进行评测  

一些超参数进行了修改, 按照 MobileNet 文章的说法, 轻量化模型通常需要面对 欠拟合问题 而非过拟合  

## 4.1. Ablation Study

**Pointwise Group Convolutions**
评价  group 数对性能的影响

g=1 相当于标准的 depthwise separable conv in Xecption  


结论: 在计算量统一的情况下, 分组 g 越大, 性能提升越多
* 可能是因为 特征图的通道数更大
* 轻量化模型的 通道数基数更小, 因此更可能受这个改动影响效果  
* 在 3 个计算量维度  140/38/13  MFlops 都取得了性能进步  
  * 然而 38 MFlops 维度中, g=8 的性能反而弱于 g=4
  * 因此这个并不是绝对的, 在较大计算量和 channel shuffle 式 group conv 在特别小计算量的网络中都可能比较有效果 

**Channel Shuffle vs. No Shuffle**

这部分实验在 相同的分组数 g 的情况下, 应用 channel shuffle 所带来的性能改进  
* 文章提供了 g=3  和 g=8 的结果, 都证明了 channel shuffle 效果的有效性
* g 越大, channel shuffle 所带来的性能提升越高  

## 4.2. Comparison with Other Structure Units

这个年份 (2017) 有名的网络结构都是专注于精度, 计算量都是 1GFlops 以上  

作者通过调整通道数, 来在维持计算量相同的情况下与其他的网络模块做比较  (那看来 无法证明shufflenet 的性价比无法完全高于高精度unit, 即计算量不平等的情况下实现 精度又高同时速度也快)

在完全相同的训练参数下 (没有考虑到网络的收敛速度不同的问题) 得出 shuffnet 的精度最高 with significant margin  

在这之外的发现, 除了 ShuffleNet 以外, 其他几个模型的精度也和通道数成正相关, 这意味着 高效率的网络设计允许更多的 特征通道, 而特征通道数量对于精度是最关键的  

该章节 中文章提供了 500MFLOPs 的精度, 胜过了 MobileNet

## 4.3. Comparison with MobileNets and Other Frameworks

阐述了 MobileNet 的 idea 是源于 Xception 取得了小模型的 SOTA

该章节的表证明了 ShuffleNet 在各方面都优于 MobileNet
* 即使是在相对高计算量的情况下  
* 通过调整通道数, 在计算量不变的情况下, 将 ShuffleNet 的深度50降低到 MobileNet 的同等水平 28, 也精度更高
说明 ShuffleNet 从结构本身上的效率高于 MobileNet, 并不单单是因为在相同的计算量下能够实现更深的层数


作者还阐述了 ShuffneNet 的结构本身是可以和其他 idea 结合的  
和 Squeeze-and-Excitation (SE) 结合可以更进一步提高精度  

## Generalization Ability


作者认为 ShuffleNet 的结构简单, 更适用于 TransferLearning


## Actual Speedup Evaluation

在 ARM 的高通骁龙820 上进行了实际测试  单线程运行 (应该用的是CPU?)  

因为缓存击中率的问题, g 并不是越大越好, 实验性的 g=3 是取得了最好的权衡   




