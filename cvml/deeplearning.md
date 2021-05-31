- [1. Neuron Network 优化方法](#1-neuron-network-优化方法)
  - [1.1. Activate Function](#11-activate-function)
  - [1.2. LRN 和 BN](#12-lrn-和-bn)
    - [1.2.1. LRN](#121-lrn)
    - [1.2.2. BN](#122-bn)
  - [1.3. Attention Mechanism](#13-attention-mechanism)
- [2. Data Augmentation](#2-data-augmentation)
  - [2.1. Reference](#21-reference)
  - [2.2. Traditional](#22-traditional)
  - [2.3. Geometric / Spatial](#23-geometric--spatial)
  - [2.4. 融合 augmentation 参数和网络参数](#24-融合-augmentation-参数和网络参数)
- [3. Dataset](#3-dataset)
  - [3.1. Text spot](#31-text-spot)
    - [3.1.1. Scene Text](#311-scene-text)
    - [3.1.2. Handwritten Text](#312-handwritten-text)
- [4. 网络结构](#4-网络结构)
  - [4.1. Recognition 分类网络 Backbone](#41-recognition-分类网络-backbone)
    - [4.1.1. Alexnet](#411-alexnet)
    - [4.1.2. VGGNet](#412-vggnet)
    - [4.1.3. GoogLeNet](#413-googlenet)
    - [4.1.4. ResNet](#414-resnet)
  - [4.2. Detection 系列网络](#42-detection-系列网络)
    - [4.2.1. R-CNN](#421-r-cnn)
    - [4.2.2. SPP-Net](#422-spp-net)
    - [4.2.3. Fast-RCNN](#423-fast-rcnn)
    - [4.2.4. FasterRCNN RPN](#424-fasterrcnn-rpn)
  - [4.3. 语义分割 Semantic Segmentation](#43-语义分割-semantic-segmentation)
    - [4.3.1. FCN - Fully Convolutional Networks](#431-fcn---fully-convolutional-networks)
  - [4.4. FPN - Feature Pyramid Networks](#44-fpn---feature-pyramid-networks)
- [5. RNN](#5-rnn)
# 1. Neuron Network 优化方法

* ReLU ( Rectified Linear Units) 在 Alexnet 中被发扬光大, 被证明在深层网络中远比 tanh 快, 成功解决了Sigmoid在网络较深时的梯度弥散问题
* Dropout 在 Alexnet 被实用化, 验证了其避免模型过拟合的效果, 在 Alexnet 中主要是最后几个全连接层使用了 Dropout
* MaxPool 在 Alexnet 中被发扬光大, 避免了平均池化的模糊效果, 并且池化核的步长比核的尺寸小, 让池化层的感受野有重叠, 提高了特征的丰富性


## 1.1. Activate Function

* Sigmoid
  * 1/(1-e^(-z))
  * 计算梯度
    * a=sig(z)
    * dz=a(1-a)
* tanh
  * (e^(z)-e^(-z))/(e^(z)+e^(-z))
  * 计算梯度
    * a=tanh(z)
    * dz=1-a*a

* 梯度消失在 Sigmoid 和 tanh 中都会发生

选择激活函数的规则
* 如果是 Binary Classification
  * 只在输出层用 Sigmoid 函数
  * 其他层用 ReLU 或者 Leaky ReLU
* 

## 1.2. LRN 和 BN

* LRN : 局部响应归一化 ( Local Response Normalization )
* BN  : 批量归一化     ( Batch Normalization )

归一化是深度神经网络中的重要步骤
* 弥补 ReLU ELU 等函数的无界性问题
  * 输出层的值不会被限制在范围中 (tanh的 (-1,1)), 可以根据需要尽可能高的增长
* 为了在一定程度上限制激活函数的输出值

### 1.2.1. LRN

* Alexnet 架构中被引入, 一个不可训练的层
* 为了鼓励 横向抑制 ( lateral inhibition )
  * 组织兴奋神经元向其临近的神经元传递其动作趋势
  * 目的是进行局部对比度增强, 以便使局部最大像素值作用于下一层的激励
* 在VGG的原始论文中提到了 LRN 基本没什么作用, 只会增加计算量


* 通道间的 LRN ( Inter-Channel LRN ) , 在 Alexnet 中被使用, 定义的邻域在整个通道上, 即对于 x,y 位置, 在 RGB 维度上进行归一化
![LRN公式](https://images4.pianshen.com/914/5f/5f02a0b2599665e5d8111caddd4676a2.png)
$b_{x,y}^i=a_{x,y}^i/(k+\alpha\sum_{j=max(0,i-n/2)}^{min(N-1,i+n/2)}(a_{x,y}^j)^2)^\beta$

* 通道内的LRN ( Intra-Channel LRN ), 邻域仅在同一 channel 中进行拓展
![LRN公式](https://images3.pianshen.com/609/2b/2b636b49e4e3fab6520a4adbdc05e489.png)
$b_{x,y}^k=a_{x,y}^k/(k+\alpha\sum_{i=max(0,x-n/2)}^{min(W,x+n/2)}\sum_{j=max(0,y-n/2)}^{min(H,y+n/2)}(a_{i,j}^k)^2)^\beta$
* 希腊字符 ( k,α,n,β ) 用于定义 LRN 的超参数
  * k 用于避免分母为零的常数
  * α 归一化常数
  * β 对比度常数
  * n 用于定义归一化的邻域长度
  
### 1.2.2. BN

* Batch Normalization, 同LRN不同, 这是一个可以被训练的层
  * 该层主要用于解决内部协变量偏移 ( Internal Covariate Shift ICS )
  * 如果不同的Batch的特征有不同的分布,即batchA有很多A类, batchB有很多B类, 会减慢学习的速率
    * 在训练的时候应该尽可能确保每个batch中的成员不属于相同的类别, 拥有同等数量的 A和B
  * 但是对于隐藏的神经元来说, 就算 Batch是随机选择的, 也可能会有协变量偏移
    * 因为隐藏层的协变量偏移不可直接控制
    * 通过使用 BN 来减轻这种情况

* BN层的操作: 隐藏神经元的输出按如下处理, 再传入激活函数
  * 将整个 batch 归一化为0均值和单位方差
    * 计算 mini-batch 输出的平均值
    * 计算整个 mini-batch 输出的方差 $\sigma$
    * 通过减去平均值然后除以方差来归一化 mini-batch
  * 两个可训练的参数 gamma 和 beta
    * 代表 缩放和移动, 通过训练来最终确定是否需要归一化, 因为存在不归一化会有更好的结果的情况
    * $\hat{x}$ 代表完成0均值和单位方差归一化后的输出
    * $y_i=\gamma\hat{x}+\beta\equiv BN_{\gamma,\beta}(x_i)$

## 1.3. Attention Mechanism

* Attention 从出发点上说是用于提升基于 RNN 的S2S模型效果的机制
* 目前广泛应用于机器翻译, 语音识别, 图像标注等领域
* Attention 给模型赋予了 区分辨别的能力, 使得网络模型的学习变得更加灵活
* Attention本身可以作为一种对齐的关系, 可以用来解释翻译输入输出句子之间的对其关系


总结:
  - Attention 可以帮助模型对输入的X的每个部分赋予不同的权重, 抽出更加关键的信息
  - 并不会对模型的计算和存储带来更大的开销

# 2. Data Augmentation

Data augmentation is a low cost way.

## 2.1. Reference

1. Jaderberg, M., Simonyan, K., Vedaldi, A., & Zisserman, A. (2014). Synthetic Data and Artificial Neural Networks for Natural Scene Text Recognition. Computer Vision and Pattern Recognition. http://arxiv.org/abs/1406.2227
2. Luo, C., Zhu, Y., Jin, L., & Wang, Y. (2020). Learn to augment: Joint data augmentation and network optimization for text recognition. Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition, 13743–13752. https://doi.org/10.1109/CVPR42600.2020.01376

## 2.2. Traditional 

Traditional augmentation methods such as rotation, scaling and perspective transformation,

## 2.3. Geometric / Spatial

## 2.4. 融合 augmentation 参数和网络参数


# 3. Dataset

## 3.1. Text spot

3755 classes (Ievel-l set of GB2312-80)  ｄ
### 3.1.1. Scene Text
* IIIT 5K-Words  (IIIT5K) contains 3000 cropped word images for testing.
* Street View Text (SVT) consists of 647 word images for testing. Many images are severely corrupted by noise and blur.
* Street View Text Perspective (SVT-P) contains 645
cropped images for testing. Most of them are perspective distorted.
* ICDAR 2003 (IC03) contains 867 cropped images after discarding images that contained non-alphanumeric characters or had fewer than three characters.
* ICDAR 2013 (IC13) inherits most of its samples from IC03. It contains 1015 cropped images.
* ICDAR 2015 (IC15) is obtained by cropping the words using the ground truth word bounding boxes and includes more than 200 irregular text images.

### 3.1.2. Handwritten Text

* IAM contains more than 13,000 lines and 115,000 words written by 657 different writers.
* RIMES contains more than 60,000 words written in French by over 1000 authors. 4.3.

# 4. 网络结构 

* 简单粗暴的方法来提高精度就是加深网络以及加宽网络, 缺点有
  * 容易过拟合
  * 迅速增加的参数个数以及大量为0的无效神经点带来的浪费计算, 难以应用
  * 网络越深, 越容易梯度弥散
    * 即使使用了 BN, 几十层的CNN也非常难训练
* 基础的方法是增加网络的稀疏性, 尽可能减少全连接层


## 4.1. Recognition 分类网络 Backbone

* AlexNet
  * 推广了 ReLU 和 Dropout
* VGGnet 证明了网络深度与高级 feature 之间的关系
  * 因为网络深度, 训练时间更长, 但是收敛的迭代次数减小
  * 在识别和检测中都有较好效果
* GoogLeNet 彻底摆脱了全连接层
  * AlexNet 和 VGGNet 中的FC层占据了9成的参数, 而且会导致过拟合
  * 使用模块化的 Inception 来抽取特征

原始论文:
0. (1998)Gradient-based learning applied to document recognition
1. 

### 4.1.1. Alexnet

* ReLU 激活函数被应用在了所有卷积层和FC层的后面

PS : 更新知识点 
* Alex Krizhevsky在2014年发表《One weird trick for parallelizing convolutional neural networks》。
* Pytorch官方给出的AlexNet模板是基于2014年这篇论文。

前5层卷积层的网络结构:
* 输入:  原论文中是 `224*224*3` 的图片, 经过`迷之` padding 到 227
* Conv1 `kernel=11 stride=4 padding=2` 
  * 原论文中的 channel 是 96, 分到了 2 台GPU上, 每台 48
  * Pytorch 预定义的 channel 是 64
  * 输出 `55*55*(96 or 64)`
* ReLU1层
* 池化层 MaxPool ` kernel=3 stride=2 padding=0 `
  * 输出 `27*27*(96 or 64)`
* Conv2 `kernel=5 stride=1 padding=2`
  * 原论文中的 channel 是 256, 分到了 2台GPU上, 每台 128 接受前一步的 48 channel
  * Pytorch 预定义的是 192
  * 输出 `27*27*(256 or 192)`
* ReLU2层
* 池化层 MaxPool `keynel=3 stride=2 padding=0`
  * 输出 `13*13*(256 or 192)`
* Conv3 `kernel=3 stride=1 padding=1`
  * 原论文的 channel 是 384, 分到 2台GPU每台 192, 注意这一步每台 GPU 接受了前一步所有的 256 channel, 即一台GPU上的 kernel维度是 (256,192)
  * Pytorch 中也定义的 384
  * 输出 `13*13*384`
* ReLU3层
* Conv4 `kernel=3 stride=1 padding=1`
  * 原论文中的这一步仍然是2台GPU各自卷积, 分别各自输入 192 输出 192
  * Pytorch 预定义的 256
  * 输出 `13*13*(384 or 256)`
* ReLU4层
* Conv5 `kernel=3 stride=1 padding=1` 开始降维
  * 原论文中 的channel 是256, 分到 2台GPU每台128, 各自接受前一步的 192
  * Pytorch 预定义的是 256
  * 输出 `13*13*256`
* ReLU5层
* MaxPool `kernel=3 stride=2 padding=0`
  * 输出 `6*6*256`
* AdaptiveAvgPool2d((6, 6))
后三层FC层的网络结构
* 输入: `6*6*256` 
* (Dropout) FC层: 输入 9216 输出 4096
* ReLU
* (Dropout) FC层: 输入 4096 输出 4096
* ReLU
* (Dropout) FC层: 输入 4096 输出 1000 ( class_num )

```py
# Pytorch 源代码
class AlexNet(nn.Module):

    def __init__(self, num_classes: int = 1000) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
```
### 4.1.2. VGGNet

* 网络模型
  * 输入大小为 224*224 
  * 预处理只有减去均值
  * 所有的卷积核都是3*3, 步长=1, padding=1
  * 池化层固定为5层, 窗口2*2, 步长=2
  * 三层全连接层, 两层4096, 最后一层为分类层
* 特点
  * 深的同时结构简单
  * 1*1 的卷积层不影响感受野, 看起来是线性的, 但是因为有激活函数ReLU使得实际的运算是非线性 
* 参数初始化
  * 渐层网络随机初始化 N(0,0,01), 深层网络用浅层网络的参数初始化
* 数据增强
  * 先对训练图片进行随机缩放, 再随机裁剪
  * 再加上随机水平翻转和色彩偏移


### 4.1.3. GoogLeNet

GoogLeNet的发展共有四个版本  

GoogLeNet 提出了 Inception Module 的概念, 应用了 Network In Network 的思想  
* 推动深度学习的继续发展, 而不是"深度调参"
* Inception 结构:
  * 用密集成分来近似最优的局部稀疏结构
  * 每一个 Module 拥有四个分支
    * 3个不同的卷积核大小加上3*3 max pooling
    * `5*5` 卷积层的参数过多
    * 改进后的模型用`1*1` 的卷积核来对 `5*5` 卷积层降维, 减少参数
  * 四个分支最后通过一个聚合操作合并


GoogLeNet的结构
* 使用模块化的 Inception 构成
* 网络的最后使用平均池化来代替全连接层
* 就算没有全连接层, 也使用了 dropout
* 额外增加了两个辅助的 softmax 用于前向传导梯度


Inception V2:
* 提出了著名的 Batch Normalization
  * 可以将学习速率提高很多倍
  * 使用 BN时, sigmoid激活函数比ReLU效果好
* 参考了 VGG, 用两个 3X3 代替了 5X5

Inception V3:
* 给予了一些设计网络结构的通用准则
* 避免表达瓶颈, 
  * 过于快速的降低特征图的维度会导致丢失较多信息, 产生瓶颈, 使用复杂的池化来保留更多的信息
  * 在低维的特征可以放心的进行降维, 而不用担心产生严重的后果
* 高维特征信息包含的更多, 更容易加快训练


### 4.1.4. ResNet

* Residual Network
* 灵感来源于: 如果使用恒等映射层, 那么网络的加深起码不会带来训练误差的上升
  * 问题: 让网络学习 F(x)= x比较难
  * 解决方法: 让网络学习 F(x)=0, 让最终的输出为 H(x)=F(x)+x
* 残差: 观测值与估计值之间的差
  * 求解残差映射函数 $F(x)=H(x)-x\quad or \quad H(x)=F(x)+x$
  * H(x) : 观测值, 当前残差层的最终输出特征
  * x    : 估计值, 即上一层 ResNet输出的特征映射, 当前深度的最优解的特征
  * 目标 : F(x)=0, 即 H(x)=x ,代表确保错误率起码不上升
  * 升维 : F(x) 和 x 的维度不同的话是没法相加的
    * HW维度升维用 padding
    * channel 升维用 1X1 卷积
* 结果: 网络可以上升到百层
  * 改进的网络可以上升到千层



## 4.2. Detection 系列网络

Detection 相比 Recognition 是更复杂的任务:
1. Detection 的过程中往往也就顺带进行了分类识别
2. Detection 的特征提取往往更复杂:
   * 目前的手法大多数是输入一个区域, 区域经过CNN计算, 得到该区域是否是正区域来得到结果
   * 以前的手法更多的使用 sliding windows, 窗口移动方法, RPN 网络也是基于 Sliding-windows
   * R-CNN 的出现引领了 Region-Based 的风潮

针对速度和精度的权衡 trade-off between speed and accuracy:
- Multi-scale Feature 即使在深度学习的时代, 依然能够带来更高的精度
- 如何快速的计算出 Multi-scale Feature 也是一个研究方向. eg. FPN  


* Regions with CNN feature 是将CNN方法应用到目标检测的方法, 是目标检测问题的一个里程碑.  
* SPP-Net 为Fast-RCNN提供了灵感
* RPN网络分支为加速训练提供了候选区域筛选方法, Faster RCNN 基本实现了端到端的检测
* YOLO 直接舍弃了 RPN的候选框提取, 直接进行整图回归
* SSD 解决了 YOLO的目标定位偏差问题和小目标遗漏问题  

原始论文:
1. (2013)Selective search for object recognition 
2. (2014)Rich feature hierarchies for accurate object detection and semantic segmentation 
3. (2014)Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition
4. (2017)Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks

### 4.2.1. R-CNN

R-CNN算法可以分为四步:
1. 候选区域选择, 即 Region Proposal
2. 使用CNN提取候选区域的特征
3. 对CNN提取的特征向量进行分类
4. 边界回归得到精确的目标区域


缺陷:
* 多个候选区域 (2K) 需要预先提取, 非常占用磁盘
* CNN需要固定尺寸输入, 因此对候选区域的拉伸或截取会丢失信息
* 候选区域存在重叠, 因此计算浪费

### 4.2.2. SPP-Net

由其他作者对RCNN的一个实质性的改进  
1. 取消了 crop/warp 的图像归一化过程, 解决了图像变形导致的信息丢失
2. 用 空间金字塔池化 ( Spatial Pyramid Polling ) 替换了全连接层之前的最后一个池化层  
3. 解决了卷积层的重复计算问题 速度提高了 24~102 倍

缺陷:
1. 和RCNN一样，训练过程仍然是隔离的，提取候选框 | 计算CNN特征| SVM分类 | Bounding Box回归独立训练，大量的中间结果需要转存，无法整体训练参数；
2. SPP-Net在无法同时Tuning在SPP-Layer两边的卷积层和全连接层，很大程度上限制了深度CNN的效果；
3. 在整个过程中，Proposal Region仍然很耗时。


### 4.2.3. Fast-RCNN

主要贡献在于对RCNN进行加速, 受SPP-Net启发而来  
1. 借鉴了SPP-Net的思路, 提出了简化版的 ROI池化层 (注意并非金字塔), 加入了候选框映射功能, 使得网络可以反向传播, 解决了SPP的整体网络训练问题  
2. 多任务Loss, SoftmaxLoss代替了SVM, 证明了Softmax比SVM有更好的效果
3. SVD加速全连接层的训练
4. 

### 4.2.4. FasterRCNN RPN

Faster RCNN 基本实现了端到端的实时检测  


RPN网络分支:
* RPN作为一个分支网络模块, 完美的解决了 Selective Search 提取候选区域慢的问题  
* RPN作为网络分支, 和 backbone 时别网络共享权值, 一起训练
* 参数:  n是 slide windows 的边长, k 是锚点box的个数

1. 输入任意大小的图像, 输出一组矩形候选区域和对应的 objectness score
2. 和最初RCNN的Selective Search 不同, RPN是滑窗搜索 (Slide Windows) 论文中滑窗大小 3X3
3. 和滑窗搜索一起出现的一个新名词 - 锚点 (anchor box) 

RPN网络过程
1. RPN 作用于共享网络的最后一层 (feature map output by the last shared conv layer), 这最后一层网络输出的特征图的WH先不管, 通道是 256(ZFnet) 或 512 (VGG)
2. RPN 网络在一个位置, 产生 k 个候选区域, 经过参数化后对应 k 个 anchors, 若 feature map 是WXH, 那么总共的 anchors 为 WHk个



## 4.3. 语义分割 Semantic Segmentation

图像分割: 让网络做像素级别的预测, 直接得出label map  

1. (2017)Fully Convolutional Networks for Semantic Segmentation



### 4.3.1. FCN - Fully Convolutional Networks

深度学习应用在图像分割的代表成就
* FCN 接受任意尺寸的输入图像
* 对最后一个卷积层的输出feature map 使用反卷积进行上采样, 恢复到原本输入图像的尺寸
* 对每一个像素都产生了一个预测, 同时保留了原始输入图像中的空间信息
* end-to-end, pixels-to-pixels

概念: 全连接层与全卷积层的转换
- 任意一个卷积层都可以又一个全连接层代替 (该全连接层的权重矩阵中由很多0, 有很多重复模块)
- 任何一个全连接层都可以转化为一个卷积层 (卷积核的大小=上层特征大小, 卷积核的个数=下层特征大小)

全连接层->全卷积层:
* 将 1000 个节点的全连接层改成了 1000个 1X1卷积核的卷积层
* Feature map 经过该卷积层, 还是得到二维的 Feature map, 因此对输入图片的 size 没有了限制
* 例: 全连接层 4096->4096->1000, 对应的卷积核 (4096,1,1)-> (4096,1,1)-> (1000,1,1)

## 4.4. FPN - Feature Pyramid Networks

论文: Feature pyramid networks for object detection  
一种获得多缩放特征( Multi-scale Feature )的方法, 独立于 Backbone 网络.  

结构定义:
* 背景和概念:
  * 输入一个 single-scale 图像, 任意大小, 得到一系列成比例大小的特征图
  * FPN由三部分组成, 自底向上路径, 自顶向下路径, 水平连接 三部分
  * 定义输出特征图的大小相同的网络layers为同一个stage, 整个网络因此被分成了几个stage
  * 依据各个stage输出的大小, 网络的输出被分成了 几个部分, 特征图的大小依次降低
  * stage1因为输出的size太大了而不使用
* Bottom-up Pathway:
  * 前向传播过程中, 取各个stage最后一层的输出作为 Ci
  * C1不使用
* Top-down Pathway:
  * 从后层的输出开始, 以2为单位进行升采样, 直接使用最近邻方法来进行填充
* Lateral connection:
  * 使用 1X1 卷积核对 Bottom-up 进行通道降维
  * 直接将 BU和TD的特征图进行元素层面的加法
  * 最后一层C5不用进行merge, 直接降维后输出
  * 在merge后的特征图加一个3X3的卷积层, 用于消除升采样带来的误差
  * 最终获得了 P2~P5 的特征图

# 5. RNN

* 普通的神经网络以及CNN, 都是假设元素之间是相互独立的, 输入与输出也是独立的
* 循环神经网络的特点: 拥有记忆能力, 输入依赖于当前的输出和记忆

定义:
* Xt 表t时刻的输入
* Ot 表t时刻的输出: $o_t=softmax(VS_t)$
  * $V$ 即记忆的权重矩阵, 表示根据当前的记忆来决定输出
  * 很多任务并不需要中间的输出, 只关注最后的输出
* St 表t时刻的记忆: $S_t=f(U*X_t+W*S_{t-1})$  
  * $f()$是激活函数
  * 可以把St当作一个隐状态
* 网络中的所有细胞共享参数 $UVW$



























