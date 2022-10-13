- [1. Deep Learning 概念](#1-deep-learning-概念)
  - [1.1. 卷积](#11-卷积)
    - [1.1.1. Padding 策略](#111-padding-策略)
- [2. 深度学习的训练 - 优化](#2-深度学习的训练---优化)
  - [2.1. 基本概念](#21-基本概念)
    - [2.1.1. 基本算法 SGD](#211-基本算法-sgd)
    - [2.1.2. 动量 momentum](#212-动量-momentum)
  - [2.2. 自适应学习率算法](#22-自适应学习率算法)
    - [2.2.1. AdaGrad](#221-adagrad)
    - [2.2.2. RMSProp 和 Adadelta](#222-rmsprop-和-adadelta)
    - [2.2.3. Adam](#223-adam)
  - [2.3. Activate Function 激活函数](#23-activate-function-激活函数)
  - [2.4. LRN 和 BN](#24-lrn-和-bn)
    - [2.4.1. LRN](#241-lrn)
    - [2.4.2. BN](#242-bn)
  - [2.5. 训练的配置](#25-训练的配置)
    - [2.5.1. AlexNet 训练](#251-alexnet-训练)
    - [2.5.2. VGG 训练](#252-vgg-训练)
    - [2.5.3. ResNet 训练 ImageNet](#253-resnet-训练-imagenet)
    - [2.5.4. CRNN](#254-crnn)
- [3. Data Augmentation](#3-data-augmentation)
  - [3.1. Reference](#31-reference)
  - [3.2. Traditional](#32-traditional)
  - [3.3. Geometric / Spatial](#33-geometric--spatial)
  - [3.4. 融合 augmentation 参数和网络参数](#34-融合-augmentation-参数和网络参数)
- [4. 网络结构 与模型](#4-网络结构-与模型)
  - [4.1. Backbone网络](#41-backbone网络)
    - [4.1.1. Alexnet](#411-alexnet)
    - [4.1.2. VGGNet](#412-vggnet)
    - [4.1.3. GoogLeNet](#413-googlenet)
    - [4.1.4. ResNet](#414-resnet)
    - [4.1.5. MobileNet 轻量化网络](#415-mobilenet-轻量化网络)
      - [4.1.5.1. MobileNetV1](#4151-mobilenetv1)
      - [4.1.5.2. MobileNetV2](#4152-mobilenetv2)
  - [4.2. Detection 系列网络](#42-detection-系列网络)
    - [4.2.1. R-CNN](#421-r-cnn)
    - [4.2.2. SPP-Net](#422-spp-net)
    - [4.2.3. Fast-RCNN](#423-fast-rcnn)
    - [4.2.4. FasterRCNN RPN](#424-fasterrcnn-rpn)
  - [4.3. 语义分割 Semantic Segmentation](#43-语义分割-semantic-segmentation)
    - [4.3.1. FCN - Fully Convolutional Networks](#431-fcn---fully-convolutional-networks)
  - [4.4. FPN - Feature Pyramid Networks](#44-fpn---feature-pyramid-networks)
- [5. 序列建模 - 循环和递归网络 RNN](#5-序列建模---循环和递归网络-rnn)
  - [5.1. Recurrent Neural Networks (RNN)](#51-recurrent-neural-networks-rnn)
  - [5.2. Long Short-Term Memory LSTM](#52-long-short-term-memory-lstm)
- [6. Attention](#6-attention)
- [7. Multi Object Tracking (MOT)](#7-multi-object-tracking-mot)
- [8. Transfer Learning](#8-transfer-learning)
  - [8.1. Domain Adaptation (DA)](#81-domain-adaptation-da)
  - [8.2. Domain Generalization](#82-domain-generalization)
- [9. View Synthesis](#9-view-synthesis)
  - [9.1. Neural Rendering  的各种渲染方法](#91-neural-rendering--的各种渲染方法)
    - [9.1.1. Volume 体数据  体渲染](#911-volume-体数据--体渲染)
  - [9.2. NeRF Neural Radiance Fields](#92-nerf-neural-radiance-fields)
    - [9.2.1. Vanilla NeRF](#921-vanilla-nerf)
  - [Neural 3D shape representations](#neural-3d-shape-representations)
- [10. Dataset](#10-dataset)
  - [10.1. Text spot](#101-text-spot)
    - [10.1.1. Scene Text](#1011-scene-text)
    - [10.1.2. Handwritten Text](#1012-handwritten-text)


# 1. Deep Learning 概念

## 1.1. 卷积

### 1.1.1. Padding 策略

* padding 是填充零的处理
* 在实现中不同框架对 padding 的处理是不同的

一般情况下 ：
* padding时, 一般是对称地补, 左／右各padding一列 或者 上下各padding一行
* 计算公式 `output_shape = (image_shape-filter_shape+2*padding)/stride + 1`


对于 stride 是偶数, 而输入维度与卷积核的插值是奇数, 无法除尽的情况下:
* 一般的处理是再次补一列 padding, 使得除法的结果能够向上取整: 
  * caffe偷偷摸摸地把一行0补在上面 或者 把一列0补在左边
  * tensorflow正好镜像对称，把一行0补在下面或者把一列0补在右边
* 对于 Pytorch 
  * padding 不存在隐式补零的情况, 输出维度会向下取整
  * 相比tensorflow，PyTorch需要用户清楚的知道的自己的卷积核选取对结果的影响



# 2. 深度学习的训练 - 优化

相关名词:
* 目标函数 objective function = 准则 criterion = 代价函数 cost function = 损失函数 loss function = 误差函数 error function
* 导数为0 : 临界点 critical point = 驻点 stationary point
  * 或者梯度为0
* 梯度 gradient : 是一个多为函数 f 的所有偏导数的向量
* 梯度下降      : 沿着梯度的负方向移动 函数参数, 可以被称为 最速下降 method of steepest descent


* ReLU ( Rectified Linear Units) 在 Alexnet 中被发扬光大, 被证明在深层网络中远比 tanh 快, 成功解决了Sigmoid在网络较深时的梯度弥散问题
* Dropout 在 Alexnet 被实用化, 验证了其避免模型过拟合的效果, 在 Alexnet 中主要是最后几个全连接层使用了 Dropout
* MaxPool 在 Alexnet 中被发扬光大, 避免了平均池化的模糊效果, 并且池化核的步长比核的尺寸小, 让池化层的感受野有重叠, 提高了特征的丰富性


## 2.1. 基本概念 

### 2.1.1. 基本算法 SGD

* SGD stochastic gradient descent - 随机梯度下降
  * 相比于正规的梯度下降需要计算整个训练集上的平均损失, 然后计算梯度
  * 最早的 SGD 是对每一个样本单独计算梯度并立即更新
  * 后来提出了 minibatch 的概念
  * 现在通用的基本上指代 应用 minibatch 的SGD, 表示仅使用 minibatch 的训练样本即可更新一次参数
* 在实际使用中
  * 我们一般会设置线性衰减学习率 $\epsilon_k = (1-\alpha)\epsilon_0+\alpha\epsilon_\tau$
  * 即在 $\tau$ 次迭代后, 学习率保持常数不再下降

算法过程
* 参数: 学习率 $\epsilon_k$ , 初始化的网络参数 $\theta$
* while 循环条件 do
* ---- 采集 minibatch 数据 $\{x^{(1)},...,x^{(m)}\}$
* ---- 计算梯度 $\hat{g} \leftarrow +\frac{1}{m}\nabla_\theta \sum_iL(f(x^{(i)};\theta),y^{(i)})$
* ---- 更新参数 $\theta \leftarrow \theta - \epsilon \hat{g}$
* end while

### 2.1.2. 动量 momentum

动量的概念是为了加速学习
* 动量会积累 : 之前迭代梯度的 指数级衰减的 移动平均
* 动量的表示 : $\upsilon$ 代表速度, 表示参数空间的 梯度方向和速率
* 动量在大部分情况都设置成 0.9

带梯度的 SGD算法过程
* 参数: 学习率 $\epsilon_k$, 动量参数 $\alpha$, 初始化的网络参数 $\theta$, 初始速度 $v$
* while 循环条件 do
* ---- 采集 minibatch 数据 $\{x^{(1)},...,x^{(m)}\}$
* ---- 计算梯度 $g \leftarrow +\frac{1}{m}\nabla_\theta \sum_iL(f(x^{(i)};\theta),y^{(i)})$
* ---- 计算速度 $\upsilon \leftarrow \alpha\upsilon-\epsilon g$
* ---- 更新参数 $\theta \leftarrow \theta + \upsilon$
* end while
* 拆开来看 带动量就是更新参数的时候既用梯度又用速度 
  * $\theta \leftarrow \theta -\epsilon g+ \alpha\upsilon$
  * $\upsilon \leftarrow \alpha\upsilon-\epsilon g$



Nesterov accelerated gradient(NAG): 带有预知的动量
* Nesterov 动量对于RNN的学习有很大的帮助, 和普通动量一样常常和其他优化算法进行结合
* 参数: 学习率 $\epsilon_k$, 动量参数 $\alpha$, 初始化的网络参数 $\theta$, 初始速度 $v$
* while 循环条件 do
* ---- 采集 minibatch 数据 $\{x^{(1)},...,x^{(m)}\}$
* ---- 根据动量计算临时更新后的参数 $\tilde{\theta}\leftarrow \theta + \alpha v$
* ---- 在临时点计算梯度 $g\leftarrow +\frac{1}{m}\nabla_{\tilde{\theta}}\sum_iL(f(x^{(i)};\tilde{\theta}),y^{(i)})$
* ---- 计算速度 $\upsilon \leftarrow \alpha\upsilon-\epsilon g$
* ---- 正式更新参数 $\theta \leftarrow \theta + \upsilon$


## 2.2. 自适应学习率算法

基于 minibatch 的自适应学习率算法是目前应用最广的算法

* 自适应算法的核心是不再拥有全局统一的学习率, 而是根据每一个单独参数的梯度历史来动态调整对应参数自己的学习率
* AdaGrad 几乎不需要手动调整学习速率, 因为真正的学习速率都是每次迭代重新计算的, 用默认的 0.01即可, 缺点就是容易学习率降低太快
* Adadelta 和 RMSProp 是独立被研究的, 都是为了解决 AdaGrad学习率降速的问题
  * RMSProp 推荐 学习率 0.001, 梯度累计衰减率 0.9
  * Adadelta 在 RMSProp 的基础上再加入了参数更新量的累计, 直接省略了初始学习率的超参数

### 2.2.1. AdaGrad

AdaGrad : (2011)有较好的理论性质, Ian 的书中表示经验上 AdaGrad 实际上会导致 有效学习率过早和过量的减少(一直在累加梯度)
* 适合应用在稀疏数据上
* 对非频繁的参数应用大更新, 对频繁的参数应用小更新, 具体表现在对于参数 $\theta$ , 不再使用统一的学习率 $\epsilon$ , 而是对于每一个单独参数在每一次参数迭代应用单独的学习率
* 参数: 全局学习率 $\epsilon_k$, 初始化的网络参数 $\theta$
* 用于除数非零的一个常数 $\delta =10^{-7}$
* 梯度累计变量 $r=0$
* while 循环条件 do
* ---- 采集 minibatch 数据 $\{x^{(1)},...,x^{(m)}\}$
* ---- 计算梯度 $g \leftarrow +\frac{1}{m}\nabla_\theta \sum_iL(f(x^{(i)};\theta),y^{(i)})$
* ---- 累计梯度的平方 $r\leftarrow r+g\odot g$
* ---- 计算更新 $\Delta\theta\leftarrow -\frac{\epsilon}{\delta+\sqrt{r}}\odot g$
* ---- 应用更新 $\theta \leftarrow \theta+\Delta\theta$

### 2.2.2. RMSProp 和 Adadelta

RMSProp : (2012)基于 AdaGrad 算法, 相比于一直累计梯度, RMSProp 指数级丢弃遥远梯度, 有效最常见算法之一
* 参数: 全局学习率 $\epsilon_k$, 初始化的网络参数 $\theta$
* 用于除数非零的一个常数 $\delta =10^{-6}$
* 梯度累计变量 $r=0$ , 梯度累计的舍弃速度 $\rho =0.9$
* while 循环条件 do
* ---- 采集 minibatch 数据 $\{x^{(1)},...,x^{(m)}\}$
* ---- 计算梯度 $g \leftarrow +\frac{1}{m}\nabla_\theta \sum_iL(f(x^{(i)};\theta),y^{(i)})$
* ---- 累计梯度的平方 $r\leftarrow \rho r+(1-\rho)g\odot g$
* ---- 计算更新 $\Delta\theta\leftarrow -\frac{\epsilon}{\delta+\sqrt{r}}\odot g$
* ---- 应用更新 $\theta \leftarrow \theta+\Delta\theta$

可以看出相比于 AdaGrad, RMSProp 只进行了很小的改变, 就是对梯度的累计进行了指数衰减

Adadelta : (2012) 同样是为了解决 AdaGrad 莽撞的降低学习速率的设定, 在累积了历史梯度的同时, 还附加累计了参数的历史更新量
* 参数: 初始化的网络参数 $\theta$, 完全不用设置初始学习速率
* 用于除数非零的一个常数 $\delta =10^{-6}$
* 梯度累计变量 $r=0$, 参数更新量的累计变量 $t=0$ , 两个累计的舍弃速度 $\rho =0.9$
* while 循环条件 do
* ---- 采集 minibatch 数据 $\{x^{(1)},...,x^{(m)}\}$
* ---- 计算梯度 $g \leftarrow +\frac{1}{m}\nabla_\theta \sum_iL(f(x^{(i)};\theta),y^{(i)})$
* ---- 累计梯度的平方 $r\leftarrow \rho r+(1-\rho)g\odot g$
* ---- 用新的梯度累计和旧的参数更新累计来计算参数更新 $\Delta\theta\leftarrow -\frac{\delta+\sqrt{t}}{\delta+\sqrt{r}}\odot g$
* ---- 累计更新量的平方 $t=\rho t +(1-\rho)\Delta\theta^2$
* ---- 应用更新 $\theta \leftarrow \theta+\Delta\theta$

### 2.2.3. Adam

Adaptive Moment Estimation(Adam) : (2014)引入梯度的一二阶矩, 对超参数的设置非常鲁棒  
Adam 用另一种方式实现了动量, 即将动量表示成梯度的一阶矩累计(相当于学习率=1 的动量)  

* 参数: 全局学习率 $\epsilon_k=0.001$, 初始化的网络参数 $\theta$
* 用于除数非零的一个常数 $\delta =10^{-8}$
* 两个梯度累计变量 $r=0 s=0$
* 两个梯度累计的舍弃速度 $\rho_1 =0.9, \rho_2=0.999$
* 时间戳 $t$
* while 循环条件 do
* ---- 采集 minibatch 数据 $\{x^{(1)},...,x^{(m)}\}$
* ---- 计算梯度 $g \leftarrow +\frac{1}{m}\nabla_\theta \sum_iL(f(x^{(i)};\theta),y^{(i)})$
* ---- 时间戳 $t\leftarrow t+1$
* ---- 累计梯度的一阶 $s\leftarrow \rho_1 s+(1-\rho_1)g$
* ---- 修正一阶累计值 $\hat{s}\leftarrow\frac{s}{1-\rho_1^t}$
* ---- 累计梯度的二阶 $r\leftarrow \rho_2 r+(1-\rho_2)g\odot g$
* ---- 修正二阶累计值 $\hat{r}\leftarrow\frac{r}{1-\rho_2^t}$
* ---- 根据修正值$\hat{s}\hat{r}$计算更新 $\Delta\theta\leftarrow -\frac{\epsilon}{\delta+\sqrt{\hat{r}}}\odot \hat{s}$
* ---- 应用更新 $\theta \leftarrow \theta+\Delta\theta$ 



## 2.3. Activate Function 激活函数

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

## 2.4. LRN 和 BN

* LRN : 局部响应归一化 ( Local Response Normalization )
* BN  : 批量归一化     ( Batch Normalization )

归一化是深度神经网络中的重要步骤
* 弥补 ReLU ELU 等函数的无界性问题
  * 输出层的值不会被限制在范围中 (tanh的 (-1,1)), 可以根据需要尽可能高的增长
* 为了在一定程度上限制激活函数的输出值

### 2.4.1. LRN

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
  
### 2.4.2. BN

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


## 2.5. 训练的配置

涉及的概念:
* 训练多少 epoch 
* batchsize 多大
* 设置初始学习速率, 动量, weight decay
* 配置 学习速度下降策略
* 什么时刻训练终止

### 2.5.1. AlexNet 训练

* SGD batchsize=128
* LR=0.01, 除以10 每当 loss 不再下降
* weight decay=0.0005, 动量 0.9
* 简单的训练 90个 epoches, 数据量 1.2m

### 2.5.2. VGG 训练

* SGD batch size=256
* weight delay 0.0005 动量 0.9
* LR=0.01, 除以10 每当 loss 不再下降
* 最多训练 370K 个迭代

### 2.5.3. ResNet 训练 ImageNet

* 设置 BN
* SGD batchsize=256
* LR=0.1  除以十每当 loss 不再下降
* weight decay=0.0001 , 动量 0.9
* 不使用 dropout
* **最多训练 600k 个迭代**
  
### 2.5.4. CRNN

* ADADELTA 
* 


# 3. Data Augmentation

Data augmentation is a low cost way.

## 3.1. Reference

1. Jaderberg, M., Simonyan, K., Vedaldi, A., & Zisserman, A. (2014). Synthetic Data and Artificial Neural Networks for Natural Scene Text Recognition. Computer Vision and Pattern Recognition. http://arxiv.org/abs/1406.2227
2. Luo, C., Zhu, Y., Jin, L., & Wang, Y. (2020). Learn to augment: Joint data augmentation and network optimization for text recognition. Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition, 13743–13752. https://doi.org/10.1109/CVPR42600.2020.01376

## 3.2. Traditional 

Traditional augmentation methods such as rotation, scaling and perspective transformation,

## 3.3. Geometric / Spatial

## 3.4. 融合 augmentation 参数和网络参数




# 4. 网络结构 与模型

* 简单粗暴的方法来提高精度就是加深网络以及加宽网络, 缺点有
  * 容易过拟合
  * 迅速增加的参数个数以及大量为0的无效神经点带来的浪费计算, 难以应用
  * 网络越深, 越容易梯度弥散
    * 即使使用了 BN, 几十层的CNN也非常难训练
* 基础的方法是增加网络的稀疏性, 尽可能减少全连接层


## 4.1. Backbone网络 

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





### 4.1.5. MobileNet 轻量化网络

* MobileNet 提出了同时注重速度和模型大小的网络结构, 更贴合实际应用


#### 4.1.5.1. MobileNetV1

* 提出了两个超参数用来管理网络结构 : width multiplier, resolution multiplier

传统的卷积运算的参数和计算量:
* 假设卷积核的大小为 K, 矩形输入特征图的大小为 F(同时假设输出特征图大小也为F), 输入输出通道数为 M N
* 假设以 stride=1 进行卷积
* 卷积核参数 $K^2\times M \times N$ 
* 一次卷积的计算量 $K^2\times F^2 \times M \times N$

其灵感来源于 depthwise separable convolutions, 是一种 factorized convolutions  
depthwise conv 可以看作 Group Conv 的极端, 即分组数 g = 输入通道数
* 最早提出于 : 
* 将原本的卷积预算分成了两个分开独立的步骤
* filtering layer 进行 depthwise convolutions
  * 不管输出通道数 N 是多少, 卷据核的个数都是输入通道数 M, 产生 M 通道的输出
  * 卷积核参数 $K^2 \times M$
  * 计算量为 $K^2\times F^2 \times M$
  * 实际应用中, 产生的输出图已经进行了 resolution 变化
* combining layer 进行 pointwise convolutions
  * 相当于对每个像素进行独立的全连接映射,  $M\times F^2 \rightarrow N\times F^2$ 
  * 每个像素的参数是 $M\times N$ 
  * 卷积参数和计算量都为 为 $F^2\times M\times N$
* 可以戏剧性的降低模型的运算和大小, 解除了 `输出通道数N` 和 `卷积核大小` 的乘法关系
* 只牺牲了很小的精确率

#### 4.1.5.2. MobileNetV2

* 通过结合 ResNet的 shoutcut连接, 实现了V2, 核心是 inverted residual block
* residual block (ResNet) v.s. inverted residual block (MobileNetV2)
  * 都使用了 `1*1 -> 3*3 -> 1*1` 的卷积模式
  * 都使用了 shortcut 添加了将输入与输出直接相加的连接
  * 通道数:
    * ResNet 使用 `1*1` 进行降维, 卷积, 再用 `1*1` 升维 
    * MobileNetV2 使用 `1*1` 进行升维到6倍通道, 卷积, 再用 `1*1` 降维到原本输出通道 
    * 两边窄中间宽, 所以叫 inverted
  * 所有3*3 卷积都是 depthwise conv
  * 去掉了最后的ReLU

* inverted residual block的导入原因
  * ReLU的辨证探讨:
    * ReLU会使得一些神经元失活
    * 在高维通道数的ReLU可以保留低维特征的完整信息, 同时不损失非线性的特性
    * 而在低维空间时则会破坏特征, 不如线性的效果好
  * depthwise conv 本身没有改变输入通道的能力:
    * 上一层输入多少通道, depthwise 就输出多少通道
    * 使得卷积本身的次数太少, 特征提取的不够充分
  * 采用 inverted 中间宽+两边宅的方式, 来确保输入的低维信息可以完整保留

* V2的特点:
  * 因为使用了 inverted residual block, 所以公式上的计算量增大了
  * 在实际实现中, V2可以使用更低的通道数达到同样的效果, 所以实际上还是加速了
  * 精度方面的提升并不明显


Pytorch源代码:

```py

# 通道数标准化函数, 确保所有层的通道数都可以被8 (或者指定别的除数) 整除
def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

# 将 Depth wise 卷积封装, 包括了一个 dw 卷积, batch norm 和 relu 激活函数
class ConvBNActivation(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
        dilation: int = 1,
    ) -> None:
        padding = (kernel_size - 1) // 2 * dilation
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super().__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups,
                      bias=False),
            norm_layer(out_planes),
            activation_layer(inplace=True)
        )
        self.out_channels = out_planes

# necessary for backwards compatibility
ConvBNReLU = ConvBNActivation

# 残差卷积 block
class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int, 
        stride: int,
        expand_ratio: int,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))

        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            # 1*1 卷积升维
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        
        layers.extend([
            # dw 进行卷积, 这里直接 groups = hidden_dim 即 depthwise conv
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, 
              norm_layer=norm_layer),

            # pw-linear, 1*1 卷积降维, 注意这里后面是没有接激活函数的 
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
              norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x: Tensor) -> Tensor:
        # 残差网络的设置方法, 输入值x + 传播值 conv(x)
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use
        """
        super(MobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        # 默认是 7 个block
        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # 第一层网络, 无残差结构
        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features: List[nn.Module] = [ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer)]


        # building inverted residual blocks
        # 建立 blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            # n 代表重复的 block 次数
            for i in range(n):
                stride = s if i == 0 else 1
                # 加入残差 block 
                # expand_ratio=t , t 就是 inverted 残差block 的扩张倍数
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel

        # building last several layers
        # 最后一个 block 也是没有残差结构
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        # 分类器直接使用 一个线性层, 带上 dropout
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

```


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

# 5. 序列建模 - 循环和递归网络 RNN

* 普通的神经网络以及CNN, 都是假设元素之间是相互独立的, 输入与输出也是独立的
  * CNN 专门用于处理网格化数据
* 循环神经网络的特点: 拥有记忆能力, 输入依赖于当前的输出和记忆
  * 处理序列数据

网络结构:
* Recurrent neural networks (RNN)
* Long short-term memory (LSTM)
* Gated recurrent neural networks. (GRNN)

应用领域:
* Sequence modeling
  * Language modeling
* Transduction problem
  * Machine translation

##  5.1. Recurrent Neural Networks (RNN)

定义一个单节点的 RNN 网络 $U,V,W$, 给定输入序列 $x_t$ 表t时刻的输入
* 有t时刻的记忆(状态序列): $s_t=f(Ux_t+Ws_{t-1})$  
  * $f()$是激活函数
  * 可以把st当作一个隐状态
* 有 t 时刻的网络输出 $o_t=f(Vs_t)$
  * $V$ 即记忆的权重矩阵, 表示根据当前的记忆来决定输出
  * 很多任务并不需要中间的输出, 只关注最后的输出
* 网络中的所有细胞共享参数 $UVW$

## 5.2. Long Short-Term Memory LSTM

对比朴素 RNN :
* 将传统 RNN 的状态 s 序列定义成短期记忆
* 添加了另外一条状态序列 c, 称为 长期记忆

对于状态序列 c 的更新:
* 定义 遗忘长期参数 : $f1=sigmoid(w_1[s_{t-1},x_t]^T+b_1)$, 获得 遗忘系数
* 定义 新增短期记忆 : $f2=sigmoid(w_2[s_{t-1},x_t]^T+b_2)*tanh(\hat{w_2}[s_{t-1},x_t]^T+\hat{b_2})$
* 定义 新的长期记忆状态 $c_t=f1\times c_{t-1}+f_2$

长期记忆状态序列 c 会参与短期记忆 s 的更新
* 定义短期记忆 $s_t=sigmoid(w_3[s_{t-1},x_t]^T+b_3)*tanh(\hat{w_3}c_t+\hat{b_3})$

# 6. Attention

Recurrent模型通常会根据符号的输入顺序来分解计算, 根据t-1时刻的状态和t时刻的输入来决定t时刻的输出和新状态。这从原理上使得并行变得不可能, 同时导致内存开销极大.  
一些基于Block的并行化方法同时输入并计算各个位置上的特征, 但导致学习远距离关联变得很难.  
Transformer网络的提出彻底并行化了相关问题, 将远距离的计算降到了常数项.  

* Attention 从出发点上说是用于提升基于 RNN 的S2S模型效果的机制
* Attention 给模型赋予了 区分辨别的能力, 使得网络模型的学习变得更加灵活
* Attention本身可以作为一种对齐的关系, 可以用来解释翻译输入输出句子之间的对其关系


Self-Attention:
* 将单个序列中, 不同位置的信息关联起来, 最终计算出整个序列的特征表示
* Output : 输出, 为 key 的加权和, 权重 Weight 则通过兼容函数(compatibility function) 计算
* fun(query, key) -> weight, sum(weight-value)-> output
* 这里 query 和 key 的维度相同都是 $d_k$
* Dot-Product Attention function (Query, key, value)
  * 无 Scale $Attention(Q,K,V)=softmax(QK^T)V$
  * 有 Scale $Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt d_k})V$
    * Scale主要用于高维度特征向量时, 向量点乘的结果往往过大, 导致 softmax 梯度消失
  * 点乘的特点是相较于Additive便于使用矩阵优化
* Additive attention
  * Using a feed-forward network with single hidden layer.
  * 在 dk 比较小的时候效果优于 doc-product attention
* Multi-Head Attention
  * 在 Scaled Dot-product attention 的基础上, 利用线性变换改变 Q K V 维度, 组成 one head 
    * 这里假设$d_{model}$是根据任务的不同而变换的输入维度, 可以是 QKV的任意
    * $head_i=Attention(QW_i^Q,K_i^K,V_i^V)$
    * $W^Q,W^K,W^V \mathbb{R}^{d_{model}\times d_K}$
    * 对于原本有相同维度$d_{model}$ 的 Q K V, 也用一组 learned linear projections 投影到 $d_k, d_v$维向量
   * 这里的投影可以解释为权重矩阵, 是可学习的 (learned)
     * $W_i^Q\in \mathbb{R}^{d_{model}\times d_k}$
     * $W_i^K\in \mathbb{R}^{d_{model}\times d_k}$
     * $W_i^V\in \mathbb{R}^{d_{model}\times d_v}$
   * 最后 concat h 个 $d_v$ 维的向量形成最终输出
   * $headi_i=Attention(QW_i^Q,KW_i^K,VW_i^V)$  
   * $MultiHead(Q,K,V)=Concat(head_1,...,head_h)W^O$
  * 原论文中
    * head 的个数 h=8
    * $d_k=d_v=d_{model}/h=64$ , 即相对于原本的维度, 投影后的向量实现了降维
    * 大大降低了计算量, h=8 总共的计算量也小于之前保持维度的单个计算量

总结:
  - Attention 可以帮助模型对输入的X的每个部分赋予不同的权重, 抽出更加关键的信息
  - 并不会对模型的计算和存储带来更大的开销
* Attention mechanism - 注意力机制, 一种思想
  * Self-attention (intra-attention)
* Transformer - a transduction model
  * relying entirely on self-attention to compute representations.
  * without using sequence-aligned RNN or convolution.


# 7. Multi Object Tracking (MOT)

当前的 MOT 手法可以分成 2 个大类 
1. Batch Tracking
  * 使用当前帧的前后序列信息
  * 精度高
  * 因为使用了未来帧的信息因此不能应用在实时项目中, 只能 offline
2. Online Tracking
  * 使用当前帧 + 过去帧
  * 当前的精度比起 Batch Tracking 仍有不足
  * 实时追踪理论上可行, 19年时间点上仍然困难

总体上, 主流的 MOT 算法可以分成 4 个步骤
1. Detection stage      : 标准物体检测 (检出)
2. Feature extraction / Motion prediction stage   : 提取各个对象的用于追踪的特征图 / 或者直接预测对象的下一时间点的位置
3. Affinity stage       : 通过特征图或者预测信息, 比较下一时间点的物体检测得到的位置信息, 计算距离分数
4. Association stage    : 距离较小的对象认为是同一个物体, 实现追踪


MOT 常用的数据库:
* MOT Challenge datasets
* KITTI datasets : 人と車の両方の追跡用
* UA-DETRAC tracking benchmark : 交通用の監視カメラで撮影した車の見下ろし画像
* TUD datasets    : 人物追踪用的数据库
* PETS2009 datasets : 人の追跡用

MOT 的评价方法:
* 同样分 Accuracy, Precision ->  MOTA MOTP
* IDF1 (Identification F1)

# 8. Transfer Learning


## 8.1. Domain Adaptation (DA)

域适应: 指同一个模型在两个相似任务的下实现尽可能相同的效果, 是迁移学习的一种, 也是当前迁移学习的主要研究方向
* 两个任务有可能具有细微的差异 (光照, 姿态, 图像质量), 同一个模型直接使用会有很大的效果差异
* Domain Adaptation 是为了减少两个域在特征空间的差异, 使得模型学到更普适的特征

目前的 DA 可以主要分为两种
* One-step DA
  * 假设源域和目标域是相关的, 只是域的分布上有差异, 通过调整域间分布来实现域适应
* Multi-step DA
  * 假设源域和目标域是无关的 (更符合现实)
  * 通过在源域和目标域建立一些桥梁 (中间域), 多步实现域适应

根据数据类型分为 同构/异构
* 同构: 数据空间相同的数据. e.g. 不同类别但是分辨率相同的图片
  * 监督域自适应
  * 半监督域自适应
  * 无监督域自适应  (主流研究方向) 
* 异构: 数据空间不同的数据. e.g. 文字和图片之间

手法的方向:
* 特征的适应 (主流方向): 把源域和目标域的特征提取到统一的特征空间中, 让不同特征之间的距离足够近
* 实例的适应 (小技巧)  : 提取出源域中与目标域更相似的一部分数据, 给予该部分数据更大的权重
* 参数的适应 (小技巧)  : 直接修改模型的参数使得模型适应新的任务



## 8.2. Domain Generalization

对比 域适应 (Domain Adaptation) 更加泛化的域适应的研究方向  


# 9. View Synthesis

视角合成任务, 通过输入对一系列对物体不同角度的图像, 来生成新的角度下的图像
* 对于一个训练好的模型
* 通常输入的数据是一系列图像, 并且带有对应的角度数据, e.g. 空间坐标 (x,y,z) 视角 (theta, phi)
* 输出是
  * the volume density 
  * and view-dependent emitted radiance at that spatial location, 可以直接理解成新视角下的图像



通常的手法使用一个中间的 3D 场景表征来作为中介, 并以此生成高质量的虚拟视角, 根据该中间表征的形式, 可分为:
* 显式 Explicit representation : 例如 Mesh, Point Cloud, Voxel, Volume 等等, 对场景进行显式建模, 但是这些显式类型一般都是离散的, 有精度问题
* 隐式 Implicit representation : 用一个函数来描述几何场景, 一般是一个不可解释的 MLP 模型, 输入 3D 空间坐标, 输出对应的几何信息, 是一种连续的表示 (Neural Fields, 神经场)   


Neural Fields  神经场:
* 场 Fields   : 是一个物理概念, 对所有 (连续)时间 或 空间 定义的量, 如电磁场, 重力场, 对 场的讨论一定是建立在目标是连续概念的前提上
* 神经场表示用神经网络来 全部或者部分参数化的场
* 在视觉领域, 场即空间, 视觉任务的神经场即 以 `空间或者其他维度 时间, 相机角度等` 作为输入, 通过一个神经网络, 获取目标的一个标量 (颜色, 深度 等) 的过程   

## 9.1. Neural Rendering  的各种渲染方法

即中间层的显式表达方法 Mesh, Point Cloud, Voxel, Volume 等

### 9.1.1. Volume 体数据  体渲染

从体数据渲染得到想要的 2D 图片  

体数据是一种数据存储格式, 例如 医疗中的 CT 和 MRI, 地址信息, 气象信息
* 是通过 : 追踪光线进入场景并对光线长度进行某种积分来生成图像或者视频   Ray Casting Ray Marching Ray Tracing
* 这种数据需要额外的渲染过程才能显示成 2D 图像并被人类理解  
* 对比于传统的 Mesh, Point 等方法, 更加适合模拟光照, 烟雾, 火焰等非刚体, 在图形学中也有应用   
* 体渲染是一种可微渲染  



## 9.2. NeRF Neural Radiance Fields

2019年开始兴起, 在 2020 年 ECCV 中得到 Best Paper Candidate  

NeRF 是一种隐式的 3D 中间表示, 但是却使用了 Voluem 的规则, 即一个 隐式的 Volume, 实现了 神经场 Neural Field 与图形学组件 Volume Rendering 的有效结合  
* 本身的方法非常简洁, 且有效, 说明是合理的
* 对于启发 计算机视觉和图形学的交叉领域 有很大的功劳


### 9.2.1. Vanilla NeRF

将一个 scene 表示成一个 5D vector-valued function.
* 输入 3D location X=(x,y,z) 和 viewing direction d=(theta, phi)
* 输出 emitted color $c=(r,g,b)$ 和 volume density $\sigma$
* volume density sigma(x) 可以解释为一个 ray 在空间中的无限微小点 X 终止的微分概率  


基于 NeRF 的 Volume Rendering
* 对于一个 camera ray  $r(t)=o+td$  t 是 camera ray 的远近距离 t_n t_f
* camera ray 得到的颜色 C(r)可以写作  
$$C(r)=\int_{t_n}^{t_f}T(t)\sigma(r(t))c(r(t),d)dt$$
* $T(t)=exp(-\int_{t_n}^t\sigma(r(s))ds)$
  * 该公式代表了一个 accumulated transmittance, 
  * 对于一个距离 t 从 t_n 到 t, 光线最终没有被遮蔽的概率  
* 从一个 NeRF 模型中渲染出一个 view 需要
  * estimating this integral C(r) for a camera ray traced through each pixel of the desired virtual camera.
  * 从虚拟摄像头中, 对穿越每一个像素的 camera cay 计算 C(r)


NeRF 本身的问题, 有如下:
* 速度慢  : 对于每个输出像素分别进行前向预测, 因此计算量很大  
* 只能应用于静态场景
* 泛化性差
* 需要大量的视角  : 需要数百张不同视角的图片来训练  

## Neural 3D shape representations

在 NeRF 提出之前的主流方案, 对于一个连续的 3D shape
* map xyz coordinates to signed distance functions or occupancy fields
* 这种方案最早的时候需要 GT 3D geomerty, 因此在研究中经常使用 synthetic 3d shape
* 后来有直接输出每个坐标对应的 feature vector 和 RGB function, 在通过复杂的 rendering function 得到2D img 再计算 Loss

# 10. Dataset

## 10.1. Text spot

3755 classes (Ievel-l set of GB2312-80) 

### 10.1.1. Scene Text

* IIIT 5K-Words  (IIIT5K) contains 3000 cropped word images for testing.
* Street View Text (SVT) consists of 647 word images for testing. Many images are severely corrupted by noise and blur.
* Street View Text Perspective (SVT-P) contains 645
cropped images for testing. Most of them are perspective distorted.
* ICDAR 2003 (IC03) contains 867 cropped images after discarding images that contained non-alphanumeric characters or had fewer than three characters.
* ICDAR 2013 (IC13) inherits most of its samples from IC03. It contains 1015 cropped images.
* ICDAR 2015 (IC15) is obtained by cropping the words using the ground truth word bounding boxes and includes more than 200 irregular text images.

### 10.1.2. Handwritten Text

* IAM contains more than 13,000 lines and 115,000 words written by 657 different writers.
* RIMES contains more than 60,000 words written in French by over 1000 authors. 4.3.