- [1. Neuron Network](#1-neuron-network)
  - [1.1. Activate Function](#11-activate-function)
  - [1.2. LRN 和 BN](#12-lrn-和-bn)
    - [1.2.1. LRN](#121-lrn)
    - [1.2.2. BN](#122-bn)
- [2. CNN](#2-cnn)
  - [2.1. Recognition](#21-recognition)
    - [2.1.1. Alexnet](#211-alexnet)
    - [2.1.2. VGGNet](#212-vggnet)
    - [2.1.3. GoogLeNet](#213-googlenet)
    - [2.1.4. ResNet](#214-resnet)
- [3. RNN](#3-rnn)
# 1. Neuron Network

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


# 2. CNN 

* 简单粗暴的方法来提高精度就是加深网络以及加宽网络, 缺点有
  * 容易过拟合
  * 迅速增加的参数个数以及大量为0的无效神经点带来的浪费计算, 难以应用
  * 网络越深, 越容易梯度弥散
    * 即使使用了 BN, 几十层的CNN也非常难训练
* 基础的方法是增加网络的稀疏性, 尽可能减少全连接层

## 2.1. Recognition

* AlexNet
  * 推广了 ReLU 和 Dropout
* VGGnet 证明了网络深度与高级 feature 之间的关系
  * 因为网络深度, 训练时间更长, 但是收敛的迭代次数减小
  * 在识别和检测中都有较好效果
* GoogLeNet 彻底摆脱了全连接层
  * AlexNet 和 VGGNet 中的FC层占据了9成的参数, 而且会导致过拟合
  * 使用模块化的 Inception 来抽取特征



### 2.1.1. Alexnet

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
### 2.1.2. VGGNet

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


### 2.1.3. GoogLeNet

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


### 2.1.4. ResNet

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



# 3. RNN

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































