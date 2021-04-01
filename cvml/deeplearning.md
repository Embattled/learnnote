# 1. Neuron Network

* ReLU ( Rectified Linear Units) 在 Alexnet 中被发扬光大, 被证明在深层网络中远比 tanh 快, 成功解决了Sigmoid在网络较深时的梯度弥散问题
* Dropout 在 Alexnet 被实用化, 验证了其避免模型过拟合的效果, 在 Alexnet 中主要是最后几个全连接层使用了 Dropout
* MaxPool 在 Alexnet 中被发扬光大, 避免了平均池化的模糊效果, 并且池化核的步长比核的尺寸小, 让池化层的感受野有重叠, 提高了特征的丰富性


## 1.1. LRN 和 BN

* LRN : 局部响应归一化 ( Local Response Normalization )
* BN  : 批量归一化     ( Batch Normalization )

归一化是深度神经网络中的重要步骤
* 弥补 ReLU ELU 等函数的无界性问题
  * 输出层的值不会被限制在范围中 (tanh的 (-1,1)), 可以根据需要尽可能高的增长
* 为了在一定程度上限制激活函数的输出值

### LRN

* Alexnet 架构中被引入, 一个不可训练的层
* 为了鼓励 横向抑制 ( lateral inhibition )
  * 组织兴奋神经元向其临近的神经元传递其动作趋势
  * 目的是进行局部对比度增强, 以便使局部最大像素值作用于下一层的激励


* 通道间的 LRN ( Inter-Channel LRN ) , 在 Alexnet 中被使用, 定义的邻域在整个通道上, 即对于 x,y 位置, 在 RGB 维度上进行归一化
![LRN公式](https://images4.pianshen.com/914/5f/5f02a0b2599665e5d8111caddd4676a2.png)
* 通道内的LRN ( Intra-Channel LRN ), 邻域仅在同一 channel 中进行拓展
![LRN公式](https://images3.pianshen.com/609/2b/2b636b49e4e3fab6520a4adbdc05e489.png)

* 希腊字符 ( k,α,n,β ) 用于定义 LRN 的超参数
  * k 用于避免分母为零的常数
  * α 归一化常数
  * β 对比度常数
  * n 用于定义归一化的邻域长度
  
### BN

* Batch Normalization, 同LRN不同, 这是一个可以被训练的层
  * 该层主要用于解决内部协变量偏移 ( Internal Covariate Shift ICS )
  * 如果不同的Batch的特征有不同的分布, 会减慢学习的速率
    * 对于一个给定的
    * 

# 2. CNN 

## 2.1. Recognition


### 2.1.1. Alexnet

* ReLU 激活函数被应用在了所有卷积层和FC层的后面

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