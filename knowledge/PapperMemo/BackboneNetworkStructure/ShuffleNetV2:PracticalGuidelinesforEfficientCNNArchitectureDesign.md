
Ma, N., Zhang, X., Zheng, H.-T., & Sun, J. (2018).
ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design.
In Computer Vision – ECCV 2018,Lecture Notes in Computer Science (pp. 122–138). https://doi.org/10.1007/978-3-030-01264-9_8


提出了更贴近实际的网络复杂度计算指标  

同时提出了 ShuffleNet V2 的网络结构

# 1. Introduction

引用了其他工作的结论 : FLOPs 不能作为评判网络实际运行速度的唯一指标  

其他的影响因素包括:
* memory access cost (MAC). group convolution 通常很影响内存命中率
* degree of parallelism  网络的并行可能性
更比如说一些库还会针对特殊的网络结果做专属优化, 例如 3x3 Conv


本文提出的  ShuffleNet 在 40M FLOPs 的计算复杂度限制下, 比初版 ShuffleNet 精度更高, 速度更快


# 2. Practical Guidelines for Efficient Network Design


在 GPU 和 ARM 高通骁龙 810 评测, ARM 为单线程评测  

FLOPs 只会统计卷积部分的计算量, 对于其他部分 
* I/O
* data shuffle
* element-wise operations (AddTensor, ReLU, etc)
没有计入统计


**1 Equal channel width minimizes memory access cost (MAC)**

输入输出通道越接近, MAC 的下限越低  

提供的实验结果证明了上述理论

理论依据: depthwise separable convolutions 的计算量特点  
* 计算量的大头是 channel-wise (1x1 卷积), 对于输入输出分别是 c1, c2 的 1x1 卷积层
  * 其计算量为 `B = hwc1c2`
* 假设极限情况 处理器的 cache 能够存储所有参数和特征, 即不考虑 cache 命中率的因素
  * 参数访问量 `c1 x c2`
  * 输入特征量 `c1 x h x w`
  * 特征输出量 `c2 x h x w`
  * 总 MAC 为 `MAC = hw(c1+c2) + c1c2`
* 带入 B 有
  * `Mac = B(c1+c2)/c1c2 + B/hw`
* 复习均值不等式, 只有所有 x 相等的时候, 等号才会成立  
  * `Q >= A >= G >= H `
  * Q `平方平均数` 平方的均值再开方   $\sqrt{\frac{x^2_1+x^2_2+...+ x^2_n}{n}}$ 
  * A `算数平均`  average $\frac{x_1+x_2+...+x_n}{n}$
  * G `几何平均数` 累乘再开n次方 $\sqrt[n]{x_1x_2...x_n}$
  * H `调和平均数` 倒数求平均再取倒数 $\frac{1}{\frac{\frac{1}{x_1}+\frac{1}{x_2}+...+\frac{1}{x_n}}{n}} = \frac{n}{\frac{1}{x_1}+\frac{1}{x_2}+...+\frac{1}{x_n}}$

* 依据均值不等式
  * 算数平均大于几何平均, 有 $\frac{c1+c2}{2} >= \sqrt{c1c2}$
  * 则替换掉 (c1+c2) 有
  * `Mac >= 2B sqrt(c1c2) /c1c2 + B/hw`
  * `Mac >= 2B sqrt(c1c2) /c1c2 + B/hw`
  * `Mac >= 2hw sqrt(c1c2) + B/hw`
  * `Mac >= 2 sqrt(hwB) + B/hw`

因此, Mac 的 low bound 依据不等式为 c1 c2, 只有 c1=c2 不等式才成立  





**2 Excessive group convolution increases MAC**

group 分组数量会增加 MAC, 导致运行速度下降   

分组情况下单个层的 MAC 为 `MAC = hw(c1+c2) + c1c2/g`  输入输出的特征量数据没有变, 但是参数变少了使得 MAC 减少

* 带入 `B = hwc1c2/g` 有
* `MAC = hwc1 + hwc2 + c1c2/g`
  * `hwc2 = Bg/c1`
  * `c1c2/g = B/hw`
* `MAC = hwc1 + Bg/c1 + B/hw`


因此不能简单的通过增加分组 g 提高通道数来无脑的获得更高精度, g 过高可能显著的降低运行速度   



**3 Network fragmentation reduces degree of parallelism**

fragmentation reduces the speed significantly on GPU 

卷积切片并且并行涉及到 线程同步等操作, 会严重影响计算速度  

这里的 fragmentation 就是 group conv 中的分组


**4 Element-wise operations are non-negligible**

各元素级别的运算尽管没有 FLOPs, 但仍然有较重的运行时间和 MAC  


通过 bottlenect 构造来测量 Element-wise Operations  的 MAC/FLOPs  


在移除 ReLU 和 shortcut 后, GPU 和 ARM 上的运行速度提高了 20%   


**Conclusion and Discussions**

通过实验, 总结出来的高速网络设计经验为:
* 结构尽可能 输入输出通道数相同
* group conv 的使用需要谨慎
* reduce the degree of fragmentation 
* 减少 element-wise 运算


# 3. ShuffleNet V2: an Efficient Architecture  

提高 channel 的数量, 并且维持 输入输出通道相等, 同时不要过度增加 group 数量

从 V1 的经验, 在维持计算量不变的情况下, 通道数越多网络能力越强  
* poinitwise group convolution
* bottlenect-like structures

根据第二章的内容
* 上述两个方案都会增加 MAC, 因此成本并不是可以忽略的, 违反1,2 条
* 使用太多的 group 同时也违反了第 3 条   
* 在 shortcut 上使用的 add 属于 element-wise 运算, 因此违反了 第 4条


Shuffle V1 的结构
* block 分支
  * identity 和 卷积分支
  * 两个分支通过 ADD 合并
* 卷积分支里应用 V1 的 channel-wise group 和 channel shuffle
  * 包括 1x1 GConv, Channel Shuffle, 3x3 Depth-wise, 1x1 GConv

Shuffle V2 的结构
* 每一个 block 在一开始一分为 2, 且通道也是一分为2 (Channel Split)
  * 并不是相同的特征图执行不同分支的运算, 而是整个输入特征图一分为2 走两条 分支
  * 其中一个分支为 identity
  * 两个分支合并的时候使用 concat 而不再是 add, 相当于整个 block 的特征通道数不变  
    * 分支合并之后执行 channel shuffle
* 具有卷积的计算分支包括 3 个卷积, 分别是 1x1 channel-wise, 3x3 depth-wise, 1x1 channel-wise
* 关于降采样
  * ShuffleNet V1 是将 identiy 换乘 3x3 AVG Pool, 同时 depth-wise 的 stride 为2, 最终 分支执行 concat 使得通道数翻倍
  * ShuffleNet V2 则是跳过了 channel-split, 然后 idenetity 换成 `3x3 depth-wise stride2 + 1x1 channel-wise`, 计算分支 也是 depth-wise 的 stride 为2
    * 因为跳过了 channel-split 所以最终的通道数翻倍
* 两个 block 相连的地方可以合并
  * 前一个 block 的 concat 后接 channel shuffle, 然后再次 channel split


