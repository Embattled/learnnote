
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


**2 Excessive group convolution increases MAC**

group 分组数量会增加 MAC, 导致运行速度下降   

因此不能简单的通过增加分组 g 提高通道数来无脑的获得更高精度, g 过高可能显著的降低运行速度   


**3 Network fragmentation reduces degree of parallelism**

fragmentation reduces the speed significantly on GPU 


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


