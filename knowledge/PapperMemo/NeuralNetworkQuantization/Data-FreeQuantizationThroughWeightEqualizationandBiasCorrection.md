
Nagel, Markus, et al. 
Data-Free Quantization Through Weight Equalization and Bias Correction.
2019 IEEE/CVF International Conference on Computer Vision (ICCV), 2019, https://doi.org/10.1109/iccv.2019.00141.  


高通的后量化文章  
2019  

不需要 fine-tuning 和 超参数选择  

8-bit fixed-point 

基于 scale-equivariance property of activation function  
应用 biases correction 修复量化误差  

# Introcution

推荐了一个 量化的白皮书  Quantizing deep convolutional networks for efficient inference: A whitepaper

常用的缓解量化误差的方法需要 data 或者 fune-tuning  
这只适用于硬件或者客户直接相关者, 很麻烦  

高通在本论文阐述了他们的 无需数据的量化方法  

通过调整 weight tensor 的值来使得权重更加适合 量化  
同时会主动考虑量化过程中带来的误差偏差  


论文阐述了实际生产中的量化解决方案, 并按照应用工作量和难度的顺序进行排序  
1. 无需任何 数据和 反向传播, 使用任何模型, 仅需调用一次 API
2. 需要数据, 用于 re-calibrate batch normalization statistics, 或者计算 layer-wise loss function 来提高量化性能
3. 需要数据和反向传播, 需要调整训练超参数, 使用完整的 training pipeline
4. 从网络结构的设计上考虑量化的适配性, 需要 train from scratch


# Background and related work


介绍了到成文的时间位置  
* 主要的量化优化方法都是 QAT
* per-channel 的概念
* 其他方法都是 level 4 的方法


# Motivation


**Weight tensor channel ranges**
阐述了 通道之间的 权重范围可能有很大的差异  

通过调整权重使得他们的范围相似可以改善量化精度  

**Biased quantization error**

量化后的权重会导致输出本身有偏差  

特别是对于 depth-wise conv, 如果是 3x3 卷积, 其每一个通道的输出特征图只取决于 9 个权重

# Method

## Cross-layer range equalization

**Scaling equivariance in neural networks**:  
通过完整的考虑两个相邻 layer 的计算逻辑, 将前后两个 layer 的权重进行等价置换, 使得对应 范围差异较大的 channel 的权重能够更贴近全局的参数分布   

得到缩放系数后  
* `W_2'=W_2 S`
* `W_1'=S^(-1) W_1`
* `b_1'=S^(-1) b_1`


**Equalizing ranges over multiple layers**:
阐述了最优化模型参数 scaling 的计算逻辑  
每一个通道的量化精度 p_i 等于其权重范围 r_i 除以权重 tensor 的范围R:  $p_i = \frac{r_i}{R}$  

那么定义最优的缩放系数 S 为:  $\underset{S}{\max}\sum_ip_i^{(1)}p_i^{(2)}$  
即前后相邻两个层各自通道的精度乘积    

根据不等式, 可以最终拿到某个通道最优的缩放系数为 $s_i = \frac{1}{r_i^{(1)}}\sqrt{r_i^{(1)}r_i^{(2)}}$  
此时 $\forall i : r_i^{(1)}=r_i^{(2)}$  

整个运算是通过不停迭代所有相邻层的 S 使得最终收敛  

**Absorbing high biases**  

通过缩放, 权重可以调整范围, 但是如果 `s<1`, 对应layer的权重bias $b_i^{(1)}$ 就会变大  

会导激活值变大, 那么通过将 bias 的 效果转移到后续 layer 的权重中, 减少高 bias 带来的影响  


## Quantization bias correction


