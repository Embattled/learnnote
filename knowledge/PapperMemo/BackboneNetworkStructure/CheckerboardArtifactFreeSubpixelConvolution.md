
Checkerboard artifact free sub-pixel convolution


A note on sub-pixel convolution, resize convolution and convolution resize



关于 sub-pixel convolution 的棋盘格子伪像的 笔记


上采样的卷积层通常叫做 deconvolution layer, 有多种实现方法, 名称也很多, 包括
* sub-pixel 
* fractional convolutional layer
* transposed convolutional layer
* inverse, up, backward convolutional layer
* etc


在 Distill 的文章 Deconvolution and Checkerboard Artifacts https://distill.pub/2016/deconv-checkerboard/
有总结, checkerboard 伪像的原因有 3 中
* deconvolution overlap
  * 上采样的卷积中, 若卷积核 size 不能被 stride 整除, 那么每个 High-Resolution 计算的 由来 Low-Resolution 的特征量数量不是定值
* random initialization
  * 即使 stride 能够整除, 也会由于 random initialization 出现格子
* loss function
  * 不均匀梯度更新 inhomogeneous gradient updates
  * downsampling : strided convolutions
  * max-pooling in loss function
* 该文章提出的 resize convolution 可以抑制 前两条原因导致的伪影
  * 先通过 NN 强行提高特征图尺寸
  * 再应用 一个标准的 conv layer with both input and output in HR space
  * 在生成式模型中非常受欢迎



sub-pixel convolution 是另一种 上采样 deconvolution layer
* 在 LR 空间中执行正常 conv, 然后应用 periodic shuffling operation, 即 pixel-shuffle
* 这种方法对比 resize convolutions, 参数更多, 因此模型容量更大
* 这种方法从原理上不会发生 deconvolution overlap
* 但是受限于 random initialization, 反而最终会产生更多的 棋盘伪像
  * 本文就职着眼于 改善 sub-pixel, 减轻由于 random initialization 产生的 checkerboard 
  * 在此基础上证明了 sub-pixel 的更强的建模能力能够提高模型效果

# 1. Section 1: Sub-pixel convolution and resize convolution 



## 1.1. Sub-pixel 随机初始化 棋盘伪影分析 


pixel shuffle 上采样层的公式总结为

$$
    I^{SR} = P (W * f^{L-1}(I^{LR})+b)
$$
* P 是最终的 pixel shuffle
* W 的卷积操作
* b 是卷积的 bias, 在后续的公式中省略

在上采样系数为 2 的时候, 这里 W 卷积的 shape 为 `(12, 64, 5, 5)`, 因为是输出 3 通道的 RGB, 这里 pixel shuffle 会将通道缩减为 4 分之一, 因此卷积的输出通道为 12


sub-pixel 的概念理解
* 一种是正常输出 4 倍通道的卷积, 然后执行 pixel shuffle
* 另一种是想象一个 sub-pixel space
  * 该空间基于输入 tensor, 尺寸扩大, 同时尺寸的间隙填充 0 
  * 在该新空间上执行卷积操作
  * 该空间上的卷积核大小 Wsp, 输出通道变为 1/4, 尺寸变为 2 倍, 即 `(3, 64, 10, 10)`
* 基于 sub-pixel 空间的 sub-pixel convolution 可以理解为
$$
  I^{SR} = P (W * f^{L-1}(I^{LR})) = W^{sp} * SP(f^{L-1}(I^{LR}))
$$


给出 sub kernel 的定义
* 给定一个正常卷积核  W $(c_o, c_i, w, h)$, 这里 co 是输出通道数, 且能够倍 采样系数的平方整除, 例如 r=2, 则 12 能够被 4 整除
* 对于 一个 以 r^2 个卷积核为一组的 卷积核组 k, k in {0,..., co/(r^2-1)}
* 那么这里的一组 卷积核 `(r^2, 1, w, h)` 可以同等的被替换为 sub-pixel space 下的卷积核 $(1, 1, wr, hr)$
* 即 上述两种 sub-pixel 解释是可以同等替换的, 统一 sub-pixel convolution 的定义为

$$
    I^{SR}_n = W_n * f^{L-1}(I^{LR}), n \in (0, ..., r^2-1), I^{SR}=P(I^{SR}_n)
$$

也就是说, 基于 pixel-shuffle 的上采样方式, 每个HR像素点仅仅取决于一组 卷积核 $W_n$
* 然而每一组卷积核的输入特征都相同, 都是 $f^{L-1}(I^{LR})$
* 因此倘若 卷积核都是独立的随机初始化, 那么其根据随机初始化产生的输出自然相差很远
  * 而相差很远的输出再通过 pixel-shuffle 组合成临近的像素, 那么自然而然就会有棋盘伪影



## 1.2. NN resize convolution

$$
    I^{SR} = W^{sp}* N(f^{L-1}(I^{LR}))
$$
对比基于 sub-pixel space 的 sub-pixel 方法, 就是把 SP 换成了 NN

同样的, 该方法也解决了 overlap 问题, 因为 stide 始终是 1, 也不存在随机初始化的问题  

然而这种方法的缺点是
* 通过 sub-pixel space 的视角 统一  resize convolution 和 sub-pixel 方案
* 可以注意到 sub-pixel 是隐形的跳过了 sub-pixel space 的 0 值特征计算
* 而 NN 上采样后, 重复的计算会大量出现
* 因此结论是, 在 **相同计算资源的情况下, sub-pixel 能够使用更多的参数**



# 2. Section 2: Initialization to convolution resize

推理逻辑
* 先 NN 后 conv
  * 无法跳过 sub-pixel space 的 0 值像素计算
* 先 conv 后 NN
  * NN 本身不可训练
* 整合 sub-pixel
  * 使得 sub-pixel 的初始化参数同 NN 方法一致即可
  * 即初始化的参数会输出 r^2 个相同值的通道
使得

$$
    P(W'*f^{L-1}(I^{LR})) = N(W_0*f^{L-1}(I^{LR}))
$$

即创建 $W'n  = W_0$
* 对于输出通道为 k 并且打算通过 pixel-shuffle 提高解析度, 通道数降低为 1/r^2 的情况
* 随机初始化 k/r^2 通道的卷积核即可, 然后重复 r^2 次, 生成输出通道为 k 的完整卷积核

* `tensor(k,h,w) * w(k) ->`

实际实现中, 随机初始化后, 最方便的是 W_0 直接重复 r^2 次, 从通道顺序上来说
* 结果的特征图 
* `k -> r^2, k/r^2`  然后 pixel- shuffle 靠前的通道即可


# 3. Section 3: Experiments

对比了
* sub-pixel convolution
* NN resize convolution
* sub-pixel convolution **Initialized to Convolution NN Resize** (ICNR)


sub-pixel 参数更多, 其中 ICNR 收敛的更快, 最终 Loss 更小