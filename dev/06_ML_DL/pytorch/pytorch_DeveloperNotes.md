# Developer Notes

独立章节介绍一个实践性的主题内容


# CUDA semantics

`torch.cuda` 用于 set up and run CUDA 运算。  

模组功能：
* 追踪当前使用的 GPU, 所有 CUDA tensors 都会默认创建在选定的 GPU 上
* 选定的 GPU 可以通过 context manager `torch.cuda.device` 来控制

要注意, 一旦 Tensor 被创建, 那么其运算结果总是会保存在相同的 GPU 上, 而不受 `torch.cuda` 模组的自动控制影响

跨 GPU 的运算是默认不被允许的, 除了 拷贝以及带有拷贝性质的操作: `copy_() to() cuda()` 等
对分布在不同设备上的 Tensor 执行运算, 除了配置好 `peer-to-peer memory access` 的情况以外都会引发错误



## TensorFloat-32 (TF32) on Ampere (and later) devices

Pytorch 1.7 以后
* 添加 了 `allow_tf32` flag
* `torch.backends.cuda.matmul.allow_tf32`
  * 1.7 ~ 1.11 之间的版本该 flag 默认为 True
  * 1.12 及以后默认为 False
* `torch.backends.cudnn.allow_tf32`
  * 卷积运算的操作默认允许 TF32

该 Flag 会允许 GPU 使用 TensorFloat32(TF32) 张量来计算矩阵乘法和卷积运算
* Matmul (Matrix multiplies and batched matrix multiplies)
* convolutions
* 具体的运算是将输入数据转为 `have 10 bits of mantissa` 底数只有 10 bit 的数字 进行乘法和卷积, 然后以 FP32 的精度执行加法
* 两个运算是有独立的 flag 控制的
  * 乘法 `torch.backends.cuda.matmul.allow_tf32`
  * 卷积 `torch.backends.cudnn.allow_tf32`

除了明面上的乘法与卷积以外, 所有的隐含的 乘法与卷积的操作都会受影响
* These include nn.Linear, nn.Conv*, cdist, tensordot, affine grid and grid sample, adaptive log softmax, GRU and LSTM.


启用 TF32 后, A100 上的速度 比 FP32 大约会快 7倍.
如果要确保 FP32 的精度, 就关闭这两个 Flag

```py
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
```