# Pytorch Quantization


Pytorch 的量化学习仍然处于 beta 版本

## Introduction to Quantization

量化指的是在低精度下存储并运算深度学习网络  

主要是一种加速 inference 的技术, 且量化运算仅支持前向运算


量子化全局介绍参考 [上一级笔记](../quantization.md)
[官方文档](https://pytorch.org/docs/stable/quantization.html)

Pytorch 在底层提供了一些 量化的 Tensor 以及相关的运算, 用于在根本上构建一个低精度的计算流.  
Pytorch 的高级 API 则是体哦那个了相对应的转换方法将传统 float 模型转化为 低精度网络, 并尽可能降低精度丢失  

## Pytorch Quantization API Summary

Pytorch 的API 提供了两种不同的量子化 Model
* Eager Mode Quantization (beta feature)
  * User needs to do fusion and specify where quantization and dequantization happens manually
  * also it only supports modules and not functionals. 
  * 手动的模式, 需要手动指定要进行量化和反量化的位置, 且仅支持模块, 暂不支持自动转换的函数
* FX Graph Mode Quantization (prototype feature)
  * New users of quantization are encouraged to try out FX Graph Mode Quantization first
  * automated quantization framework in PyTorch
  * improves upon Eager Mode Quantization by adding support for functionals and automating the quantization process
  * people might need to refactor the model to make the model compatible with FX Graph Mode Quantization (symbolically traceable with torch.fx)
  * FX Graph Mode Quantization is not expected to work on arbitrary models since the model might not be symbolically traceable.


对于一个任意的自定义模型 (arbitrary):
* Pytorch provide general guidelines, but to actually make it work, users might need to be familiar with `torch.fx`, especially on how to make a model symbolically traceable.
* 用户需要自己熟知 torch.fx 并且知道如何去构建符号追踪


Pytorch两种 量化模式的比较:  
| Function                         | Eager Mode      | Fx Graph Mode   |
| -------------------------------- | --------------- | --------------- |
| Release Status                   | beta            | prototype       |
| Operator Fusion                  | Manual          | Automatic       |
| Quant/DeQuant Placement          | Manual          | Automatic       |
| Quantizing Modules               | Supported       | Supported       |
| Quantizing Functionals/Torch Ops | Manual          | Automatic       |
| Support for Customization        | Limited Support | Fully Supported |
总的来说, Fx 模式都是更加方便使用的


Pytorch 所支持的 3 中量化类型
1. dynamic quantization 动态量化 : weight act 都是以 float 存储的, 仅在 计算 的时候量化
2. static quantization : weight act 都量化, 需要在 训练后进行 calibration
3. static quantization aware training : 也是 weight act 都量化, 但在训练的过程中进行量化



# Quantization API Reference (torch.quantization)

torch.quantization

