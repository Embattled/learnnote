# Pytorch Quantization

量子化全局介绍参考 [上一级笔记](../quantization.md)

[官方文档](https://pytorch.org/docs/stable/quantization.html)

## Pytorch Quantization API

Pytorch 的API 提供了两种不同的量子化 Model
* Eager Mode Quantization (beta feature)
  * User needs to do fusion and specify where quantization and dequantization happens manually
  * also it only supports modules and not functionals. 
* FX Graph Mode Quantization (prototype feature)
  * New users of quantization are encouraged to try out FX Graph Mode Quantization first
  * automated quantization framework in PyTorch
  * improves upon Eager Mode Quantization by adding support for functionals and automating the quantization process
  * people might need to refactor the model to make the model compatible with FX Graph Mode Quantization (symbolically traceable with torch.fx)
  * FX Graph Mode Quantization is not expected to work on arbitrary models since the model might not be symbolically traceable.

对于一个任意的自定义模型 (arbitrary):
* Pytorch provide general guidelines, but to actually make it work, users might need to be familiar with `torch.fx`, especially on how to make a model symbolically traceable.


# Quantization API Reference (torch.quantization)

torch.quantization

