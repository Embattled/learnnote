MambaIR: A Simple Baseline for Image Restoration with State-Space Model

Guo, H., Li, J., Dai, T., Ouyang, Z., Ren, X., & Xia, S. T. (2024, September). Mambair: A simple baseline for image restoration with state-space model.  
In European conference on computer vision (pp. 222-241). Cham: Springer Nature Switzerland.


MambaIR:ASimpleBaselineforImageRestorationwithState-SpaceModel

CNNs 和 Transformers 的既存 backbones 通常会面临 global receptive fields 和 计算效率的 两难问题, 会限制在实际产品中的应用  


Selective Structured State Space Model, 以及它的新形式 Mamba, 提供了 long-range dependency modeling with linear complexity 的可能性

标准的Mamba 应用在 low-level vision 上有一定的问题

本文提出了改进的 Mamba 使得
* local enhancement
* channel attention
* 维持了 Mamba 的
  * local pixel similarity
* reduces the channel redundancy
* 在较小的计算量上实现了 global receptive field

结果优于 swinIR


# Introduction

CNNs 和 Transforms 是图像 IR 的两大主流 backbone 方法  

模型的性能通常依赖于 感受野的增加
* 更多的像素能够促进 reconstruction of the anchor pixel  
* 网络可以提取 higher-level patterns and structures in the image, 对于降噪任务中, 一些纹理结构的复原非常重要
* 之前的论文 Activating More Pixels in Image Super-Resolution Transformer
  * 证实了在 Transformer-based 的方法中,  activating more pixels 会导致更好的结果




