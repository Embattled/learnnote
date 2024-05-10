# Feature Matching - 特征匹配





# DeepLearning Feature Matching - 基于深度学习的特征匹配



## LightGlue

Lindenberger, Philipp, Paul-Edouard Sarlin, Marc Pollefeys, Eth Zurich, and Microsoft Mixed. n.d. “LightGlue: Local Feature Matching at Light Speed.”


基于 SuperGlue : Transformer 原理的匹配算法, 并专注于推理和学习的轻量化
* 总体思想
  * 在每个 block 后都进行关联的推理
  * 使得模型可以审视自己, 在早期推理就丢弃不可能匹配的点
  * 动态调整模型的大小, 而不是通过限制模型整体的容量来提速
* 模型:
  * 输入, 两个图像 AB, 特征向量以及对应的坐标 (因为该算法只进行匹配)
    * A:(dA,pA) B:(dB,pB)   p是归一化后的坐标, d 是特征向量

