# Backbone Network in DeepLearning


# Transformer

## RNN 的缺点
Recurrent模型通常会根据符号的输入顺序来分解计算, 根据t-1时刻的状态和t时刻的输入来决定t时刻的输出和新状态。这从原理上使得并行变得不可能, 同时导致内存开销极大.  
一些基于Block的并行化方法同时输入并计算各个位置上的特征, 但导致学习远距离关联变得很难.  
Transformer网络的提出彻底并行化了相关问题, 将远距离的计算降到了常数项.  
RNN 的主要缺点
* 对于长序列的顺序计算导致很慢
* 构造容易导致对于距离较远的输入之间的 梯度消失/爆炸
* 很难保留较早输入的信息, 这对于 长字符顺序输入的翻译任务是致命的


Transformer的构造包括  
* encoder
* decoder

* Attention 从出发点上说是用于提升基于 RNN 的S2S模型效果的机制
* Attention 给模型赋予了 区分辨别的能力, 使得网络模型的学习变得更加灵活
* Attention本身可以作为一种对齐的关系, 可以用来解释翻译输入输出句子之间的对其关系


Self-Attention: 将单个序列中, 不同位置的信息关联起来, 最终计算出整个序列的特征表示, 本质上是特征提取器 (Backbone Network)
* Output : 输出, 为 key 的加权和, 权重 Weight 则通过兼容函数(compatibility function) 计算
* fun(query, key) -> weight, sum(weight-value)-> output
* 这里 query 和 key 的维度相同都是 $d_k$
* Dot-Product Attention function (Query, key, value)
  * 无 Scale $Attention(Q,K,V)=softmax(QK^T)V$
  * 有 Scale $Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt d_k})V$
    * Scale主要用于高维度特征向量时, 向量点乘的结果往往过大, 导致 softmax 梯度消失
  * 点乘的特点是相较于Additive便于使用矩阵优化
* Additive attention
  * Using a feed-forward network with single hidden layer.
  * 在 dk 比较小的时候效果优于 doc-product attention
* Multi-Head Attention
  * 在 Scaled Dot-product attention 的基础上, 利用线性变换改变 Q K V 维度, 组成 one head 
    * 这里假设$d_{model}$是根据任务的不同而变换的输入维度, 可以是 QKV的任意
    * $head_i=Attention(QW_i^Q,K_i^K,V_i^V)$
    * $W^Q,W^K,W^V \mathbb{R}^{d_{model}\times d_K}$
    * 对于原本有相同维度$d_{model}$ 的 Q K V, 也用一组 learned linear projections 投影到 $d_k, d_v$维向量
   * 这里的投影可以解释为权重矩阵, 是可学习的 (learned)
     * $W_i^Q\in \mathbb{R}^{d_{model}\times d_k}$
     * $W_i^K\in \mathbb{R}^{d_{model}\times d_k}$
     * $W_i^V\in \mathbb{R}^{d_{model}\times d_v}$
   * 最后 concat h 个 $d_v$ 维的向量形成最终输出
   * $headi_i=Attention(QW_i^Q,KW_i^K,VW_i^V)$  
   * $MultiHead(Q,K,V)=Concat(head_1,...,head_h)W^O$
  * 原论文中
    * head 的个数 h=8
    * $d_k=d_v=d_{model}/h=64$ , 即相对于原本的维度, 投影后的向量实现了降维
    * 大大降低了计算量, h=8 总共的计算量也小于之前保持维度的单个计算量

总结:
  - Attention 可以帮助模型对输入的X的每个部分赋予不同的权重, 抽出更加关键的信息
  - 并不会对模型的计算和存储带来更大的开销
* Attention mechanism - 注意力机制, 一种思想
  * Self-attention (intra-attention)
* Transformer - a transduction model
  * relying entirely on self-attention to compute representations.
  * without using sequence-aligned RNN or convolution.



# U-Net

## NAFNet  - Simple Baselines for Image Restoration

2022/04/10  
旷视科技的文章, 化繁为简, 设计了一个及其简单的网络并达成了 NR 和 Deblur 任务的 SOTA

Inter-block Complixity : 块间复杂度, 设计多了级联的 U-Net 来提高性能
* Multi-Stage Progressive Image Restoration(2021)
* HINet: Half Instance Normalization Network for Image Restoration (2021)

Intra-block Complexity : 块内复杂度, 引入 Attention 进行降噪  

