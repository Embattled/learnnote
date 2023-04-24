# QualcommAI

基于高通的处理器的相关开发


Hexagon 是高通开发的一种产品名, 和 骁龙一样, 每一代都会升级
* HTP  (Hexagon Tensor Processor)   : Hexagon DSP 的完整产品名称  
  * 提供了 INT8 INT16
  * 提供了 32-bit 16-bit floating-point 的计算, 但是小数运算不会通过 HVX 进行加速
  * Qfloat 位置
  * 骁龙 888 开始搭载
* HVX  (Hexagon Vector Extension)   : Hexagon DSP 的向量演算回路, 用于并列加速计算
* HTA  (Hexagon Tensor Accelerator) : AI 运算的加速器, 从 Snapdragon855 开始搭载, 骁龙 888 开始被替换成 HTP
* HNN  (Hexagon Neural Network)     : 提供了 HVX 和 HTA 的使用接口

* SNPE      : 高通出品的, CPU/GPU/DSP全局调用的 推论 SDK
* QNN       : 同 SNPE 一样, 但是是 SNPE 的低一级的底层构成
* 高通推荐将使用的 HTA 或者 HVX 的软件移植成 QNN 调用的实现, 但是据说目前 QNN 接口仍然是 HTP 专用的


Adreno 高通处理器的 GPU 产品名称