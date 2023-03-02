# OpenVINO

为了挖掘处理器的性能, 硬件厂商自己也会发布对深度学习进行优化后的软件套件

例如 NVIDIA 的 TensorRT, OpenVINO则是Intel发布的对于AI工作负载的部署工具  

OpenVINO支持主流的深度学习框架  
* TensorFlow
* Paddle
* Pytorch
* Caffe
* ONNX
* mxnet
* Keras

具体的部署流程包括两步
* 模型优化器 (Model Optimizer) : 用于离线的压缩深度学习模型, 将框架训练好的模型转化成推理引擎专用的中间调达
  * 优化结构包括两个文件
  * xml  网络结构的描述
  * bin  权重文件
  * 其他的包括 层的融合, 内存优化
* 推理引擎  : 提供了 C/C++, Python 的 API 接口, 用于最终的推论实现  
* 四个辅助模块:
  * Open Model Zoo : 提供了超过200+个预训练好的模型仓库, 可以直接使用作为原型开发
  * OpenVINO Model Server : 模型服务工具, 类似于 TensorFlow Serving
  * Samples: 示例程序
  * DL Workbench : OpenVINO 的可视化工作台, 用于模型管理, 训练后量化, 可视化网络结构等
* 