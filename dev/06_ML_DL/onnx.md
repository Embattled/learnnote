# Open Neural Network Exchange

是一种针对机器学习所设计的, 开放的文件格式, 用于存储训练好的模型.  
* 使得不同的人工智能框架 (Pytorch, MXNet等) 可以采用相同的格式存储模型数据并进行交互
* 主要由微软, 亚马逊, Facebook和IBM等公司共同开发
* 官方支持 加载ONNX模型并进行推理的深度学习框架有
  * Caffe2
  * PyTorch
  * MXNet
  * ML.NET
  * TensorRT
  * Microsoft CNTK
  * [等等很多](https://onnx.ai/supported-tools.html#buildModel)
*  TensorFlow 非官方的支持ONNX


ONNX是一个开放式规范, 由以下组件组成
* 可扩展计算图模型的定义
* 标准数据类型的定义
* 内置运算符的定义

