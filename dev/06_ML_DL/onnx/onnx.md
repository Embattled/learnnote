# 1. Open Neural Network Exchange

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

ONNX的数据传输基于 Protobuf, 因此了解一些 Protobuf 的知识对于使用 ONNX 有帮助, 定义ONNX格式的文件为 `onnx.proto`

ONNX是一个开放式规范, 由以下组件组成
* 可扩展计算图模型的定义
* 标准数据类型的定义
* 内置运算符的定义

## 1.1. ONNX 格式

ONNX的组织格式, 以 `onnx.proto` https://github.com/onnx/onnx/blob/master/onnx/onnx.proto 来定义





# 2. onnx - Python API 

https://onnx.ai/onnx/intro/python.html

onnx 自己的 python API 库

## 2.1. Protos

定义了以 `protobuf` 标准的各种 数据结构, 书写在 `onnx/*.proto`  中.  官方推荐使用 `onnx.helper` 来创建这些 Protos 而不是直接构造它们.  

所有的 Protos shructure 都可以通过 print 来打印, 同时可以被 rendered as a json string.  

核心的几个对象, 以及之间的关系
* ModelProto
  * 加载了一个 ONNX 模型后, 得到的就是一个 ModelProto
  * 包括了一些版本信息
  * 包括了一个 GraphProto
* GraphProto
  * 包含了四个 repeated 数组分别是
  * node, NodeProto 类型
  * input, ValueInfoProto 类型, 存放了模型的输入节点
  * output, ValueInfoProto 类型, 存放了模型的输出节点
  * (通过 Input和 output 的指向关系, 即可还原出网络的拓扑结构)
  * initializer, TensorProto 类型, 存放了模型的所有权重参数
* NodeProto
  * 存放了模型中所有的计算节点
  * 每一个计算节点包括了一个 AttributeProto 数组用于存储节点的属性
  * 不同类型的计算节点有不同的属性
  * 每一个计算节点的属性，输入输出信息都详细记录在 https://github.com/onnx/onnx/blob/master/docs/Operators.md
* ValueInfoProto
  * 存放了模型的输入和输出节点
* TensorProto
  * 存放了模型的所有权重参数
* AttributeProto
  * 用于描述一个计算节点的属性
  * 例如 Conv 节点的属性就包括: group, pad, strides


# 3. Pytorch torch.onnx

https://pytorch.org/docs/stable/onnx.html

The torch.onnx module can export PyTorch models to ONNX.  
The model can then be consumed by any of the many runtimes that support ONNX.  



