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

ONNX的数据传输基于 Protobuf, 因此了解一些 Protobuf 的知识对于使用 ONNX 有帮助, 定义ONNX格式的文件为 `onnx.proto`

ONNX是一个开放式规范, 由以下组件组成
* 可扩展计算图模型的定义
* 标准数据类型的定义
* 内置运算符的定义

## ONNX 格式

ONNX的组织格式, 以 `onnx.proto` https://github.com/onnx/onnx/blob/master/onnx/onnx.proto 来定义

以 message 关键字开头的对象是需要关心的

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


## ONNX Runtime

ONNX Runtime 是由微软维护的一个跨平台机器学习推理加速器, 即实际进行推理的 推理引擎 
* ONNX Runtime 是直接对接 ONNX 的
* ONNX Runtime 可以直接读取并运行 .onnx 文件, 不需要另外的转换
* 对于 PyTorch - ONNX - ONNX Runtime 这条部署流水线, 得到 .onnx 文件 并通过  ONNX Runtime 上运行模型, 模型部署就算大功告成了

对于模型的推论:
* onnx.reference 是 onnx 内部自带的用于测试的推论接口
* onnxruntime 是一个独立的, 可以直接对接 onnx 的高速推论引擎
* 因为这二者的概念不同, 所以 onnxruntime 放于别的章节里



# onnx - Python API 

https://onnx.ai/onnx/intro/python.html

onnx 自己的 python API 库

# Pytorch torch.onnx

https://pytorch.org/docs/stable/onnx.html

The torch.onnx module can export PyTorch models to ONNX.  
The model can then be consumed by any of the many runtimes that support ONNX.  



