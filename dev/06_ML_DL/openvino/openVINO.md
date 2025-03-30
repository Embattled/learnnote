# 1. OpenVINO

为了挖掘处理器的性能, 硬件厂商自己也会发布对深度学习进行优化后的软件套件

OpenVINO 是 Intel 发布的对于AI工作负载的部署工具, 与之类似的几个项目有:
* NVIDIA 的 TensorRT, 用于在 NVIDIA GPU 上进行深度学习推理的高性能库
* NCSDK (Movidius Neural Compute SDK) :由英特尔 Movidius 开发, 用于在英特尔 Movidius 系列硬件上进行深度学习推理的工具
* TVM (Tensor Virtual Machine) : 一个由社区驱动的深度学习编译器和优化器, 旨在在各种硬件上实现高效的深度学习推理


OpenVINO支持主流的深度学习框架  
* TensorFlow
* Paddle
* Pytorch
* Caffe
* ONNX
* mxnet
* Keras

# 2. Sample Applications

定义了许多使用 OpenVINO 的样例软件, 同时还有一个用于测试模型性能的应用  


## 2.1. Benchmark Application

Estimates deep learning inference performance on supported devices for synchronous and asynchronous modes.
  
该应用有  C++ 和 python 两种实现, 都有这相同的 command interface and backend. 但是根据应用场景的语言的不同, 更加推荐使用相同语言的实现.


3. https://docs.openvino.ai/2023.0/openvino_inference_engine_tools_benchmark_tool_README.html
=

在通过 pip 安装好 OpenVINO 开发环境后, 即可通过命令行直接访问 benchmark_app, 支持 OpenVINO IR (即 mode.xml model.bin) 和 ONNX 格式的模型, 其他模型需要提前转化.  

`benchmark_app -m model.xml`  

benchmark 的工作流程为:
* 将模型加载到 CPU 平台, 同时打印 benchmark 的各项参数
* 为模型 feed 随机生成的输入数据, 持续推论60秒
* 结束 benchmark 后, 报告 minimum, average, and maximum inferencing latency and average the throughput.


## 3.1. Basic Configuration Options



# 4. OpenVINO Workflow

OpenVINO 的具体的部署流程
* 模型准备 Model Preparation ： 下载预训练模型或者将自己训练模型, 并将模型转换成 OpenVINO 格式, 将框架训练好的模型转化成推理引擎专用的中间调达
  * Open Model Zoo : 提供了超过200+个预训练好的模型仓库, 可以直接使用作为原型开发
* 模型优化 (Model Optimizer and Compression) : 描述了集中 traning and post-traning stages 的模型优化方法

* Running and Deploying Inference 推理引擎  : 提供了 C/C++, Python 的 API 接口, 用于最终的推论实现  

辅助模块:
  * OpenVINO Model Server : 模型服务工具, 类似于 TensorFlow Serving
  * Samples: 示例程序
  * DL Workbench : OpenVINO 的可视化工作台, 用于模型管理, 训练后量化, 可视化网络结构等


## 4.1. Model Preparation

首先, 所有 DL 工作流都需要从获取模型开始, OpenVINO 环境允许从多种框架获取模型源文件, 并转化成 OpenVINO 自己的表达 `openvino.runtime.Model` (ov.Model). 


有许多可选项用于实现把一个模型从原本的框架转化成 OpenVINO model format (ov.Model)


使用 Python 的接口
* `read_model()` 接口可以从一个文件读取模型并且创建 ov.Model.
  * 如果该模型属于一个被支持的 framework, 则转换会自动进行.
  * 如果该模型本身就是 OpenVINO IR 格式, 则会以 `as-is` 的方法读取, 不进行任何转化.
  * 对于 `ov.Model` 格式的模型, 可以通过 `ov.serialize()` 方法进行序列化, 序列化后的模型可以进一步通过 NNCF 来进行 post-training quantization 优化
* `openvino.tools.mo`
  * 通过使用 `mo.convert_model()` API 来将一个模型转化成 ov.Model
  * `mo.convert_model()` 接口中同时还有一些参数可以用来修改模型, 例如  cut the model, set input shapes or layout, add preprocessing

具体的使用方法为两种:
* Pytorch model object / TensorFlow model object (or model file path) / ONNX, PaddlePaddle model file path  
* -> convert_model()
* ov.Model
* -> compile_model()
* ov.CompiledModel

以及事实上 ONNX, PaddlePaddle, TensorFlow and TensorFlow Lite 的模型并不需要 `mo.convert_model` 这一步骤, 可以直接通过 `read_model` 接口来实现转化, 直接成为 ov.Model
* ONNX, PaddlePaddle, TF Lite or TF model file path
* -> read_model()
* ov.Model
* -> compile_model()
* ov.CompiledModel


使用 CLI 工具, Convert a model with `mo` command-line tool
* mo 是一个跨平台工具, 主要用于 训练和部署环境之间的转化
* 可以执行 静态模型分析, 调整深度学习模型, 用于在目标设备上最佳化运行
* 与 `mo.convert_model` 方法基本相同


具体流程为
* ONNX, PaddlePaddle, TensorFlow model file path
* -> mo
* IR  即 OpenVINO IR 表示文件 (网络结构的描述.xml  权重文件.bin)
* -> read_model()
* ov.Model
* -> compile_model()
* ov.CompiledModel




## 4.2. Model Optimization Guide - Model Optimization and Compression

模型优化是 OpenVINO 工作流中的一个 optional 模块, 通过离线的各种优化方法来提高模型的性能.  例如:
* 量化 quantization 
* 剪枝 pruning
* 预处理 preprocessing optimization  等等.

OpenVINO 提供了几种不同的工具用于在不同阶段对模型进行优化
* Model Optimizer : 用于对模型的参数进行调整, 包括但不限于 mean/scale values, batch size, RGB vs BGR input channels. 以及其他用于加速推论的预处理, 详细内容在 Model Preparation 章节

* Post-traning Quantization : 用于在学习后量化优化 DL 模型, 不需要 retraining or fine-tuning
  * 在 accuracy 与 performance 的 trade-off 中受到一定程度的限制, 精度在某些情况下会降低
  * 因此 training-time optimization 能够获得更好的优化效果
* Training-time Optimization : 用于在学习中对模型进行量化, supports methods like `Quantization-aware Training` and `Filter Pruning`. 
  * NNCF优化后的模型可以在 OpenVINO 的工作流中使用  



### 4.2.1. Quantizing Models Post-training - Post-training Quantization


训练后优化是通过一些方法使得不经过 retraining or fine-tuning 来优化模型使其更加 hardware-friendly.  

最为流行和广泛传播的方法是 8-bit post-training quantization 
* 容易使用
* 精度保存较高
* 显著提高推论性能
* 适用于许多 hardware available in stock, 因为很多硬件都支持原生 8-bit 计算  

通过 8 位整数量化, 来将权重和激活的精度降低至 8 位, 对比 32 位浮点数可以使得模型占用空间减少接近 4 倍, 并且能够显著提高推理速度.  这主要是因为降低精度后所需要的吞吐量降低. 该过程可以在实际推理之前离线完成, 模型会被转化成量化表示, 并且不需要原本训练框架的源代码 训练 pipeline 以及 训练数据.

在 OpenVINO 中, 为了应用 post-training methods, 需要提供:
* 一个浮点精度的模型 FP32 or FP16. 并且模型需要转化成 OpenVINO 的中间表达 (Intermediate Representation, IR) 用于将模型运行在 CPU 上
* 一个 代表性的 calibration dataset, 用于代表该模型的实际使用场景, 例如大概 300 个样本.
* 如果存在精度的要求限制, 则需要额外提供数据集用于 validation 以及实际计算 accuracy 的 matrics.

OpenVINO 目前在实际操作上支持两种 workflow 用于应用 post-training quantization 
* Post-training Quantization with POT - 即通过 OpenVINO IR 本身　(OpenVINO 2023.0 deprecated)
* Post-training Quantization with NNCF - 跨框架的解决方法, 适用于多种框架下的模型, 同时 API 更加简单

#### 4.2.1.1. Post-training Quantization with POT

使用 Post-training Optimization Tool (POT) 来实现 uniform integer quantization

该方法可以将 weights 和 activations  floating-point precision 移动到 integer precision, 例如 8-bit.
量化后模型会被转化, 具体的量化操作发生在推论的时候

#### 4.2.1.2. Post-training Quantization with NNCF (new)

NNCF 更多的用于 Training-time Quantization, 但也提供了 API 用于 Post-training 方法. 



### 4.2.2. Compressing Models During Training

Training-time model compression 可以提高模型的性能, 通过在训练的时候应用一些 optimizations (Such as quantization).  注意这里的总标题用的不是 Optimzing 而是 Compressing.  

Generally, training-time model optimization results in better model performance and accuracy than post-training optimization, but it can require more effort to set up.
能够获得更好的 performance and accuracy, 但是步骤较为繁琐.  

OpenVINO 的 training-time model compression 通过外置的 NNCF tool 来实现. NNCF 是一个 python 库, 可以嵌入到 Pytorch 和 TensorFlow 的 training pipelines.  

因此, 要想通过 OpenVINO 来应用 training-time compression. 需要使用 NNCF 模组并且:
* A floating-point model from the PyTorch or TensorFlow framework.
* A training pipeline set up in the PyTorch or TensorFlow framework.
* Training and validation datasets.
  
尽管如此, 通过 NNCF 为训练添加压缩优化仅仅只需要几行代码, 对压缩优化的配置并不是通过代码而是通过 configuration file 来指定的.  
NNCF学习后的模型可以通过 OpenVINO 环境推论, 即不再能够通过原本的学习框架推论  

NNCF 的 Training-time compression methods:
* Quantization 量化学习
  * 将模型中的 weights 和 activation values 从高精度格式 (例如 float32) 转化为低精度格式, 用于降低模型的内存占用以及延迟,
  * NNCF 使用 quantization-aware training 来量化模型, 通过为模型中插入用于模拟低精度效果的节点, 使得模型在训练的时候能够考虑到量化误差
  * 这使得模型在实际量化的时候能够获得更高的精度, NNCF 官方支持的方法为 uniform 8-bit quantization
* Filter pruning 剪枝
  * 使得模型在训练的时候会根据 filter 的重要程度来修建模型的权重, 将不重要的 filter 的 convolutional layer 的权重设置为 0 
  * 具体实现为, 在 fine-tuning 的时候 将一部分不重要的 layer 检测出, 并设置权重为 0, 在 fine-tuning 结束后, zeroed-out filters 会被从网络中删除.  
* 一些更加高级的实验性的方法 : state-of-the-art compression techniques that are still in experimental stages of development and are only recommended for expert developers.
  * Mixed-precision quantization
  * Sparsity
  * Binarization


#### 4.2.2.1. Quantization-aware Training (QAT) - 通过 NNCF 来实现量化学习

记录了基于 PyTorch 或 TensorFlow 的详细 量化学习工作方法  

要点
* nncf 要在 torch 之后立刻导入
* 从 nncf 导入 NNCFConfig

```python
import torch
import nncf  # Important - should be imported right after torch
from nncf import NNCFConfig
from nncf.torch import create_compressed_model, register_default_init_args

nncf_config_dict = {
    "input_info": {"sample_size": [1, 3, 224, 224]}, # input shape required for model tracing
    "compression": {
        "algorithm": "quantization",  # 8-bit quantization with default settings
    },
}

nncf_config = NNCFConfig.from_dict(nncf_config_dict)
nncf_config = register_default_init_args(nncf_config, train_loader) # train_loader is an instance of torch.utils.data.DataLoader

# 对进行优化 wrap the original model object with the create_compressed_model() API 
model = TorchModel() # instance of torch.nn.Module
compression_ctrl, model = create_compressed_model(model, nncf_config)

# 如果要使用分布式 GPU 进行多 GPU 训练的话, 要调用该接口
compression_ctrl.distributed() # call it before the training loop

# tune quantized model for 5 epochs as the baseline
for epoch in range(0, 5):
    # NNCF 特殊的 epoch_step 管理 API
    compression_ctrl.scheduler.epoch_step() # Epoch control API

    for i, data in enumerate(train_loader):
        # 同理
        compression_ctrl.scheduler.step()   # Training iteration control API
        ... # training loop body

# 通过 pytorch 训练的模型只能导出为 onnx
compression_ctrl.export_model("compressed_model.onnx")

# checkpoint 的保存与载入, NNCF 训练的模型具有特殊的 checkpoint 格式
checkpoint = {
    'state_dict': model.state_dict(),
    'compression_state': compression_ctrl.get_compression_state(),
    ... # the rest of the user-defined objects to save
}
torch.save(checkpoint, path_to_checkpoint)

# 通过 torch.load 获取 checkpint object
resuming_checkpoint = torch.load(path_to_checkpoint)
# 获取对应 NNCF 的压缩信息
compression_state = resuming_checkpoint['compression_state']
# 使用创建好的空模型 object, nncf_config, compression_state 重新 wrap model
compression_ctrl, model = create_compressed_model(model, nncf_config, compression_state=compression_state)
# 之后再单独读取 stat_dict
state_dict = resuming_checkpoint['state_dict']
model.load_state_dict(state_dict)
```

# 5. OpenVINO python API

API 的内容种类不多

* openvino.runtime    : openvino module namespace, exposing factory functions for all ops and other classes.
* openvino.runtime.op : Package: openvino.op Low level wrappers for the c++ api in ov::op.
* openvino.preprocess : Package: openvino Low level wrappers for the PrePostProcessing C++ API.
* openvino.frontend   : Package: openvino Low level wrappers for the FrontEnd C++ API.
* openvino.runtime.opset1 ~ openvino.runtime.opset11


## 5.1. openvino.runtime - 主要的运行时 API

openvino module namespace, exposing factory functions for all ops and other classes.

里面定义了数十种类, 以及一些最基本功能的函数  


## 5.2. openvino.runtime.Core

`class openvino.runtime.Core`
Bases: `openvino._pyopenvino.Core`

`__init__(self: openvino._pyopenvino.Core, xml_config_file: str = '') → None`

Core class represents OpenVINO runtime Core entity.  一个表示运行时核心实体的类   
用户应用其实可以创建复数个 Core class instances.  但这种情况下, 底层插件会被多次创建, 并且在 Core 实例之间不会共享, 因此不具有实际意义, 而推荐的做法就是每个应用程序只使用一个 Core 实例.  


Attribute: Core 实例的可访问的属性只有一个
* available_devices   : Returns devices available for inference Core objects goes over all registered plugins.


### 5.2.1. 实际方法

Property 可以参考 https://docs.openvino.ai/2023.0/groupov_property_c_api.html#doxid-group-ov-property-c-api

属性配置接口
* `get_property(*args, **kwargs)`   : 重载方法
  * `get_property(self: openvino._pyopenvino.Core, device_name: str, property: str) -> object`  : 获取某个 device 的专有属性
  * `get_property(self: openvino._pyopenvino.Core, property: str) -> object`                    : 获取 Core 本身的属性  
  * `device_name` 是 string 输入  
  * `Property` 可以是 string 也可以是 Property 实例


* `set_property(*args, **kwargs)`   : 重载方法
  * `set_property(self: openvino._pyopenvino.Core, properties: Dict[str, object]) -> None`    : 字典输入设置多个属性
  * `set_property(self: openvino._pyopenvino.Core, property: Tuple[str, object]) -> None`     : Tuple 输入设置单个属性
  * `set_property(self: openvino._pyopenvino.Core, device_name: str, properties: Dict[str, object]) -> None`  : 字典输入为 device 设置多个专有属性
  * `set_property(self: openvino._pyopenvino.Core, device_name: str, property: Tuple[str, object]) -> None`   : Tuple 输入为 device 设置单个专有属性
  * property 一律是 string



模型载入相关接口
* `read_model(*args, **kwargs)` : 重载方法
  * Reads models from IR / ONNX / PDPD / TF and TFLite formats.
  * `read_model(self: openvino._pyopenvino.Core, model: bytes, weights: bytes = b’’) -> openvino._pyopenvino.Model`  : 从各种其他框架的实例中读取模型
    * weights: 用于 OpenVINO IR 读取时候的 .bin 输入
  * `read_model(self: openvino._pyopenvino.Core, model: str, weights: str = ‘’) -> openvino._pyopenvino.Model` : 参数为模型的路径
  * 其他的重载也是同理, 即 model 可以是 实例, 路径, 或者打开的 IO 流, weight 可以是比特流, Tensor 或者路径
* `compile_model(model: Union[openvino._pyopenvino.Model, str, pathlib.Path], device_name: Optional[str] = None, config: Optional[dict] = None) → openvino.runtime.ie_api.CompiledModel`
  * 创建一个 compiled model.  可以直接调用而不经过 read_model.  用户可以创建任意多个 compiled_model 并且并发的执行它们
  * `model` : 已经从 read_model 读取的 IR 模型, 或者 path to a model in IR / ONNX / PDPD / TF and TFLite format.
  * `device_name` (str), Name of the device to load the model to. If not specified, the default OpenVINO device will be selected by AUTO plugin.
  * `config` (dict, optional) – Optional dict of pairs: (property name, property value) relevant only for this load operation.
  * Return type  : openvino.runtime.CompiledModel


### 5.2.2. 隐藏方法

## 5.3. openvino.runtime.CompiledModel

`class openvino.runtime.CompiledModel(other: openvino._pyopenvino.CompiledModel)`
Bases: `openvino._pyopenvino.CompiledModel`

CompiledModel 代表了 一个为指定设备编译好的模型, 包括应用了多种 optimization transformations, 以及 mapping to compute kernels


### 5.3.1. 隐藏方法

* `__call__(inputs: Optional[Union[dict, list, tuple, openvino._pyopenvino.Tensor, numpy.ndarray]] = None, shared_memory: bool = True) → openvino.runtime.utils.data_helpers.wrappers.OVDict`
  * 简易的调用推论的方法, Infers specified input(s) in synchronous mode. 在同步模式下进行推论.  
  * Blocks all methods of CompiledModel while request is running. 可能与 OpenVINO 的并发有关系.  
  * 实际的执行流程是
    * creates new temporary InferRequest
    * run inference on it
    * 创建好的 InferRequest 会作为一个属性存储在 CompiledModel 实例里面, 用于之后的 __call__ 调用
  * 官方推荐显式的定义一个 InferRequest 并利用它来进行推论, 可以实现配置其他高级设置用于最优化性能.  

### 5.3.2. 实际方法

* `create_infer_request() → openvino.runtime.ie_api.InferRequest`
  * 创建一个 inference request object used to infer the compiled model.
  * 创建的 request 实例会自动分配 input output Tensors
  * Return type :  `openvino.runtime.InferRequest`


* `get_property(self: openvino._pyopenvino.CompiledModel, property: str) → object`   : 获取当前 model 的属性
* `set_property(*args, **kwargs)`   : 重载方法
  * `set_property(self: openvino._pyopenvino.CompiledModel, properties: Dict[str, object]) -> None`  : 字典输入为当前 model 配置属性
  * `set_property(self: openvino._pyopenvino.CompiledModel, property: Tuple[str, object]) -> None` ： Tuple 输入为当前 model 配置属性 

## 5.4. openvino.runtime.InferRequest

`class openvino.runtime.InferRequest(other: openvino._pyopenvino.InferRequest)`
Bases: `openvino.runtime.utils.data_helpers.wrappers._InferRequestWrapper`

InferRequest class represents infer request which can be run in asynchronous or synchronous manners.  
一个推论请求的类, 代表了一个可以被同步或者异步方式运行的推论请求  


### 5.4.1. 实际方法

* `infer(inputs: Optional[Any] = None, shared_memory: bool = False) → openvino.runtime.utils.data_helpers.wrappers.OVDict`
  * 基本上与 `CompiledModel.__call__()` 相同
  * Infers specified input(s) in synchronous mode. Blocks all methods of InferRequest while request is running. Calling any method will lead to throwing exceptions.


* `get_profiling_info(self: openvino._pyopenvino.InferRequest) → List[ov::ProfilingInfo]`
  * 获取所有节点的 profile_info
  * 按照每一层进行性能测量, 可以用来获取最耗时的操作. 但是要注意, 并不是所有的 plugins 都能提供有意义的数据  
  * Return type : `List[openvino.runtime.ProfilingInfo]`
