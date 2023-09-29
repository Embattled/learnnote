# 1. Neural Network Compression Framework (NNCF)

NNCF provides a suite of advanced algorithms for Neural Networks inference optimization in OpenVINO™ with minimal accuracy drop.
* 提供了一套   `post-training` 和 `training-time` 的算法, 用于 OpenVINO 中的神经网路推理优化
* 其中 OpenVINO 的 taining-time 优化必须要通过 NNCF 来实现.  

NNCF is designed to work with models from PyTorch, TensorFlow, ONNX and OpenVINO™.

整个组件被整合成了一个独立于 OpenVINO 环境 python 的包, 可以方便的为 Pytorch 和 TensorFlow 添加不同的压缩算法.  
注意, 通过 NNCF 进行优化的模型不再属于原本的框架, 而是属于 OpenVINO IR 的表达, 要进行推论的话则需要使用 OpenVINO Runtime.  

NNCF 的官方推荐工作流:
1. 优先进行  `Post-training quantization` , 并进行性能比较
2. 如果性能损失过大, 进行 `Quantization-aware training` 来提高精度同时保持优化后的推论速度
3. 如果量化后的模型仍然速度不够快, 进行 `Filter Pruning` 来进一步删减模型的参数来提高模型的速度.  

# 2. NNCF configuration file

https://openvinotoolkit.github.io/nncf/schema/

顶层节点包括
* input_info `Required`
* target_device
* compression
* accuracy_aware_training
* compression_lr_multiplier
* log_dir


input_info : 描述模型的具体输入大小, NNCF 会根据模型的输入大小来生成对应的 dummpy Tensor 用于作为 forward 的默认参数, 如果不设置的话 forward 里的 tensor 会作为位置参数. 是 `single_object_version` 或 `array_of_objects_version` 其中之一
* single_object_version : 
  * `type` : tensor 的数据类型, Data type of the model input tensor, `float` 之类的
  * `sample_size`  : Shape of the tensor expected as input to the model. 用列表表示的模型输入 shape `[1, 4, 512, 512],`
* array_of_objects_version: single 版本的列表封装, 包含多个实例


target_device: 描述要优化的对应目标平台, 默认值 `ANY`
* `ANY` : 任何平台都是适应的兼容性优化
* `TRIAL` : use a custom quantization schema.
* `CPU` `GPU` `VPU` `CPU_SPR`



# 3. NNCF - Top level API

位于 nncf. 直下的顶层API, 是主要的使用对象

## 3.1. Class

顶层类

### 3.1.1. NNCFConfig

`from nncf import NNCFConfig` 量化学习的最关键的类, 用于配置整个量化学习过程  
Contains the configuration parameters required for NNCF to apply the selected algorithms.  

会从一个 dict 或者 JSON 文件中读取内容, 并依据相应的 schema 进行检查  


* `class nncf.NNCFConfig(*args, **kwargs)`  类本身的构造函数并不经常使用, 主要通过类方法来获取对应的实例
* `classmethod from_dict(nncf_dict)`        : 从一个 dict 中读取配置
* `classmethod from_json(path)`             : 从一个 JSON 文件中读取配置
* `static schema()`                         : 用于获取 用来检测对应输入是否合法的 `JSONSchema`  


register_extra_structs(struct_list)
get_redefinable_global_param_value_for_algo(param_name, algo_name)

# 4. nncf.torch

NNCF PyTorch functionality.


```py
import torch
import nncf  # Important - should be imported right after torch
```

## 4.1. create_compressed_model

创建优化后的模型的核心接口 

`nncf.torch.create_compressed_model(model, config, compression_state=None, dummy_forward_fn=None, wrap_inputs_fn=None, wrap_outputs_fn=None, dump_graphs=True)`



返回值:
* `Tuple[nncf.api.compression.CompressionAlgorithmController, nncf.torch.nncf_network.NNCFNetwork]`
* CompressionAlgorithmController: 用于控制压缩算法的控制器
* NNCFNetwork : 经过 NNCF 压缩封装后的模型实例 

