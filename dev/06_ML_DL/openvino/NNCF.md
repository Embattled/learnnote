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

# 2. Document - Usage

github 中的说明文档
https://github.com/openvinotoolkit/nncf/tree/develop/docs


文档结构在 2024 年中有一次结构更改
To align documentation with the latest changes in torch QAT


## 2.1. Post Training Compression

### 2.1.1. Post Training Quantization

后量化, 不需要重新训练模型, 使用 initial dataset 中的一个子集来标定量化常数
量化的实现原理写在了 LegacyQuantization 部分

模型后量化支持 PyTorch, TorchFX, TensorFlow, ONNX, and OpenVINO 格式
当后量化不满足精度要求的时候, 使用 QAT 进行量化感知学习

NNCF 提供了先进的 PTQ 算法, 包括
* MinMaxQuantization - 分析模型, 同时插入额外的基于 subset 标定的量化层
* FastBiasCorrection/BiasCorrection - 降低 量化前后的 bias errors 

使用方法: 需要提供一下实体
* 原始 model
* validation part of the dataset, 验证数据集
* Data Transformation Function (??)

使用方法
```py

# 1. 构建 Data Transformation Function
def trans_func(data_item):
    images, _ = data_item
    # 拿到输入数据
    return images

# 2. 创建 NNCF 数据集对象 nncf.Datset
# data_source : 一个 可迭代的 python 对象, 用于获取 data
# trans_cunc  : 第一步创建的 转换函数
calibration_dataset = nncf.Dataset(val_dataset, transform_fn)

# 3. 直接运行量化 pipeline 就行了
quantized_model = nncf.quantize(model, calibration_dataset)
```


**Data Transformation Function**
一个 NNCF 的专有概念

应该只是用来方便实现数据格式对齐的接口  
不同的 pipeline/框架 可能输入模型的数据格式不同, 这个函数专门用来实现格式转换  
最典型, 可能就是最主要的应对对象就是 onnx 的超麻烦的输入格式  

PyTorch, TorchFX, TensorFlow, OpenVINO 中
trans_func 的返回值直接输入模型  

ONNX 中
ONNX Runtime 接受的 pipeline 输入是 `Dict[str, np.ndarray]`, 即 输入的变量名称也需要传入

```py
# PyTorch, TorchFX, TensorFlow, OpenVINO 中的推理方法
for data_item in val_loader:
    model(transform_fn(data_item))

# onnx 的推理方法
sess = onnxruntime.InferenceSession(model_path)
output_names = [output.name for output in sess.get_outputs()]
for data_item in val_loader:
    sess.run(output_names, input_feed=transform_fn(data_item))
```


**DataSource**

nncf.Dataset 需要传入可迭代的数据集
`calibration_dataset = nncf.Dataset(data_source, transform_fn)`
该变量可以直接传入 `pytorch.Dataloader` or `tf.data.Dataset`

但是要注意这些 loader 中 batch_size 的情况, NNCF 支持标定 loader 按照 batch 来读取数据, 但是根据模型的不同, 对应的 batch 所代表的含义有所不同  
例如 transformers 或者其他非传统的结构, batch 的 axis 并不位于约定俗成的位置, 这种情况下 batch_size 不为 1 会导致标定错误  
具体原因为 
`It happens because certain models' internal data arrangements may not align with the assumptions made during quantization, leading to inaccurate statistics calculation issues with batch sizes larger than 1.`   

Please keep in mind that you have to recalculate the subset size for quantization according to the batch size using the following formula: `subset_size = subset_size_for_batch_size_1 // batch_size`  
最终 nncf 量化所接受的参数 `subset_size` 标定子集的大小也受 loader 的 batch_size 的影响, 需要用户 手动重新计算

### 2.1.2. Weights Compression



## 2.2. Training Time Compression


### 2.2.1. Quantization Aware Training (QAT) - Usage

Use NNCF for Quantization Aware Training

在既存的 PyTorch 或者 TensorFlow 项目中集成 NNCF 包 的整体简要文档
预设用户已经实现了训练 pipeline 并能够生成 floating point 的预训练模型  

The task is to prepare this model for accelerated inference by simulating the compression at train time. Please refer to this document for details of the implementation.
模拟训练时候的压缩

该文档不涉及 QAT 的实现详细, 具体实现写在了 `LegacyQuantization` 章节


**Basic usage 基本用法**

pytorch
```py
# 1.应用 后量化, 在应用后量化的同时模型的 QAT 训练也会一并完成初始化, 是NNCF新版本的改进点
model = TorchModel() # instance of torch.nn.Module
quantized_model = nncf.quantize(model, ...)

# 2.QAT训练
# QAT训练完全继承金 training pipeline 中, 完全不需要再附加任何 pipeline 的修改
# 唯一需要注意的是 模型需要置 model.train() 状态, 即关闭 Dropout 以及 DropConnect

# 3.导出QAT后的模型为 ONNX, 注意格式为 Onnx 但是无法通过 ONNXRuntime 运行
# To OpenVINO format
import openvino as ov
ov_quantized_model = ov.convert_model(quantized_model.cpu(), example_input=dummy_input)
```

TensorFlow
```py
model = TensorFlowModel() # instance of tf.keras.Model
quantized_model = nncf.quantize(model, ...)

# To OpenVINO format
import openvino as ov

# Removes auxiliary layers and operations added during the quantization process,
# resulting in a clean, fully quantized model ready for deployment.
stripped_model = nncf.strip(quantized_model)

ov_quantized_model = ov.convert_model(stripped_model)
```


**Saving and loading compressed models**

NNCF模型的完整信息包括模型参数本身以及 NNCF Config    

model : 包含了 模型的参数以及模型的拓扑结构  
NNCF config: 包括如何恢复模型的量化附加结构的信息   

NNCF config 可以通过接口 `nncf.torch.get_config` 从一个既存的压缩模型中获取  

而还原模型则通过 `nncf.torch.load_from_config`  

NNCF 的保存规则允许模型只存储 参数和 NNCF config 以及一个 dummy_input 即可完整的复原模型


```py
import nncf.torch

# save part, 量化过程
quantized_model = nncf.quantize(model, calibration_dataset)
# 只需要保存两个部分
checkpoint = {
    'state_dict': quantized_model.state_dict(),
    'nncf_config': nncf.torch.get_config(quantized_model),
    ...
}
torch.save(checkpoint, path)

# load part, 读取字典
resuming_checkpoint = torch.load(path)

nncf_config = resuming_checkpoint['nncf_config']
state_dict = resuming_checkpoint['state_dict']

# 传入 模型结构, nncf_config, dummy_input
quantized_model = nncf.torch.load_from_config(model, nncf_config, dummy_input)
# 单独再读取 模型参数
quantized_model.load_state_dict(state_dict)

```


```py
from nncf.tensorflow import ConfigState
from nncf.tensorflow import get_config
from nncf.tensorflow.callbacks.checkpoint_callback import CheckpointManagerCallback

nncf_config = get_config(quantized_model)
checkpoint = tf.train.Checkpoint(model=quantized_model,
                                 nncf_config_state=ConfigState(nncf_config),
                                 ... # the rest of the user-defined objects to save
                                 )
callbacks = []
callbacks.append(CheckpointManagerCallback(checkpoint, path_to_checkpoint))
...
quantized_model.fit(..., callbacks=callbacks)

from nncf.tensorflow import ConfigState
from nncf.tensorflow import load_from_config

checkpoint = tf.train.Checkpoint(nncf_config_state=ConfigState())
checkpoint.restore(path_to_checkpoint)

quantized_model = load_from_config(model, checkpoint.nncf_config_state.config)

checkpoint = tf.train.Checkpoint(model=quantized_model
                                 ... # the rest of the user-defined objects to load
                                 )
checkpoint.restore(path_to_checkpoint)

```

**Advanced usage**

NNCF 只支持 pytorch 原生结构的 压缩, 不支持自定义模块, 如果有自定义模块, 需要预先注册

```py
import nncf

@nncf.register_module(ignored_algorithms=[...])
class MyModule(torch.nn.Module):
    def __init__(self, ...):
        self.weight = torch.nn.Parameter(...)
    # ...
```
If registered module should be ignored by specific algorithms use ignored_algorithms parameter of decorator.

In the example above, the NNCF-compressed models that contain instances of MyModule will have the corresponding modules extended with functionality that will allow NNCF to quantize the weight parameter of MyModule before it takes part in MyModule's forward calculation.



### 2.2.2. Other Algorithms


#### 2.2.2.1. LegacyQuantization - Uniform Quantization with Fine-Tuning

量化的详细实现方法

uniform fake quantization 方法  

可以实现任意位数的 伪量化, 用于表示不同位宽的 weights and activations  

该方法 在 forward 的适合执行 differentiable sampling of the continuous signal, 用于实现对 整数推理的模拟  


**Common Quantization Formula**



# 3. Examples - 示例代码

# 4. NNCF configuration file

https://openvinotoolkit.github.io/nncf/schema/

顶层节点包括
* input_info `Required`
* target_device
* compression
* accuracy_aware_training
* compression_lr_multiplier
* log_dir


input_info : 描述模型的具体输入大小
  * 是 `single_object_version` 或 `array_of_objects_version` 其中之一
* 如果指定的话 ： NNCF 会根据模型的输入大小来生成对应的 dummpy Tensor 用于作为 forward 的默认参数
* 如果不设置的话 forward 里的 tensor 会作为位置参数. 
* single_object_version : 
  * `type` : tensor 的数据类型, Data type of the model input tensor, `float` 之类的
  * `sample_size`  : Shape of the tensor expected as input to the model. 用列表表示的模型输入 shape `[1, 4, 512, 512],`
* array_of_objects_version: single 版本的列表封装, 包含多个实例


target_device: 描述要优化的对应目标平台, 默认值 `ANY`
* `ANY` : 任何平台都是适应的兼容性优化
* `TRIAL` : use a custom quantization schema.
* `CPU` `GPU` `VPU` `CPU_SPR`



# 5. NNCF - Top level API

位于 nncf. 直下的顶层API, 是主要的使用对象

## 5.1. Class

顶层类

### 5.1.1. nncf.NNCFConfig

`from nncf import NNCFConfig` 量化学习的最关键的类, 用于配置整个量化学习过程  
Contains the configuration parameters required for NNCF to apply the selected algorithms.  

会从一个 dict 或者 JSON 文件中读取内容, 并依据相应的 schema 进行检查  

创建/ 构造函数
* `class nncf.NNCFConfig(*args, **kwargs)`  类本身的构造函数并不经常使用, 主要通过类方法来获取对应的实例, 例如
  * `nncf.torch.register_default_init_args`
* `static schema()`                         : 用于获取 用来检测对应输入是否合法的 `JSONSchema`  



类方法:
* `classmethod from_dict(nncf_dict)`        : 从一个 dict 中读取配置
* `classmethod from_json(path)`             : 从一个 JSON 文件中读取配置



其他方法:
* `register_extra_structs(struct_list)`
  * 在后期添加额外的 configuration 
  * struct_list `(List[nncf.config.structures.NNCFExtraConfigStruct])` – List of extra configuration structures.

* get_redefinable_global_param_value_for_algo(param_name, algo_name)


### 5.1.2. nncf.ModelType

Bases: enum.Enum  , 基于 python 的枚举类型  

定义了要被特殊考虑的模型类型, 当前版本 只是用来区分 transformer

也只有 TRANSFORMER  一种参数  


### 5.1.3. nncf.TargetDevice

同样也是基于 枚举的类型, 指定了目标终端的类型.  
```py
class TargetDevice(Enum):
    """
    Target device architecture for compression.

    Compression will take into account the value of this parameter in order to obtain the best performance
    for this type of device.
    """

    ANY = "ANY"
    CPU = "CPU"
    GPU = "GPU"
    VPU = "VPU"
    CPU_SPR = "CPU_SPR"
```


## 5.2. nncf.quantize

```py
nncf.quantize(model, calibration_dataset, preset=None, target_device=TargetDevice.ANY, subset_size=300, 
fast_bias_correction=True, model_type=None, ignored_scope=None, advanced_parameters=None)
```

主要的 Post-training quantization 函数   
* model : TModel, 要进行量化的 浮点模组, TModel ?
* calibration_dataset  : `nncf.Dataset` 一个代表数据集用来进行标定
* preset : 量化的预设模式 `nncf.QuantizationPreset`, 对称和非对称, 可选的模式有
  * performance : symmetirc quantization of weights and activations, 对权重和激活都启用对称量化
  * mixed   : 权重的对称量化, 激活的非对称量化
  * 默认值  : 对于 transformer model 启用 mixed, 对于其他的启用 performance  
* target_device : `nncf.TargetDevice` , 用于最佳化量化
* subset_size : int, subset 的大小, 用于指定 activation 统计的大小
* fast_bias_correction = True, 是否启用告诉 bias 校正算法
  * 如果置 false 的话, 精度更高, 但量化更慢, 消费内存更少
* model_type : `nncf.ModelType`  
  * 目前只是用来区分一般模型和 transformer 模型
* ignored_scope : `nncf.IgnoredScope` :
  * defined the `list of model control flow graph nodes` to be ignored during quantization.
* advanced_parameters : `nncf.quantization.advanced_parameters.AdvancedQuantizationParameters`
  * 细微的调整参数  
* 


# 6. nncf.torch

NNCF PyTorch functionality.

```py
import torch
import nncf  # Important - should be imported right after torch
```


## 6.1. Functions

nncf.torch 模组下的主要函数接口

### 6.1.1. register_default_init_args - 建立NNCF控制参数

```py
nncf.torch.register_default_init_args(nncf_config, train_loader, 
    criterion=None, 
    criterion_fn=None, train_steps_fn=None, validate_fn=None, 
    val_loader=None, 
    autoq_eval_fn=None, model_eval_fn=None, 
    distributed_callbacks=None, 
    execution_parameters=None, 
    legr_train_optimizer=None, 
    device=None)
Return type:
    nncf.NNCFConfig
```

文档中没有任何说明, 要查找只能依据各种 example
传入一个 nncf.NNCFConfig, 然后再返回一个 nncf.NNCFConfig

主要作用就是向 nncf.NNCFConfig 绑定标定用的数据集


### 6.1.2. create_compressed_model

创建优化后的模型的核心接口 

```py
def nncf.torch.create_compressed_model(model, config, 
  compression_state=None,
  dummy_forward_fn=None, 
  wrap_inputs_fn=None, wrap_outputs_fn=None,
  dump_graphs=True)
```

参数:
* config: `nncf.NNCFConfig` nncf的核心配置类
* compression_state `(Optional[Dict[str, Any]])` 
  * 压缩状态的核心 dict, 用于无差错的读取量化学习模型  
* dump_graphs : 默认值 True, 是否 dump `.dot` 格式的模型压缩前后的内部 graph representation. 输出会保存到 log directory


返回值:
* `Tuple[nncf.api.compression.CompressionAlgorithmController, nncf.torch.nncf_network.NNCFNetwork]`
* CompressionAlgorithmController: 用于控制压缩算法的控制器
* NNCFNetwork : 经过 NNCF 压缩封装后的模型实例 

### 6.1.3. load_state

`nncf.torch.load_state(model, state_dict_to_load, is_resume=False, keys_to_ignore=None)`
* nncf 框架下的 torch 模型加载, 在某些情况下稳定性要优于使用 torch 的接口    


参数:
* model : torch.nn.Module,  需要进行参数加载的 模型
* state_dict_to_load : dict, 标准的参数字典
* is_resume : 关键参数, 默认为 False
  * 主要用于在 state_dict 与模型中的参数网络不匹配的时候是否报错. 这个不匹配是双向的.
  * 通常情况下, 把一个未压缩的参数 dict 加载到一个已经应用过压缩算法的模型 (网络层有变动) 的时候, 置为 False.
  * 如果是把 压缩过的参数 dict 加载到已经应用过压缩的 模型, 则需要置 True 用于保证安全加载
* keys_to_ignore : `(List[str])`, 在加载过程中, 需要跳过 matching 过程的参数名称.  
* 返回值 : int, The number of state_dict_to_load entries successfully matched and loaded into model.





# 7. nncf.api 

非顶端接口的, 各种内部实现类  

## 7.1. nncf.api.compression

与整个模型压缩过程关联的各种内部控制类 , 其下没有函数只有类 

Classes
* CompressionAlgorithmController  : 用于整个压缩过程中的状态调整, 是其他几个类的总包  
* CompressionScheduler    : 用于学习过程中的 logic of compression method control 
* CompressionLoss         : 用于计算量化学习中的独特的 量化 loss, 在学习中通过加法添加到基本 loss 上来使用
* CompressionStage        : 描述了模型的 compression stage 
* CompressionAlgorithmBuilder   : 


### 7.1.1. CompressionAlgorithmController 

`class nncf.api.compression.CompressionAlgorithmController(target_model)`  

`Bases: abc.ABC` 该类在定义上是一个虚基类, 因此不能用来直接创建实例, 一般都是接口的返回值来获取  

用于控制整个压缩过程 : 例如 compression scheduler and compression loss.


成员:
* model: TModel, 获取该 controller 链接的模型, 可以用来进行模型导出
* (abstract property) loss: `CompressionLoss`, 虚成员, 用于获取 量化学习 loss
  * 在使用上却有点类似于接口  `ctrl.loss()` 暂不清楚原因
* (abstract property) scheduler: `CompressionScheduler`, 获取该 Controller 链接的 scheduler 
  * 一般都是通过 ctrl 获取 `ctrl.scheduler`
* (abstract property) name: str : 压缩算法的名称, 是唯一的, 没有实际确认过是否包含了一些状态字符
* (abstract property) compression_rate: float: 返回压缩比例, 0 to 1 
  * 可能是 the sparsity level , or the ratio of filters pruned
* (abstract property) maximal_compression_rate: 返回该 controller 所应用的算法的最大压缩比例
  * 可能会被用于 if 语句中用于控制压缩状态
  

虚方法: 依据压缩算法的不同而有不同具体实现  
* (abstract) **load_state**(state): 加载 ctrl 的状态, from the map of algorithm name to the dictionary with state attributes.
  * 参数 `state (Dict[str, Dict[str, Any]])`
  * 一个字典, 这个字典的格式看起来好像是个二重字典. 其中外层的是一个 字典的字典
* (abstract) **get_state**() : 对应的获取 ctrl 的状态, 与 load_state 对应的使用
  * 返回值 `Dict[str, Dict[str, Any]]`
* (abstract) **get_compression_state**() : 另一种获取 state 的接口 Returns the compression state
  * 包括了 builder and controller state.
  * 这个接口主要用于 `create_compressed_model` 中的 `compression_state` 参数, 因此是另一种无偏差的恢复 ctrl 的方法 
  * 返回值 : `Dict[str, Any]`, Compression state of the model to resume compression from it
* (abstract) **statistics**(quickly_collected_only=False) : 返回一个 nncf 的 Statistic 类保存了 压缩的统计
  * 参数: quickly_collected_only , true 的话可以加快统计过程, 需要在 epoch 中追踪统计的话可以置 true
  * 返回值 : `nncf.api.statistics.Statistics`
* (abstract) **export_model**(save_path, save_format=None, input_names=None, output_names=None, model_args=None)
  * 导出模型, 用于部署. 默认为 onnx 格式
  * 该操作有可能会产生 non-ONNX-standard operations and layers to leverage full compressed/low-precision potential of the OpenVINO toolkit, 因此可能无法通过 onnx runtime 来运行.  
  * save_format `(Optional[str])` : 不指定的话会使用 默认格式
  * input_names, output_names  `(Optional[List[str]])` : 为输入和输出赋予名字, 不知道在哪种平台能用得上
  * model_args  ` (Optional[Tuple[Any, Ellipsis]])` : 完全特殊的内容, 不理解
  * 无返回值
* (abstract) **disable_scheduler**() : 一般 压缩的 scheduler 都会动态调整压缩率, 而 disable 后就不会动态调整了



实方法: 不依据压缩算法的, ctrl 本身应该具有的功能
* **compression_stage**() : 返回 `CompressionStage` 类
  * 具体需要参考该返回值的类
  * 文档说这个类主要用于 saving best checkpoints to distinguish between uncompressed, partially compressed, and fully compressed models. 不太理解
* **strip_model**(model, do_copy=False) : strip , 在 nncf 中是剥离用于在压缩训练过程中辅助的 layer. 
  * model (TModel) – The compressed model. 为啥 ctrl 类实例里的成员函数还需要输入 model... 可能该函数被实现为了 类方法?
  * do_copy  : 是否在 copy 上进行 strip 
  * 返回值 : Tmodel, The stripped model.
* **strip**(do_copy=True) : 似乎是上述方法的 对象版本
  * Returns the model object with as much custom NNCF additions as possible removed while still preserving the functioning of the model object as a compressed model.
  * 在保留作为压缩模型的功能的前提下, 尽可能剔除 nncf 的附加. strip 后的模型可以直接用 torch 的 export 来导出  
  * 但是发现了一个问题: `RuntimeError: Converting nncf quantizer module to torch native only supports for num_bits in [8]`
  * 超过 8 bit 的量化学习模型无法被导出为 strip 模型
* **prepare_for_export**() : 准备模型进行导出, exporting to a backend-specific model serialization format
  * 无参数, 无返回值, 需要参考 sample 学习具体使用场景


### 7.1.2. CompressionScheduler

`class nncf.api.compression.CompressionScheduler`

`Bases: abc.ABC` 虚基类, 该 scheduler 的功能主要是
* logic of compression method control during the training process. 
* 用于随着训练的 epoch 和 step 动态调整 hyperparameters 
* For example, the sparsity method can smoothly increase the sparsity rate over several epochs.
* 如果是稀疏方法的话, 就会动态增加稀疏率

The step() and epoch_step() methods of the compression scheduler must be called at the beginning of each training step and epoch, respectively:  
要正确的启用 scheduler 的话, 需要按照如下方法来定义学习过程   
```py
for epoch in range(0, num_epochs):
    scheduler.epoch_step()
    for i, (x, y) in enumerate(dataset):
         scheduler.step()
         ...
```


虚方法 abstract 
* step(next_step=None) : 在每个学习 step 的开始被调用.
  * next_step : 某些训练的代码实现可能会有 step 索引, 可以用作该方法的参数
* epoch_step(next_epoch=None) : 在每个 epoch 的开始被调用
  * next_epoch : 同上
* load_state(state) ,  get_state() .  参数返回值 : `Dict[str, Any]`
  *  the compression scheduler state, but does not update the state of the compression method.
  *  仅针对 scheduler 的 state
  *  不太清楚实际中的必要应用场景

### 7.1.3. CompressionLoss

`class nncf.api.compression.CompressionLoss`  `Bases: abc.ABC`

主要用于计算量化学习的 additional loss, 具体表现为通过 model graph 来测量 variables 和 activations 的 loss.  
For example, the $L_0$-based sparsity algorithm calculates the number of non-zero weights in convolutional and fully-connected layers to construct the loss function.


成员函数: 
* (abstract) calculate(*args, **kwargs) , `__call__(*args, **kwargs)`
  * 计算 compression loss
  * 因为实现了 `__call__` 所以实际中会用 `ctrl.loss()` 来调用
* (abstract) load_state(state) get_state()
  * 状态的提取和读取, 同样的, 想想不来需要单独使用 loss 的状态提取读取应用场景  

## 7.2. nncf.api.statistics

关于 NNCF 压缩过程中的统计信息  

该模组下没有函数, 只有一个类

`class nncf.api.statistics.Statistics`  Base: abc.ABC

* Contains a collection of model- or compression-related data and provides a way for its human-readable representation.
* 保存了 压缩过程中的信息, 同时提供了一个可读的接口函数
* 应该更多的用于 经验者的 debug