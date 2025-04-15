# 1. AI Model Efficiency Toolkit (AIMET)

https://www.qualcomm.com/developer/software/ai-model-efficiency-toolkit
https://docs.qualcomm.com/bundle/publicresource/topics/80-78122-3/?product=1601111740010414

仓库  
https://github.com/quic/aimet

文档:  
https://quic.github.io/aimet-pages/releases/latest/index.html


高通的用于提高量化模型精度的 Toolkit  

看介绍好像是高通提供的 QAT 量化库  


# 2. Quick Start

python 3.10  
ubuntu 22.04  


安装: `python3 -m pip install aimet-torch`  

验证, 为一个 tensor 进行 8bit 量化  

```py
import aimet_torch.quantization as Q
scale = torch.ones(()) / 100
offset = torch.zeros(())
out = Q.affine.quantize(x, scale, offset, qmin=-128, qmax=127)
print(out)

```


量化一个模型

```py
from torchvision.models import mobilenet_v2

model = mobilenet_v2(weights='DEFAULT').eval().to(device)


# 导入组件, 后量化
from aimet_common.defs import QuantScheme
from aimet_common.quantsim_config.utils import get_path_for_per_channel_config
from aimet_torch.quantsim import QuantizationSimModel

sim = QuantizationSimModel(model, 
                           dummy_input,
                           quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                           config_file=get_path_for_per_channel_config(),
                           default_param_bw=8,
                           default_output_bw=16)


# 使用标定数据进行 标定
def forward_pass(model):
    with torch.no_grad():
        model(torch.randn((10, 3, 224, 224), device=device))

sim.compute_encodings(forward_pass)



# 评测数据
output = sim.model(dummy_input)
```




# 3. Installation

`python3 -m pip install https://github.com/quic/aimet/releases/download/2.2.0/aimet_torch-2.2.0+cu121-cp310-none-any.whl -f https://download.pytorch.org/whl/torch_stable.html`


`python3 -m pip install https://github.com/quic/aimet/releases/download/2.2.0/aimet_torch-2.2.0+cu121-cp310-cp310-manylinux_2_34_x86_64.whl -f https://download.pytorch.org/whl/torch_stable.html`


# 4. Quantization Simulation Guide

# 5. Feature Guide - Optimization techniques

最优化技巧

## 5.1. Batch norm folding



## 5.2. Cross-layer equalization

跨层均衡 (CLE) 工具  


该特性提供了如下几个技术

* Batch Norm Folding
  * 将 Batch Norm 层折叠 (整合) 到相邻的卷积层和线性层中  
  * 从原理上属于 CLE 工具, 但是详细说明在别的章节中
* Cross Layer Scaling
  * 同一层中 **不同通道** 的参数范围差异很大
  * Cross Layer Scaling 会通过缩放的技术来使得不同通道具有相似的范围, 以便于所有通道使用相似的量化参数, 即 Tensor Range
  * 跨层的缩放, 来均衡连续的层的每个通道的权重分布  
    * 从目的上有点类似于 per-channel range 但是对比 per-channel, 不会额外增加参数  
    * 从理论的性能上看, 只能解决层之间的范围不匹配问题
* High Bias Fold
  * 在 Cross Layer Scaling 的基础上进一步解决 Cross Layer Scaling 会出现的问题
  * Cross Layer Scaling 会导致某些层的 bias 参数过高
  * 那么将 bias 的一部分折叠到后续层的参数中
    * 该功能需要启用 batch norm parameters 


示例代码, 只提供了顶层 API 的用法  
```py
import torch
from torchvision.models import mobilenet_v2

# General setup that can be changed as needed
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = mobilenet_v2(pretrained=True).eval().to(device)
input_shape = (1, 3, 224, 224)

from aimet_torch.cross_layer_equalization import equalize_model

# Performs BatchNorm folding, cross layer scaling and high bias folding
equalize_model(model, input_shape)

```



# 6. API Reference

除了 torch, tensorflow 之外, 还直接提供了面向 onnx 模型的量化接口  

## 6.1. aimet_torch


### 6.1.1. aimet_torch.quantsim

创建量化模型的核心 接口  
为模型插入 FakeQuantize 层


`QuantizationSimModel` simulates quantization of a given model by converting all PyTorch modules into quantized modules with input/output/parameter quantizers as necessary.

```py
class aimet_torch.QuantizationSimModel(
  model, dummy_input, 
  quant_scheme=None, 
  rounding_mode=None,  # 弃用参数
  default_output_bw=8, 
  default_param_bw=8, 
  in_place=False, 
  config_file=None, 
  default_data_type=QuantizationDataType.int
)
# 参数说明
dummpy_input : # 用于获取 computational graph, 需要位于正确的设备上

quant_scheme : QuantScheme, optional  # 用于指定量化范围的标定逻辑,  似乎将会被弃用
# default value of quant_scheme has changed from QuantScheme.post_training_tf_enhanced to QuantScheme.training_range_learning_with_tf_init since 2.0.0
# and will be deprecated in the longer term.

default_output_bw:int # 激活的 bit 数
default_param_bw:int  # 参数的 bit 数
in_place # 替换性的创建模型 

config_file (str, optional) : # File path or alias of the configuration file. Alias can be one of { default, htp_v66, htp_v68, htp_v69, htp_v73, htp_v75, htp_v79, htp_v81 } (Default: “default”)
# 具有特殊优化的 配置文件, 或者直接是 平台

default_data_type : # 所有参数和激活的默认类型, 整数或者浮点 
# 由于不能使用 16bit 以下的浮点数  
# the mode default_data_type=QuantizationDataType.float is only supported with default_output_bw=16 or 32 and default_param_bw=16 or 32
```

类方法
* compute_encodings : 传入一个 回调函数, 用于 PTQ 量化模型, 在该方法调用之前, 模型无法进行前向传播  
  * `compute_encodings(forward_pass_callback, forward_pass_callback_args=<class 'aimet_torch.v2.quantsim.quantsim._NOT_SPECIFIED'>)`
  * 似乎是框架的关键方法

* `export(path, filename_prefix, dummy_input, *args, **kwargs)`
  * 导出量化模型, 包括
    * regular pytorch model without any simulation ops `ONNX 本身不包含量化层`
    * 量化编码导出为独立的 json file, 用于在目标设备的 runtime 中导入
    * 可选的导出 onnx
  * 参数
    * `path`: str 路径, 包括 model path and encodings, 应该是指定文件夹
    * `file_name_prefix`: 指定 model 和 encodings file 的 prefix, 相当于模型名称
    * `filename_prefix_encodings` : encodings file 使用独立的 prefix, 默认为 None
    * `export_model ` : bool, 是否导出 ONNX, 官方建议除非不想覆盖既存的 ONNX 模型, 否则总是为 True
    * `use_embedded_encodings` – If True, `another onnx model` embedded with fakequant nodes will be exported
      * 啥意思, 标准导出的 ONNX 还不带 FakeQuant?
    * `export_to_torchscript` – If True, export to torchscript. Export to onnx otherwise. Defaults to False.
      * True 的时候就不导出 ONNX 了?
    * `propagate_encodings` : 涉及到 one PyTorch ops results in multiple ONNX nodes, 默认为 False, 应该不用管
    * `onnx_export_args `: (OnnxExportApiArgs ) ONNX 的导出参数字典, 涉及到 ONNX 的 ops 版本等内容

```py
# export 的内部实现
if use_embedded_encodings:
    # 这个方法现在用不了
    self.save_model_with_embedded_quantization_nodes(self.model, path, filename_prefix, dummy_input,
                                                      onnx_export_args, export_to_torchscript, self._is_conditional)
else:
    # 导出为 torch_script
    if export_to_torchscript:
        self.export_torch_script_model_and_encodings(path, filename_prefix, filename_prefix_encodings,
                                                      model_to_export, self.model,
                                                      dummy_input,
                                                      self._excluded_layer_names)
    else:
        self.export_onnx_model_and_encodings(path, filename_prefix, model_to_export, self.model,
                                              dummy_input, onnx_export_args, propagate_encodings,
                                              self._module_marker_map, self._is_conditional,
                                              self._excluded_layer_names, quantizer_args=self.quant_args,
                                              export_model=export_model,
                                              filename_prefix_encodings=filename_prefix_encodings)





def pass_calibration_data(sim_model, use_cuda):
    data_loader = ImageNetDataPipeline.get_val_dataloader()
    batch_size = data_loader.batch_size

    if use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    sim_model.eval()
    samples = 1000

    batch_cntr = 0
    with torch.no_grad():
        for input_data, target_data in data_loader:

            inputs_batch = input_data.to(device)
            sim_model(inputs_batch)

            batch_cntr += 1
            if (batch_cntr * batch_size) > samples:
                break

sim.compute_encodings(forward_pass_callback=pass_calibration_data,
                      forward_pass_callback_args=use_cuda)
```


### 6.1.2. cross_layer_equalization

CLE 的包, 只有一个 API 接口

`aimet_torch.cross_layer_equalization.equalize_model(model, input_shapes=None, dummy_input=None)`
* The model is equalized in place.
* `input_shapes (Union[Tuple, List[Tuple], None])`, Shape of the input (can be a tuple or a list of tuples if multiple inputs)
* `dummy_input (Union[Tensor, Tuple, None])`

应用 CLE 不需要将模型创建为 SimQuantize, 单独调用该接口即可 (模型需要事先 prepar)


### 6.1.3. aimet_torch.model_preparer


使用了 pytorch 1.9+ 中新增的 new graph transformation feature  
执行用户所需的模型定义更改 (automates model definition changes)  
* 当 nn.Module 类型的模块被重复使用的时候, 会被展开 (unroll) 为独立的模块  
* 将前向传递中定义的函数转换为用于激活函数和元素函数的 nn.Module
  * changes functionals defined in forward pass to torch.nn.Module type for activation and elementwise function
  * 即如果使用了 nn.functional 中的函数性运算, 将其转为 nn.Module
  * 如果定义了 元素相加操作, 也转为 add 层

接口的限制跳过, 基本上都是常见的, 例如不支持 if-else 或者 loop 的 forward





* `aimet_torch.model_preparer.prepare_model(model, modules_to_exclude=None, module_classes_to_exclude=None, concrete_args=None)`
  * 使用 `torch.FX` 的 symbolic tracing API 来 prepare and modity pytorch model, 模型的预处理
    * 更换 nn.functional 
    * 对 reused/duplicate 模组更换为独立的 Module
* 参数
  * model : 要修改的模型
  * modules_to_exclude,  mmodule_classes_to_exclude: 指定排除 tracing 的模组(类)
  * concrete_args : 对于有控制流的类, 用于 指定参数用于 remove control flow or data structures ? 











