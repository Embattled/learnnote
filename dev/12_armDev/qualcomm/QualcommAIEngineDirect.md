# 1. Qualcomm Neural Network (QNN SDK)

https://www.qualcomm.com/developer/software  
https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk  



位于 高通开发者界面之下, 算是主推的一个功能

Qualcomm® AI Engine Direct is also referred to as Qualcomm Neural Network (QNN) in the source code and documentation.
This document provides a reference guide for Qualcomm® AI Engine Direct software development kit (SDK).


# 2. Quantization

高通 QNN SDK 中关于 量化的详细介绍

非量化模型使用 32 floating point 存储参数.  

量化模型则使用其他固定位数的 float 表示参数.  
通常是 fixed point 8bit weights 以及 8 or 32bit biases  

高通的 fixed point 参数表示于 Tensorflow 量化模型的规格相同  

通常根据推理平台, 需要首先确定 是否真的需要量化模型   

以下信息应该是针对高通家的产品  
* CPU   : 非量化, 因为量化模型推理目前不兼容 CPU 后端
* GPU   : 非量化, 同样因为不兼容量化模型
* DSP, HTP, HTA : 使用该后端都需要量化模型


## 2.1. Quantization - 通用量化算法


## 2.2. Quantization Modes - 量化模式

QNN 支持的 量化模式  

Quantization Mode 通常指代要使用的 量化范围 Quantization range

* TF    : 使用真实的数据确定 min/max, 同时调整范围以确保
  * 覆盖的区间最小 (精度最高)
  * 区间能够覆盖零值 0.0
  * 看起来就是广义上的非对称区间

* Symmetric : 采用基于 TF 的对称量化区间
  * `new_max = max(abs(min), abs(max))`
  * 调整区间为 `(-new_max, max) `

* Enhanced  : 加强模式, 作为 附加参数启用
  * 仍然会保证覆盖 0.0 
  * 会采用同真实的 `min/max` 所不同的区间, 有可能会使得一部分值范围的 weight, activation 无法落在区间之内, 但是会使得最终综合精度最高
  * 主要应对输入的 `long tails` 情况, 对于较少出现概率的极大值有较好的相性

* TF Adjusted   : 只适用于将模型量化为 8 bit fix point
  * 并非 TF 的真实 min/max 也不是 Enhanced 的应对 long tail 的 min/max
  * 而是针对 **降噪模型** 的特殊 range
  * 会根据需要, 提高 max 或者降低 min

## 2.3. Mixed Precision and FP16 Support - 混合精度以及 FP16

Mixed Precision 模式 支持在单个模型中混合采用 integer 和 float, 以及支持其他长度的整数  e.g. int16  
(所以说使用 int16 本身就代表了使用 Mixed Precision?)  

* 数据格式的转换 ops 会自动插入在 activation precision or data type 位宽不同的操作之间  
* Graph 可以存在 mix of `floating-point and fixed-point data types`.    
* 每一个 op 本身也可以有 different precision weights and activations
  * 对于特殊的 op, 例如 inputs outputs and parameters(weights/biases) will be floating-point or all will be integer type
  * 即 必须统一 是浮点还是 整数  
* 对于指定的 backends, 需要参照资料来确定 supported weight/activation bit widths for a particular op
  * 参照文档的 Operations 章节
  * 

FP16 作为特殊的浮点数, 一方面可以整个模型都转为 FP16  
也可以在 整数和小数混合的 Mixed precision graphs 中为 float ops 选择 FP16 或 FP32  
(这里说的应该是不能单独为某一层选择 FP16 或者 FP32)  


### 2.3.1. Non-quantized Mode

无量化模型, 或者说无法量化的时候执行的模式  

因为 –input_list flag is not given  

权重和激活都以 float 形式
* Non-quantized FP16 : 全模型 FP16 模式, `–float_bw 16`
* Non-quantized FP32 :  模型只是转为 QNN 的格式?  所有权重和激活都是维持 FP32
  * If `–float_bw` is absent from command line or `–float_bw 32` is given, all activation and weight/bias tensors use FP32 format.

### 2.3.2. Quantized Mode


In this mode calibration images are given (`–input_list` is given) to converter  
The converted QNN model has fixed point tensors for activations and weights.  

* No override : 无 override 的 所有 layer 进行默认量化
  * If no `–quantization_overrides` flag is given `with an encoding file`
  * all activations are quantized as per `–act_bw` (default 8)
  * parameters are quantized as per `–weight_bw/–bias_bw` (default 8/8) respectively.
* Full override : 全局 override 
  * If `–quantization_overrides` flag is given along `with encoding file` specifying encodings for all ops in the model. 
  * In this case, the bitwidth with be set as `per JSON for all ops` defined as `integer/float` as `per encoding file`
  * (dtype=’int’ or dtype=’float’ in encoding json).
* Partial override:
  * If `–quantization_overrides` flag is given along `with encoding file` specifying partial encodings
  * (i.e. encodings are missing for some ops), the following will happen.
    * (说明了一些特殊情况)
    * Layers for which encoding are available in json are encoded in same manner as full override case.


下面的 json 作为示例, 说明了如何将某一个层 mark 为指定的精度  


所有的层都指定
```json
{
   "activation_encodings": {
       "data_0": [
           {
               "bitwidth": 8,
               "dtype": "int"
           }
       ],
       "conv1_1": [
           {
               "bitwidth": 8,
               "dtype": "int"
           }
       ],
       "conv2_1": [
           {
               "bitwidth": 32,
               "dtype": "float"
           }
       ],
       "conv3_1": [
           {
               "bitwidth": 8,
               "dtype": "int"
           }
       ]
   },
   "param_encodings": {
       "conv1_w_0": [
           {
               "bitwidth": 8,
               "dtype": "int"
           }
       ],
       "conv1_b_0": [
           {
               "bitwidth": 8,
               "dtype": "int"
           }
       ],
       "conv2_w_0": [
           {
               "bitwidth": 32,
               "dtype": "float"
           }
       ],
       "conv2_b_0": [
           {
               "bitwidth": 32,
               "dtype": "float"
           }
       ],
       "conv3_w_0": [
           {
               "bitwidth": 8,
               "dtype": "int"
           }
       ],
       "conv3_b_0": [
           {
               "bitwidth": 8,
               "dtype": "int"
           }
       ]
   }
}
```


只指定特殊的层, 其他的层的 精度按照  `–act_bw/–weight_bw/–bias_bw` respectively.

```json
{
   "activation_encodings": {
       "conv2_1": [
           {
               "bitwidth": 32,
               "dtype": "float"
           }
       ]
   },
   "param_encodings": {
       "conv2_w_0": [
           {
               "bitwidth": 32,
               "dtype": "float"
           }
       ],
       "conv2_b_0": [
           {
               "bitwidth": 32,
               "dtype": "float"
           }
       ]
   },
   "version": "0.5.0"
}
```