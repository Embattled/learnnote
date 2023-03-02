- [1. Pytorch](#1-pytorch)
  - [1.1. install](#11-install)
  - [1.2. verification](#12-verification)
  - [1.3. API](#13-api)
- [2. torch.Tensor 张量类](#2-torchtensor-张量类)
  - [2.1. torch.Tensor 的格式](#21-torchtensor-的格式)
  - [2.2. 类属性](#22-类属性)
  - [2.3. 类方法](#23-类方法)
    - [2.3.1. 类型转换](#231-类型转换)
    - [2.3.2. view 变形](#232-view-变形)
    - [2.3.3. transpose](#233-transpose)
  - [2.4. 创建操作 Creation Ops](#24-创建操作-creation-ops)
    - [2.4.1. 统一值 tensor](#241-统一值-tensor)
    - [2.4.2. 随机值 random](#242-随机值-random)
    - [2.4.3. \_like 类方法](#243-_like-类方法)
    - [2.4.4. torch.from\_numpy](#244-torchfrom_numpy)
    - [2.4.5. tensor复制](#245-tensor复制)
    - [2.4.6. .new\_ 方法](#246-new_-方法)
- [3. torch](#3-torch)
  - [3.1. 序列化 Serialization](#31-序列化-serialization)
    - [3.1.1. torch.save](#311-torchsave)
    - [3.1.2. torch.load](#312-torchload)
  - [3.2. Creation Ops](#32-creation-ops)
    - [3.2.1. torch.tensor](#321-torchtensor)
  - [3.3. Math operations](#33-math-operations)
    - [3.3.1. Pointwise Ops 元素为单位的操作](#331-pointwise-ops-元素为单位的操作)
    - [3.3.2. Reduction Ops 元素之间的操作(降维)](#332-reduction-ops-元素之间的操作降维)
      - [3.3.2.1. 极值操作](#3321-极值操作)
    - [3.3.3. Comparison Ops](#333-comparison-ops)
      - [3.3.3.1. topk](#3331-topk)
  - [3.4. Indexing, Slicing, Joining, Mutating Ops 拼接与截取等](#34-indexing-slicing-joining-mutating-ops-拼接与截取等)
    - [3.4.1. torch.stack](#341-torchstack)
    - [3.4.2. torch.cat](#342-torchcat)
- [4. Tensor views](#4-tensor-views)
- [5. torch.nn](#5-torchnn)
  - [5.1. Containers 网络容器](#51-containers-网络容器)
    - [5.1.1. torch.nn.Module 类](#511-torchnnmodule-类)
      - [5.1.1.1. 基础方法及应用](#5111-基础方法及应用)
      - [5.1.1.2. 网络参数以及存取](#5112-网络参数以及存取)
      - [5.1.1.3. 残差结构](#5113-残差结构)
  - [5.2. Convolution Layers 卷积层](#52-convolution-layers-卷积层)
  - [5.3. Pooling layers 池化层](#53-pooling-layers-池化层)
  - [5.4. Padding Layers](#54-padding-layers)
  - [5.5. Non-linear Activations (weighted sum, nonlinearity) 非线性激活函数](#55-non-linear-activations-weighted-sum-nonlinearity-非线性激活函数)
  - [5.6. Normalization Layers 归一化层](#56-normalization-layers-归一化层)
  - [5.7. Linear Layers  线性层](#57-linear-layers--线性层)
    - [5.7.1. Identity](#571-identity)
    - [5.7.2. Linear](#572-linear)
  - [5.8. Loss Function 损失函数](#58-loss-function-损失函数)
- [6. torch.nn.functional](#6-torchnnfunctional)
- [7. torch.utils](#7-torchutils)
  - [7.1. torch.utils.data](#71-torchutilsdata)
    - [7.1.1. 数据集类型](#711-数据集类型)
    - [7.1.2. torch.utils.data.Dataset](#712-torchutilsdatadataset)
    - [7.1.3. torch.utils.data.DataLoader](#713-torchutilsdatadataloader)
- [8. torch.optim](#8-torchoptim)
  - [8.1. 预定义 Algorithm](#81-预定义-algorithm)
  - [8.2. 动态 Learn Rate](#82-动态-learn-rate)
    - [8.2.1. 有序调整](#821-有序调整)
  - [8.3. 定义自己的 optim](#83-定义自己的-optim)
    - [8.3.1. Optimizer 基类](#831-optimizer-基类)
    - [8.3.2. optimization step](#832-optimization-step)
    - [8.3.3. per-parameter options](#833-per-parameter-options)
- [9. TorchScript](#9-torchscript)
  - [9.1. Tracing Modules](#91-tracing-modules)
  - [9.2. Convert Modules](#92-convert-modules)
  - [9.3. Mixing Scripting and Tracing](#93-mixing-scripting-and-tracing)
  - [9.4. Saving and Loading models](#94-saving-and-loading-models)
  - [9.5. API torch.jit](#95-api-torchjit)
- [10. Pytorch C++ API](#10-pytorch-c-api)
  - [10.1. ATen](#101-aten)
- [11. 例程](#11-例程)
  - [11.1. MNIST LeNet 例程](#111-mnist-lenet-例程)
    - [11.1.1. Network structure](#1111-network-structure)
    - [11.1.2. dataset](#1112-dataset)
    - [11.1.3. iteration](#1113-iteration)
    - [11.1.4. evaluate](#1114-evaluate)
  - [11.2. MINST GAN 例程](#112-minst-gan-例程)
    - [11.2.1. dataset](#1121-dataset)
    - [11.2.2. 网络](#1122-网络)
    - [11.2.3. G 训练](#1123-g-训练)
    - [11.2.4. D 训练](#1124-d-训练)
    - [11.2.5. iteration](#1125-iteration)
  - [11.3. MINST USPS adversarial examples](#113-minst-usps-adversarial-examples)
    - [11.3.1. dataset](#1131-dataset)
    - [11.3.2. general train and evaluate](#1132-general-train-and-evaluate)

# 1. Pytorch

Pytorch 基础操作 

## 1.1. install 

官方有很方便使用的 command 自动生成交互  
https://pytorch.org/get-started/locally/


可通过 conda 或者 pip 方便安装

```sh

# 选择自己系统对应的 CUDA 版本进行安装
conda install pytorch torchvision torchaudio cpuonly -c pytorch
conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch

# pip 相对更复杂
pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch torchvision  # 这是当前默认的针对 CUDA 10.2
```

## 1.2. verification

```py
import torch
x = torch.rand(5, 3)
print(x)

torch.cuda.is_available()


```

## 1.3. API

至2020年11月  
torch 最新版是 1.7  
中文文档更新到1.4  

```py
# 开始使用
import torch
print(torch.__version__)

```


# 2. torch.Tensor 张量类

* `torch.Tensor` 是 pytorch 的张量的类名
* `torch.Tensor` is an alias for the default tensor type (torch.FloatTensor).
* `torch.` 里有许多便捷创建张量的函数
* Tensor 类里面也有许多转换格式的方法
* 几乎所有的类方法都有 torch.* 下的同名方法, 功能一样, 多一个参数是输入 tensor   

## 2.1. torch.Tensor 的格式

* `torch.Tensor` 其实是 `torch.FloatTensor` 的别称, 即默认都会创建该类型的张量  
* pytorch 的 tensor 总共支持10种数据格式, CPU tensor 名称如下  
  * 可以通过对应的类名进行创建, 但更一般的都是指定 `dtype` 来指定数据类型
* GPU的格式名是在 torch 后面加上 .cuda 即可  
  * 但一般都是指定 `device` 来指定设备


* 16,32,64 bit 的浮点数
  * torch.HalfTensor 和 torch.BFloat16Tensor
  * torch.FloatTensor
  * torch.DoubleTensor
* 32,64,128bit 的复数
  * 无 CPU tensor
* 8,16,32,64 bit 的整数 (区分有无符号)
  * torch.ByteTensor 和 (有符号)torch.CharTensor
  * torch.ShortTensor
  * torch.IntTensor
  * torch.LongTensor
* 布尔类型
  * torch.BoolTensor

## 2.2. 类属性

* 所有的张量类都有三个基础属性
  * `torch.dtype`
  * `torch.device`
  * `torch.layout`

1. dtype:  
是一个object 用来指定张量的数据类型, 和张量本身的类型有所区分  
* 省略了`torch.` 的表格, 省略了 `*.` 代表的 CPU或者GPU 指定
* 注意用的时候要指定 `torch.int8` 这样子
  
| 数据类型  | dtype              | 张量类名(传统创建) |
| --------- | ------------------ | ------------------ |
| 16浮点1   | float16/half       | HalfTensor         |
| 16浮点2   | bfloat16           | BFloat16Tensor     |
| 32浮点    | float32/float      | FloatTensor        |
| 64浮点    | float64/double     | DoubleTensor       |
| 64复数    | complex64/cfloat   | 无                 |
| 128复数   | complex128/cdouble | 无                 |
| 无符8整数 | uint8              | ByteTensor         |
| 8整数     | int8               | CharTensor         |
| 16整数    | int16/short        | ShortTensor        |
| 32整数    | int32/int          | IntTensor          |
| 64整数    | int64/long         | LongTensor         |
| 布尔      | bool               | BoolTensor         |







## 2.3. 类方法
### 2.3.1. 类型转换

1. .item()   : Returns the value of this tensor as a standard Python number
   * 只在张量中只有一个元素时生效
   * 将该元素作为数字返回
2. .tolist() : Returns the tensor as a (nested) list.
   * 作为多层python原生列表返回,保留层次结构
3. .numpy()  : Returns self tensor as a NumPy ndarray
   * 共享内存, 非拷贝
  
### 2.3.2. view 变形

`view(*shape) → Tensor`  
* 返回一个更改了维度的 tensor
* view既是一个类方法, 也是一个 tensor 变形函数的类别
* 特点是共享内存的, 不进行拷贝


```py
a = torch.randn(1, 2, 3, 4)
# torch.Size([1, 2, 3, 4])

b = a.transpose(1, 2)
# torch.Size([1, 3, 2, 4])

c = a.view(1, 3, 2, 4)
# torch.Size([1, 3, 2, 4])
```

### 2.3.3. transpose

`torch.transpose(input, dim0, dim1) → Tensor`  
* transpose 可以解释为view的一种
* 返回一个 transposed 的 tensor
* The given dimensions dim0 and dim1 are swapped.
* 该方法是共享内存的, 不进行拷贝
  

注意该方法只能交换两个维度  
同 numpy 的 transpose 不同, numpy 的transpose 可以直接交换多个维度  


## 2.4. 创建操作 Creation Ops



### 2.4.1. 统一值 tensor


### 2.4.2. 随机值 random

函数参数 : `(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)`
* size 指定大小
* out  除了返回值以外的另一种获取方法
* dtype
* layout 指定 tensor 的 layout (暂时还不懂)
* device 默认时会指定为 `torch.set_default_tensor_type()`
* requires_grad : If autograd should record operations on the returned tensor.


基础函数:
* rand(*)                Uniform distribution on interval `[0,1)`
* randint(low=0,high)    high是必须参数, Uniform 
* randn(*)               standard normal distribution
* 

特殊函数:
* normal(mean, std)      随机标准分布, 均值方差手动指定
  * 这个函数原型 mean 和 std 至少有一项必须是 tensor, 用来指明生成的 tensor 的 size
  * normal(mean, std, size) 该原型用第三个参数来指定tensor 大小
* randperm(n)            随机 0~n-1 的排列, 默认type是 int64




种子操作:
* seed   :Sets the seed for generating random numbers to a non-deterministic random number.
* manual_seed   :Sets the seed for generating random numbers.
* initial_seed   :Returns the initial seed for generating random numbers as a Python long.
* get_rng_state   :Returns the random number generator state as a torch.ByteTensor.
* set_rng_state   :Sets the random number generator state.


### 2.4.3. _like 类方法

需要获取一个不确定维度的 tensor, 即通过另一个 tensor 指定大小
* rand_like
* randint_like
* randn_like


### 2.4.4. torch.from_numpy

`torch.from_numpy` 接受一个 ndarray 并转换成 tensor 没有任何参数  
```py
torch.from_numpy(ndarray) → Tensor
```

### 2.4.5. tensor复制


* `torch.clone(input, *, memory_format=torch.preserve_format) → Tensor`
  * 返回一个完全相同的tensor,新的tensor开辟新的内存，但是仍然留在计算图中。
* `torch.Tensor.detach()`
  * detach() 属于 view 函数
  * 返回一个完全相同的tensor,新的tensor开辟与旧的tensor共享内存, 但会脱离计算图，不会牵扯梯度计算

一般彻底的复制并脱离可以使用  `tensor.clone().detach()`  这也是官方推荐的方法  


### 2.4.6. .new_ 方法

To create a tensor with similar type but different size as another tensor, use tensor.new_* creation ops.  

1. new_tensor
   * new_tensor可以将源张量中的数据复制到目标张量（数据不共享）
   * 提供了更细致的device、dtype和requires_grad属性控制
   * 默认参数下的操作等同于.clone().detach(), 但官方推荐后者
2. new_full
3. new_empty
4. new_ones
5. new_zeros




# 3. torch

* 作为最基础的包, torch 包括了tensors的数据格式以及对应的数学操作. 
* 还提供了许多工具来高效的序列化 tensors 以及其他任意数据格式   
* 该包有 CUDA 对应  

## 3.1. 序列化 Serialization

和存储相关, 将各种模型, 张量, 字典等数据类型序列化后存储到文件, 或者从文件中读取  

pytorch 的load() 是基于 pickle 模组的, 不要轻易unpick不信任的序列数据  

一共就两个函数
* torch.save()
* torch.load()

官方推荐的网络模型存取方式:
* `torch.save(model.state_dict(), PATH)`
* `model.load_state_dict(torch.load(PATH))`
完整模型的存取
* `torch.save(model, PATH)`
* `model = torch.load(PATH)`

### 3.1.1. torch.save

```py
torch.save(
  obj, 
  f: Union[str, os.PathLike, BinaryIO], 
  pickle_module=<module 'pickle' from '/opt/conda/lib/python3.6/pickle.py'>,
  pickle_protocol=2, 
  _use_new_zipfile_serialization=True # 代表使用 pytorch 1.6 后的新的压缩格式
  ) → None
```

参数意思:
* `obj`   : 要保存的对象
* f       : a file-like object
* pickle_module   :  
* pickle_protocol :

### 3.1.2. torch.load

```py
torch.load(
  f, 
  map_location=None, 
  pickle_module=<module 'pickle' from '/opt/conda/lib/python3.6/pickle.py'>, 
  **pickle_load_args
  )

""" 
f   : a file-like object
map_location    : a function, torch.device, string or a dict specifying how to remap storage locations
pickle_module   :  
pickle_load_args: (Python 3 only) optional keyword arguments passed over to pickle_module.load() and pickle_module.Unpickler()
"""

torch.load('tensors.pt')

# Load all tensors onto the CPU
torch.load('tensors.pt', map_location=torch.device('cpu'))

# Map tensors from GPU 1 to GPU 0
>>> torch.load('tensors.pt', map_location={'cuda:1':'cuda:0'})

```


## 3.2. Creation Ops

基本上所有创建 tensor 的函数都位于 `torch.` 下  

创建tensor 有许多通用的参数
* `dtype  =`
* `device =`

### 3.2.1. torch.tensor

* 从 python list 或者 numpy array 来创建张量
* `torch.tensor()` 函数总是会进行数据拷贝
  * 防止拷贝可以使用 `torch.as_tensor()`
```py
torch.tensor(data, *, dtype=None, device=None, requires_grad=False, pin_memory=False) → Tensor
``` 

## 3.3. Math operations

提供对 tensor 的各种数学操作

### 3.3.1. Pointwise Ops 元素为单位的操作

### 3.3.2. Reduction Ops 元素之间的操作(降维)

* 所有函数都默认对张量的全部元素进行运算
* 提供了针对选定维度的修改运算, 参数为
  * dim : 除非特殊说明都是int, 在该维度上进行运算, 对于其他维度的一个固定值, dim 维度的所有值视作一行, 有些函数支持 dim 为 tuple, 意味同时对多个维度进行降维
  * keepdim: bool, 如果为 True, dim 维度将保留, 并且大小为 1, 否则 dim 将会被压缩而省略掉 


简单函数:
* 布尔类:
  * torch.all() 是否全为 True
  * torch.any() 是否存在 True
* 基础运算 : dim 支持 tuple
  * torch.mean() 均值
  * torch.sum() 求和

#### 3.3.2.1. 极值操作

* max/min : 返回最大值, 默认对全元素进行, 可以输入单个整数 dim 来对某一维度进行运算, 此时会返回两个值, 第二个返回值是索引位置
* argmax/argmin : 返回最大值的索引, 默认对全元素进行操作, 可以输入单个整数 dim 来对某一维度进行运算, 等同于 max/min 输入 dim 的第二个返回值
* amax/amin : 返回最大值, 专门用来对指定维度进行运算, dim 是必须参数且可以是 int or tuple, 即可以对多个维度进行运算, 不会返回索引



### 3.3.3. Comparison Ops

专门用来比较的函数 


#### 3.3.3.1. topk

获取指定维度的 k 个最大值, 同时还能获得对应的 索引, 用在分类任务上  
`torch.topk(input, k, dim=None, largest=True, sorted=True, *, out=None) -> (Tensor, LongTensor)`  
* dim, int 只能指定单独维度, 默认会对最后一个维度进行操作
* largest: bool, true 则选择最大的top, 否则选择最小的 top
* sorted: bool, 是否按对应的顺序返回这k个值


## 3.4. Indexing, Slicing, Joining, Mutating Ops 拼接与截取等

### 3.4.1. torch.stack
`torch.stack(tensors, dim=0, *, out=None) → Tensor`  

* 将多个 tensor 叠加到一起, 并产生一个新的 dimension
* Concatenates a sequence of tensors along a new dimension.
* 因此所有 tensor 必须相同大小
* dim 指定新的 dimension 的位置

### 3.4.2. torch.cat

`torch.cat(tensors, dim=0, *, out=None) → Tensor`  

* 将多个 tensor 链接, 沿着最外层 index 或者指定 dim 链接
* All tensors must either have the same shape (except in the concatenating dimension) or be empty.
* dim (int, optional) – the dimension over which the tensors are concatenated


# 4. Tensor views

View 是一个数据的映射共享, 可以避免多余的数据拷贝  



# 5. torch.nn

These are the basic building block for graphs  
用于定义神经网络相关的内容  

分类
* Containers            承载网络的各种模块
* Convolution Layers    卷积层
* Pooling Layers        池化层
* Padding Layers        填充层
* Non-linear Act        非线性激活函数
* Normalization         标准化层
* Recurrent Layers      循环网络层
* Transformer Layer     编码器层
* Linear Layers         线性层, 全连接层
* Dropout Layers        Dropout层
* Sparse Layers         稀疏层
* Loss Functions        损失函数
* Vision Layers         视觉层
* Shuffle Layers        洗牌层
* DataParallel Layers   数据并行层
* 其他
  * Utilities
  * Quantized Functions
  * Lazy Modules Initalization

## 5.1. Containers 网络容器
 
Module            Base class for all neural network modules.  
Sequential        A sequential container.  
ModuleList        Holds submodules in a list.  
ModuleDict        Holds submodules in a dictionary.  
ParameterList     Holds parameters in a list.  
ParameterDict     Holds parameters in a dictionary.  

### 5.1.1. torch.nn.Module 类

* `class torch.nn.Module  `
  * 所有神经网络模块的基类 Base class for all neural network modules.
  * 您的模型也应该继承此类
  * 模块也可以包含其他模块，从而可以将它们嵌套在树形结构中。 您可以将子模块分配为常规属性

* A `Module` is the basic unit of composition in PyTorch.
  1. A constructor, which prepares the module for invocation.
  2. A set of `Parameters` and `sub-Modules`. These are initialized by the constructor and can be used by the module during invocation.
  3. A `forward` function. This is the code that is run when the module is invoked.

流程总结:
1. 创建一个类 , 继承于 torch.nn.Module
2. 定义构造函数, 不需要做什么, 只需要: just calls the constructor for super.
3. 定义 `forward` 函数, 

#### 5.1.1.1. 基础方法及应用

* `cpu() `                : Moves all model parameters and buffers to the CPU.
* `cuda(device: Union[int, torch.device, None] = None)`   :Moves all model parameters and buffers to the GPU
* `forward(*input: Any)`  : Defines the computation performed at every call. Should be overridden by all subclasses.

模式
* training mode.
* evaluation mode.  `net.eval()`


```py
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))

# Initialize the network
net = Model().cuda()

# 进入 evaluation mode  等同于 
net.train(False)
net.eval()

# 进入 training mode
net.train()

```

#### 5.1.1.2. 网络参数以及存取

state_dict 是网络中所有层中的所有参数   
官方推荐的模型存取方式:
* `torch.save(model.state_dict(), PATH)`
* `model.load_state_dict(torch.load(PATH))`

```py

# 返回一个字典保存了当前模型的全部参数
state_dict(destination=None, prefix='', keep_vars=False)

# 显示键名
>>> module.state_dict().keys()
['bias', 'weight']



# 读取参数
load_state_dict(state_dict: Dict[str, torch.Tensor], strict: bool = True)
# state_dict  : state_dict 的 obj, 字典类型
# strict      : 所有结构以及参数tensor的大小必须一致

""" 
返回值:
NamedTuple with missing_keys and unexpected_keys fields
"""

```
#### 5.1.1.3. 残差结构


## 5.2. Convolution Layers 卷积层 

用于在神经网络中定义卷积层  

基础卷积 : Applies a ?D convolution over an input signal composed of several input planes.
* nn.Conv1d
* nn.Conv2d
* nn.Conv3d

* nn.ConvTranspose1d
  * Applies a 1D transposed convolution operator over an input image composed of several input planes.
* nn.ConvTranspose2d
  * Applies a 2D transposed convolution operator over an input image composed of several input planes.
* nn.ConvTranspose3d
  * Applies a 3D transposed convolution operator over an input image composed of several input planes.

* nn.Unfold
  * Extracts sliding local blocks from a batched input tensor.

* nn.Fold
  * Combines an array of sliding local blocks into a large containing tensor.


```py
class LeNet(nn.Module):
  def __init__(self, input_dim=1, num_class=10):
    super(LeNet, self).__init__()

    # Convolutional layers
    # 参数分别是  输入channel 输出channel 和三个卷积参数
    self.conv1 = nn.Conv2d(input_dim, 20,  kernel_size=5, stride=1, padding=0) 
    self.bn1 = nn.BatchNorm2d(20)
    self.conv2 = nn.Conv2d(20,    50,  kernel_size=5, stride=1, padding=0) 
    self.bn2 = nn.BatchNorm2d(50)

# 定义好网络后, 可以直接通过成员网络层输出当前网络的参数
net = Model().cuda()

print(net.conv1.weight.size()) 
print(net.conv1.weight)
print(net.conv1.bias)

```
## 5.3. Pooling layers 池化层

* 最大化池
  * nn.MaxPool1d
  * nn.MaxPool2d
  * nn.MaxPool3d
  * nn.MaxUnpool1d
  * nn.MaxUnpool2d
  * nn.MaxUnpool3d
* 平均池
  * nn.AvgPool1d
  * nn.AvgPool2d
  * nn.AvgPool3d
* 不认识的
  * nn.FractionalMaxPool2d
  * nn.LPPool1d
  * nn.LPPool2d
  * nn.AdaptiveMaxPool1d
  * nn.AdaptiveMaxPool2d
  * nn.AdaptiveMaxPool3d
  * nn.AdaptiveAvgPool1d
  * nn.AdaptiveAvgPool2d
  * nn.AdaptiveAvgPool3d

## 5.4. Padding Layers

填充层, 一般对于卷积层的填充直接传入对应的参数即可, 该分类的填充层都是便于使用的常用   

具体的填充层有3+1种
* 如果需要更多维度的 pad, 需要使用 `torch.nn.functional.pad().`

三类支持 1~3D的填充
* `nn.ReflectionPad(1,2,3)d`
  * 镜像填充, 从边缘往里的值复制的填充到外边缘
  * 参数 `填充维度的 int or 2,4,6-tuple`
* `nn.ReplicationPad(1,2,3)d`
  * 重复填充, 根据边缘的值赋值的填充到外边缘
  * 参数 `填充维度的 int or 2,4,6-tuple`
* `nn.ConstantPad(1,2,3)d`
  * 固定值填充
  * 传入的参数 `填充维度的 int or 2,4,6-tuple, 填充值`

一类标准 Zeropad
* `class torch.nn.ZeroPad2d(padding)`
  *  `填充维度的 int or 4-tuple, 填充值`

## 5.5. Non-linear Activations (weighted sum, nonlinearity) 非线性激活函数

* 加权和，非线性
  * nn.ELU
  * nn.Hardshrink
  * nn.Hardsigmoid
  * nn.Hardtanh
  * nn.Hardswish
  * nn.LeakyReLU
  * nn.LogSigmoid
  * nn.MultiheadAttention
  * nn.PReLU
  * nn.ReLU
  * nn.ReLU6
  * nn.RReLU
  * nn.SELU
  * nn.CELU
  * nn.GELU
  * nn.Sigmoid
  * nn.SiLU
  * nn.Softplus
  * nn.Softshrink
  * nn.Softsign
  * nn.Tanh
  * nn.Tanhshrink
  * nn.Threshold
* 其他激活函数
  * nn.Softmin
  * nn.Softmax
  * nn.Softmax2d
  * nn.LogSoftmax
  * nn.AdaptiveLogSoftmaxWithLoss


```py

# 在网络中定义激活函数
# Activation func.
    self.relu = nn.ReLU()
```


## 5.6. Normalization Layers 归一化层

用于定义网络的归一化层  

* nn.BatchNorm1d
* nn.BatchNorm2d
* nn.BatchNorm3d
* nn.GroupNorm
* nn.SyncBatchNorm
* nn.InstanceNorm1d
* nn.InstanceNorm2d
* nn.InstanceNorm3d
* nn.LayerNorm
* nn.LocalResponseNorm


```py
class LeNet(nn.Module):
  def __init__(self, input_dim=1, num_class=10):
    super(LeNet, self).__init__()

    # Convolutional layers
    self.conv1 = nn.Conv2d(input_dim, 20,  kernel_size=5, stride=1, padding=0) 

    # 输入参数为通道数
    self.bn1 = nn.BatchNorm2d(20)
    self.conv2 = nn.Conv2d(20,    50,  kernel_size=5, stride=1, padding=0) 

    # 输入参数为通道数
    self.bn2 = nn.BatchNorm2d(50)

```



## 5.7. Linear Layers  线性层

用于构筑网络的全连接层, 例如最早期的 MLP  

* nn.Identity
* nn.Linear
* nn.Bilinear
* nn.LazyLinear

### 5.7.1. Identity

用在网络构造定义中的占位符, 类似于 pass, 可以吞入任何参数
```py
class
torch.nn.Identity(*args, **kwargs)

```

### 5.7.2. Linear

```py
class
torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)

# in_features   :每个输入样本的大小
# out_features  :每个输出样本的大小
# bias          :如果设置为False，则该图层将不会学习加法偏差。 默认值：True

# 定义一个 MLP
    # Fully connected layers
    self.fc1 = nn.Linear(800, 500)
    #self.bn3 = nn.BatchNorm1d(500)
    self.fc2 = nn.Linear(500, num_class)

```

## 5.8. Loss Function 损失函数

pytorch 的损失函数直接定义在了 torch.nn 中

    L1Loss
    MSELoss
    CrossEntropyLoss
    CTCLoss
    NLLLoss
    PoissonNLLLoss
    KLDivLoss
    BCELoss
    BCEWithLogitsLoss
    MarginRankingLoss
    HingeEmbeddingLoss
    MultiLabelMarginLoss
    SmoothL1Loss
    SoftMarginLoss
    MultiLabelSoftMarginLoss
    CosineEmbeddingLoss
    MultiMarginLoss
    TripletMarginLoss

```py

# 使用时直接定义对象即可
loss_func = nn.CrossEntropyLoss()
```


# 6. torch.nn.functional

不是直接定义层, 而是把各个网络层的运算抽出来的包  

使用较为基础的运算进行网络定制化的时候需要用到



# 7. torch.utils
## 7.1. torch.utils.data
Pytorch 自定义数据库中最重要的部分  
提供了对 `dataset` 的所种操作模式  

### 7.1.1. 数据集类型

Dataset 可以分为两种类型的数据集, 在定义的时候分别继承不同的抽象类

1. map-style datasets 即 继承`Dataset` 类
  * 必须实现 `__getitem__()` and `__len__()` protocols
  * represents a map from (possibly non-integral) indices/keys to data samples.
  *  when accessed with `dataset[idx]`, could read the idx-th image and its corresponding label from a folder on the disk.

2. iterable-style datasets 即继承 `IterableDataset` 类
  * 必须实现 `__iter__()` protocol 
  * particularly suitable for cases where random reads are expensive or even improbable, and where the batch size depends on the fetched data.
  * when called `iter(dataset)`, could return a stream of data reading from a database, a remote server, or even logs generated in real time.


`torch.utils.data.Dataset`  和  `torch.utils.data.IterableDataset`  

### 7.1.2. torch.utils.data.Dataset

* Dataset 类是一个抽象类, 用于 map-key 的数据集
* Dataset 类是 DataLoader 的最重要的构造参数  

定义关键:
1. All datasets that represent a map from keys to data samples should subclass it.
2. 所有实现类需要重写 `__getitem__()` 用于从一个 index 来获取数据和label
3. 可选的实现 `__len__()`  用于返回该数据库的大小, 会被用在 默认的 `Sampler` 上


```py
from torch.utils.data import Dataset
# 继承
class trainset(Dataset):
   def __init__(self):
    #  在这里任意定义自己数据库的内容
   
   #  也可以更改构造函数
   def __init__(self,loader=dafult_loader):
     # 路径
     self.images = file_train
     self.target = number_train
     self.loader = loader
   #  定义 __getitem__ 传入 index 得到数据和label
   #  实现了该方法即可使用 dataset[i] 下标方法获取到 i 号样本
   def __getitem__(self, index):
      # 获得路径
      fn = self.images[index]
      # 读图片
      img = self.loader(fn)
      # 得到labels
      target = self.target[index]
      return img,target
   def __len__(self):
      # 返回数据个数
      return len(self.images)
```


### 7.1.3. torch.utils.data.DataLoader

Pytorch的核心数据读取器`torch.utils.data.DataLoader`   
是一个可迭代的数据装载器  包括了功能:  
  * map-style and iterable-style datasets,
  * customizing data loading order,
  * automatic batching,
  * single- and multi-process data loading,
  * automatic memory pinning.

使用的时候预先定义好一个 dataset 然后用 DataLoader包起来  

```py
# 第一个参数是绑定的数据集  是最重要的参数  
DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None, *, prefetch_factor=2,
           persistent_workers=False)
# Test data 不需要 shuffle 
# batch_size 指定了一次训练多少个数据
# num_workers 为正数时代表指定了多线程数据装载
trainloader = utils.data.DataLoader(mnist_train, batch_size=50, shuffle=True,  num_workers=2)
testloader  = utils.data.DataLoader(mnist_test, batch_size=100, shuffle=False, num_workers=2)

train_data  = trainset()
trainloader = DataLoader(train_data, batch_size=4,shuffle=True)
# -----------------------------------------------------------
# 通过装载器获得一个 可迭代数据库  使用 iter 
iter_data = iter(trainloader)
#iter_data = iter(testloader)

# 对可迭代数据库使用 next 得到数据和label
images, labels = next(iter_data)
print(images.size())
# torch.Size([100, 1, 28, 28])   100 个数据 每个数据 1 channel , 高和宽都是28

# ------------------------------------------
# 对于非迭代型数据库 即 map-key类型
# 直接使用 for 循环即可
for images,labels in trainLoader:
    print(images.size())
    # torch.Size([5, 3, 64, 64])
```


# 8. torch.optim

是一个实现各种优化算法的包。已经支持最常用的方法，并且界面足够通用，因此将来可以轻松集成更复杂的方法。  

要使用，必须构造一个优化器对象，该对象将`保持当前状态`并将`根据计算的梯度更新参数`  


```py
import torch.optim as optim

# 如果使用 .cuda() 将模型移动到GPU, 则应该在构建优化器之前操作  因为模型参数不同
# 构建一个优化器
# 参数分别是  可迭代的网络模型的参数,  学习速率, 动量, 一般需要保证网络模型的参数在内存中的位置不变  
optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)
optimizer = optim.Adam([var1, var2], lr = 0.0001)

```

## 8.1. 预定义 Algorithm

优化方法, (参数各不相同, 一般都有 lr )
* Adadelta
* Adagrad
* Adam
* AdamW
* SparseAdam
* Adamax
* ASGD
* LBFGS
* RMSprop
* Rprop
* SGD

## 8.2. 动态 Learn Rate

`torch.optim.lr_scheduler` 提供了一些接口用来根据 epoch 或者其他计算来调整学习速率  
`torch.optim.lr_scheduler.ReduceLROnPlateau` 则可以根据一些 validation measurements 来调整学习速率  

pytorch提供的学习率调整器可以分成三大类:
* 有序调整   : 等间隔(Step), 按需调整(MultiStep), 指数衰减 (Exponential), 余弦退火
* 自适应调整 : ReduceLROnPlateau
* 自定义调整 : LambdaLR


调整器的使用方法:  
```py
# 定义优化器
optimizer = SGD(model, 0.1)
# 给优化器绑定动态学习速率
scheduler = ExponentialLR(optimizer, gamma=0.9)

# 使用方法: 一般学习速度的调整应该放在 optimizer 更新之后, 在 epoch 的循环里调整
for epoch in range(100):
    for input, target in dataset:
      forward...
      loss=...
      loss.backward()
      optimizer.step()
  scheduler.step()
```

scheduler 的通用参数:
* gamma : float, 乘法参数, 当前学习速率直接乘以该值 
* last_epoch : 

scheduler 的通用成员方法:
* print_lr(is_verbose, group, lr, epoch=None)  : 打印当前的学习速率
* get_last_lr() : 根据输入的参数计算最终的学习率


scheduler 的种类:
* torch.optim.lr_scheduler.StepLR : 最基础的种类, 每 `step_size` 个 epochs 时候降低一次学习速率
* torch.optim.lr_scheduler.MultiStepLR : 同 Step, 只不过 step 变为数组
* orch.optim.lr_scheduler.ExponentialLR : 学习率指数下降


### 8.2.1. 有序调整


* LambdaLr : 使用自定义函数来生成学习率
  * 注意这里 Lambda 函数是一个单参数的函数, 传入当前 epoch 数
  * 返回一个乘法因子, 用初始 lr 乘以该因子即更新后的学习率

```py
# 1. StepLR
# Assuming optimizer uses lr = 0.05 for all groups
# lr = 0.05     if epoch < 30
# lr = 0.005    if 30 <= epoch < 60
# lr = 0.0005   if 60 <= epoch < 90
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)


# 定义一个线性学习率降低
endepoch=10
liner_func=lambda epoch: max(0,1-(epoch/endepoch))
optim.lr_scheduler.LambdaLR(optimizer,liner_func,**kwargs)
```


## 8.3. 定义自己的 optim

### 8.3.1. Optimizer 基类

`class torch.optim.Optimizer(params, defaults)` 是所有优化器的基类, 定义了优化器的必须操作  
* 参数
  * params (iterable) – an iterable of torch.Tensor s or dict s. 指定了要优化的张量
  * defaults – (dict): a dict containing default values of optimization options (used when a parameter group doesn’t specify them).
* 方法
  * add_param_group(param_group`(dict)`) 添加新的要进行优化的 param group 
    一般用来对 pre-trained network 进行 fine tuning 时进行优化
  * state_dict()
    返回当前优化器的状态 as a `dict` 里边不仅包含优化器的状态还有 param group
  * load_state_dict(state_dict`(dict)`)
    和上一个方法匹配, 装载一个优化器状态
  * step(closure) 不多说
  * zero_grad(set_to_none: bool = False)
    初始化所有梯度为0 
    `set_to_none` 设置初始化梯度不是0而是 `None` 这会带来一些性能优化,但同时会有一些其他后果 *懒得看了

### 8.3.2. optimization step

重点: 所有 optimizers 必须实现 step 方法, 用来更新要优化的参数  

1. optimizer.step()
   * 大部分优化器都实现了的简单的版本, 在 `backward()` 方法之后调用
```py
for input, target in dataset:
    # 初始化
    optimizer.zero_grad()
    # 前向传播
    output = model(input)
    # 计算误差
    loss = loss_fn(output, target)
    # 误差 backward
    loss.backward()
    # 优化器调用 step
    optimizer.step()
```

2. optimizer.step(closure)
   一部分优化器, 例如 ` Conjugate Gradient and LBFGS` 需要多次前向传播, 所以需要传入一个`closure`方法 来允许自己定义计算的模型.
   `closure` 需要清空梯度, 然后计算 loss, 最后返回

```py
for input, target in dataset:
    # 用closure 代替其他操作, 以函数的形式定义
    def closure():
        # 初始化
        optimizer.zero_grad()
        # 前向传播
        output = model(input)
        # 计算误差
        loss = loss_fn(output, target)
        # 误差 backward
        loss.backward()
        # 返回误差
        return loss
    # 将该函数传入 step 相当于在一条代码中整合了一整次更新参数的流程
    optimizer.step(closure)

```

### 8.3.3. per-parameter options

To do this, instead of passing an iterable of `Variable` s, pass in an iterable of `dict` s.    
* dict 中指定了不同的 parameter group, 并且需要使用 `params` 关键字
* 可以在 dict 中对不同 group 的参数分别指定 options ,也可以在 dict 外指定作为其他 group 的默认的 options.

```py
# classifier 的 lr 是 1e-3
# 其他默认的 lr 是 1e-2
optim.SGD([
                {'params': model.base.parameters()},
                {'params': model.classifier.parameters(), 'lr': 1e-3}
            ], lr=1e-2, momentum=0.9)
```

# 9. TorchScript


对于一个从Pytorch创建的一个可优化和串行的模型, 使其可以运行在其他非Python的平台上  
  * From a pure Python program to a TorchScript program that can be run independently from Python.
  * An intermediate representation of a PyTorch model (subclass of nn.Module) that can then be run in a high-performance environment such as C++.

TorchScript provides tools to capture the definition of your model, even in light of the flexible and dynamic nature of PyTorch.  

1. TorchScript 有自己的解释器, 这个解释器类似于受限制的 Python 解释器, 该解释器不获取全局解释器锁，因此可以在同一实例上同时处理许多请求。
2. TorchScript 可以是我们把整个模型保存到磁盘上, 并在另一个运行环境中载入, 例如非Python的运行环境
3. TorchScript 可以进行编译器优化并以此获得更高的执行效率
4. TorchScript 可以允许与许多后端设备运行接口, 这些运行环境往往需要比单独的操作器更广泛的程序视野.


## 9.1. Tracing Modules

Trace:
1. invoked the Module
2. recorded the operations that occured when the Module was run
3. created an instance of torch.jit.ScriptModule (of which TracedModule is an instance)
   根据输入模型的运算流 创建出对应的 TorchScript 模型对象


```py

class MyCell(torch.nn.Module):
    def __init__(self):
        super(MyCell, self).__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, h):
        new_h = torch.tanh(self.linear(x) + h)
        return new_h, new_h

# 定义网络对象
my_cell = MyCell()
x, h = torch.rand(3, 4), torch.rand(3, 4)

# 创建一个 traced 对象
# 参数是 网络对象 输入值
traced_cell = torch.jit.trace(my_cell, (x, h))

# 两个模型的运行结果没有区别
print(my_cell(x, h))
print(traced_cell(x, h))


# 打印 traced 对象
print(traced_cell)
""" 
MyCell(
  original_name=MyCell
  (linear): Linear(original_name=Linear)
)
"""

# referred to in Deep learning as a graph
print(traced_cell.graph)
# graph的可读性很差
"""
graph(%self.1 : __torch__.MyCell,
      %input : Float(3:4, 4:1, requires_grad=0, device=cpu),
      %h : Float(3:4, 4:1, requires_grad=0, device=cpu)):
  %19 : __torch__.torch.nn.modules.linear.Linear = prim::GetAttr[name="linear"](%self.1)
  %21 : Tensor = prim::CallMethod[name="forward"](%19, %input)
  %12 : int = prim::Constant[value=1]() # /var/lib/jenkins/workspace/beginner_source/Intro_to_TorchScript_tutorial.py:188:0
  %13 : Float(3:4, 4:1, requires_grad=1, device=cpu) = aten::add(%21, %h, %12) # /var/lib/jenkins/workspace/beginner_source/Intro_to_TorchScript_tutorial.py:188:0
  %14 : Float(3:4, 4:1, requires_grad=1, device=cpu) = aten::tanh(%13) # /var/lib/jenkins/workspace/beginner_source/Intro_to_TorchScript_tutorial.py:188:0
  %15 : (Float(3:4, 4:1, requires_grad=1, device=cpu), Float(3:4, 4:1, requires_grad=1, device=cpu)) = prim::TupleConstruct(%14, %14)
  return (%15) 
"""

#  Python-syntax interpretation of the code
print(traced_cell.code)
""" 
def forward(self,
    input: Tensor,
    h: Tensor) -> Tuple[Tensor, Tensor]:
  _0 = torch.add((self.linear).forward(input, ), h, alpha=1)
  _1 = torch.tanh(_0)
  return (_1, _1)
"""

```

## 9.2. Convert Modules

* 对于一个带有控制流的子模型, 直接使用 Trace 不能正确的捕捉整个程序流程  
* 使用 `script compiler` 即可, 可以直接分析Python 源代码来导出 TorchScript

```py
# 带控制流的子模型
class MyDecisionGate(torch.nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            return x
        else:
            return -x

# 主模型
class MyCell(torch.nn.Module):
    # 用参数的形式引入子模型
    def __init__(self, dg):
        super(MyCell, self).__init__()
        self.dg = dg
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, h):
        # 先线性传播, 在导入子模型传播, 最后 tanh
        new_h = torch.tanh(self.dg(self.linear(x)) + h)
        return new_h, new_h

# 创建对象
my_cell = MyCell(MyDecisionGate())

# 创建追踪对象
traced_cell = torch.jit.trace(my_cell, (x, h))

# 反推 code
print(traced_cell.code)
""" 
def forward(self,
    input: Tensor,
    h: Tensor) -> Tuple[Tensor, Tensor]:
  _0 = self.dg
  _1 = (self.linear).forward(input, )
  # 在这一步可以看到 只读取出来了 dg.forward(_1), 没有读取出 dg 内部的运行
  _2 = (_0).forward(_1, )
  _3 = torch.tanh(torch.add(_1, h, alpha=1))
  return (_3, _3) """


# 从子模型的类直接推出 TorchScript 对象
scripted_gate = torch.jit.script(MyDecisionGate())
# 用 scripted 的对象定义主模型
my_cell = MyCell(scripted_gate)
# script 主模型对象
traced_cell = torch.jit.script(my_cell)
# 打印 TorchScript
print(traced_cell.code)
""" 
def forward(self,
    x: Tensor,
    h: Tensor) -> Tuple[Tensor, Tensor]:
  # 从这里可以完整的推出 dg 的运算过程
  _0 = (self.dg).forward((self.linear).forward(x, ), )
  new_h = torch.tanh(torch.add(_0, h, alpha=1))
  return (new_h, new_h) """

# 运行结果是相同的
traced_cell(x, h)
my_cell(x,h)

```

## 9.3. Mixing Scripting and Tracing

混合 Script 和 Trace

* trace, module has many architectural decisions that are made based on constant Python values that we would like to not appear in TorchScript
* In this case, scripting can be composed with tracing: torch.jit.script will inline the code for a traced module, and tracing will inline the code for a scripted module.

```py

class MyRNNLoop(torch.nn.Module):
    def __init__(self):
        super(MyRNNLoop, self).__init__()
        # 内部使用了 traced 的上文的 Mycell
        self.cell = torch.jit.trace(MyCell(scripted_gate), (x, h))

    def forward(self, xs):
        h, y = torch.zeros(3, 4), torch.zeros(3, 4)
        for i in range(xs.size(0)):
            y, h = self.cell(xs[i], h)
        return y, h

class WrapRNN(torch.nn.Module):
    def __init__(self):
        super(WrapRNN, self).__init__()
        # 内部使用了 MyRNNLoop
        self.loop = torch.jit.script(MyRNNLoop())

    def forward(self, xs):
        y, h = self.loop(xs)
        return torch.relu(y)

# 创建 Script
rnn_loop = torch.jit.script(MyRNNLoop())

# 创建 trace
traced = torch.jit.trace(WrapRNN(), (torch.rand(10, 3, 4)))


print(rnn_loop.code)
""" 
def forward(self,
    xs: Tensor) -> Tuple[Tensor, Tensor]:
  h = torch.zeros([3, 4], dtype=None, layout=None, device=None, pin_memory=None)
  y = torch.zeros([3, 4], dtype=None, layout=None, device=None, pin_memory=None)
  y0 = y
  h0 = h
  for i in range(torch.size(xs, 0)):
    _0 = (self.cell).forward(torch.select(xs, 0, i), h0, )
    y1, h1, = _0
    y0, h0 = y1, h1
  return (y0, h0)
"""

print(traced.code)
""" 
def forward(self,
    argument_1: Tensor) -> Tensor:
  _0, h, = (self.loop).forward(argument_1, )
  return torch.relu(h)
"""

# 这样 script 和 trace 可以同时使用, 并且各自被调用
# 没太懂

```

## 9.4. Saving and Loading models

save and load TorchScript modules  
这种形式的存储 包括了代码,参数,性质还有Debug信息

```py

traced.save('wrapped_rnn.zip')

loaded = torch.jit.load('wrapped_rnn.zip')

print(loaded)
print(loaded.code)

```

## 9.5. API torch.jit

* script(obj[, optimize, _frames_up, _rcb])
* trace(func, example_inputs[, optimize, …])
* trace_module(mod, inputs[, optimize, …])
* fork(func, *args, **kwargs)
* wait(future)
* ScriptModule()
* ScriptFunction
* save(m, f[, _extra_files])
* load(f[, map_location, _extra_files])
* ignore([drop])
* unused(fn)

# 10. Pytorch C++ API

## 10.1. ATen



# 11. 例程


## 11.1. MNIST LeNet 例程
### 11.1.1. Network structure
```py

class LeNet(nn.Module):

    # 在最开始获取了 输入维度和 输出的类别
  def __init__(self, input_dim=1, num_class=10):
    super(LeNet, self).__init__()


    # Convolutional layers
    self.conv1 = nn.Conv2d(input_dim, 20,  kernel_size=5, stride=1, padding=0) 
    self.bn1 = nn.BatchNorm2d(20)
    self.conv2 = nn.Conv2d(20,    50,  kernel_size=5, stride=1, padding=0) 
    self.bn2 = nn.BatchNorm2d(50)
    
    # Fully connected layers

    # 第一个全连接层将800个参数降到500
    self.fc1 = nn.Linear(800, 500)
    #self.bn3 = nn.BatchNorm1d(500)
    # 第二个全连接层就直接得到各个类的预测度了
    self.fc2 = nn.Linear(500, num_class)
    
    # Activation func.
    self.relu = nn.ReLU()


  # 注意向前推演函数的定义方法, 按照网络结构依次调用各个层即可
  def forward(self, x):

    #  第一个卷积层然后激活并归一化
    #  28 x 28 x 1 -> 24 x 24 x 20
    x = self.relu(self.conv1(x))                                  
    x = self.bn1(x)
    
    # 最大化池化后分辨率减半
    x = F.max_pool2d(x, kernel_size=2, stride=2)  
    # 12 x 12 x 20
    

    # 第二个卷积层分辨率减4
    x = self.relu(self.conv2(x))                                   
    # -> 8 x 8 x 50
    x = self.bn2(x)
    
    # 最大化池后分辨率再减半
    x = F.max_pool2d(x, kernel_size=2, stride=2)  
    # -> 4 x 4 x 50

    # 获取x 高维数据的维度  
    # batch, channels, height, width
    b,c,h,w = x.size()


    # flatten the tensor x -> 800
    x = x.view(b, -1) 

    # fc-> ReLU
    x = self.relu(self.fc1(x))          
    
    # fc
    x = self.fc2(x)                           
    return x


# 定义完成后即可初始化网络

# Initialize the network
net = LeNet().cuda()

# 直接打印网络的信息
# To check the net's architecutre
print(net)
```


### 11.1.2. dataset


```py

# 数据集
from   torchvision import datasets as datasets

# pytorch 的图像转换方法
import torchvision.transforms as transforms


import torch.utils as utils
import matplotlib.pyplot as plt
import torch
import torchvision


# ------ First let's get the dataset (we use MNIST) ready ------
# We define a function (named as transform) to 
# -1) convert the data_type (np.array or Image) of an image to torch.FloatTensor;
# -2) standardize the Tensor for better classification accuracy 

# The "transform" will be used in "datasets.MNIST" to process the images.
# You can decide the batch size and whether shuffling the samples or not by setting
# "batch_size" and "shuffle" in "utils.data.DataLoader".

# 创建 Compose 对象进行一系列变换
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])

# 创建 dataset 对象 指定了存储位置 , 各种参数 , 还输入了变换对象 transform
mnist_train = datasets.MNIST('./data', train=True,  download=True, transform=transform)
mnist_test  = datasets.MNIST('./data', train=True, download=True, transform=transform)

# 创建了 DataLoader
trainloader = utils.data.DataLoader(mnist_train, batch_size=50, shuffle=True,  num_workers=2)
testloader  = utils.data.DataLoader(mnist_test, batch_size=100, shuffle=False, num_workers=2)



# To see an example of a batch (10 images) of training data 
# Change the "trainloader" to "testloader" to see test data
iter_data = iter(trainloader)
#iter_data = iter(testloader)

images, labels = next(iter_data)
print(images.size())
print(labels)


# Show image
# 将 image 的 50或100个图像排列成 grid  
show_imgs = torchvision.utils.make_grid(images, nrow=10).numpy().transpose((1,2,0))
plt.imshow(show_imgs)

```
### 11.1.3. iteration

```py

# 网络对象
net = LeNet().cuda()
# 损失函数对象
loss_func = nn.CrossEntropyLoss()
# 优化器对象
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 当前损失
running_loss = 0.0

ct_num = 0

# 对整个 DataLoader 进行迭代  data 包含了输入数据和期望输出labels
for iteration, data in enumerate(trainloader):

  # Take the inputs and the labels for 1 batch.
  # 获得数据以及 batch size 
  inputs, labels = data
  bch = inputs.size(0)


  #inputs = inputs.view(bch, -1) <-- We don't need to reshape inputs here (we are using CNNs).
  
  # 将数据存储到GPU
  # Move inputs and labels into GPU
  inputs = inputs.cuda()
  labels = labels.cuda()

  # 每次迭代先清空优化器
  # Remove old gradients for the optimizer.
  optimizer.zero_grad()


  # 前向计算
  # Compute result (Forward)
  outputs = net(inputs)

  # 使用损失函数计算损失  
  # Compute loss
  loss    = loss_func(outputs, labels)

  # 计算梯度
  # Calculate gradients (Backward)
  loss.backward()

  # 更新参数  
  # Update parameters
  optimizer.step()
  
  # 获取总共损失和迭代次数
  #with torch.no_grad():
  running_loss += loss.item()
  ct_num+= 1

  if iteration%50 == 49:

    # 输出迭代次数和 损失
    print("[Epoch: "+str(epoch+1)+"]"" --- Iteration: "+str(iteration+1)+", Loss: "+str(running_loss/ct_num)+'.')
  # Test
  if iteration%300 == 299:

    # 运行评价模型函数
    evaluate_model()
    
    # 用于分块监测
    epoch += 1



```



### 11.1.4. evaluate

```py

def evaluate_model():

  print("Testing the network...")

  # 进入 evaluation mode
  net.eval()

  # 总共样本数和正确预测数
  total_num   = 0
  correct_num = 0

  # 迭代 test 数据 DataLoader
  for test_iter, test_data in enumerate(testloader):

    # 获取一个 batch 的数据
    # Get one batch of test samples
    inputs, labels = test_data    
    bch = inputs.size(0)
    #inputs = inputs.view(bch, -1) <-- We don't need to reshape inputs here (we are using CNNs).

    # 存入GPU
    # Move inputs and labels into GPU
    inputs = inputs.cuda()
    labels = torch.LongTensor(list(labels)).cuda()

    # 向前迭代得到输出
    # Forward
    outputs = net(inputs)   

    # 根据输出层内容的到预测值
    # Get predicted classes
    _, pred_cls = torch.max(outputs, 1)

#     if total_num == 0:
#        print("True label:\n", labels)
#        print("Prediction:\n", pred_cls)

    # Record test result
    correct_num+= (pred_cls == labels).float().sum().item()
    total_num+= bch

  # 返回training mode
  net.train()
  
  print("Accuracy: "+"%.3f"%(correct_num/float(total_num)))

```

## 11.2. MINST GAN 例程

### 11.2.1. dataset

```py
# Define transform func.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])])



# Define dataloader
# 从 pytorch 数据库定义 dataset
train_dataset = datasets.MNIST(root='./mnist_data/', train=True,  transform=transform, download=True )
test_dataset  = datasets.MNIST(root='./mnist_data/', train=False, transform=transform, download=False)

# 定义 data loader
train_loader  = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bch_size, shuffle=True )
test_loader   = torch.utils.data.DataLoader(dataset= test_dataset, batch_size=bch_size, shuffle=False)
```

### 11.2.2. 网络

```py
# 训练 epoch
epo_size  = 200

# batch size
bch_size  = 100    # batch size

# 学习速度
base_lr   = 0.0001 # learning rate

# data dimension
mnist_dim = 784    # =28x28, 28 is the height/width of a mnist image.

# z dimension
z_dim     = 100    # dimension of the random vector z for Generator's input.


# Define the two networks
# G 4个全连接层输出 784
class Generator(nn.Module):
    def __init__(self, g_input_dim=100, g_output_dim=784):
        super(Generator, self).__init__()       
        self.fc1 = nn.Linear(g_input_dim, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features*2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features*2)
        self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)
    
    # forward method
    def forward(self, x): 
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.tanh(self.fc4(x))
    
# 四个全连接层输出 1/0
class Discriminator(nn.Module):
    def __init__(self, d_input_dim=784):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features//2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features//2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)
    
    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        return torch.sigmoid(self.fc4(x))


# Initialize a Generator and a Discriminator. 
G = Generator(g_input_dim = z_dim, g_output_dim = mnist_dim).cuda()
D = Discriminator(mnist_dim).cuda()

# Loss func. BCELoss means Binary Cross Entropy Loss.
criterion = nn.BCELoss() 

# Initialize the optimizer. Use Adam.
G_optimizer = optim.Adam(G.parameters(), lr = base_lr)
D_optimizer = optim.Adam(D.parameters(), lr = base_lr)
```

### 11.2.3. G 训练

```py
# Code for training the generator
def G_train(bch_size, z_dim, G_optimizer):
    G_optimizer.zero_grad()

    # 生成一个随机 z
    z = Variable(torch.randn(bch_size, z_dim)).cuda()
    # 期望的 D 输出是全 1 
    y = Variable(torch.ones(bch_size, 1)).cuda()
    # 生成 fake image
    G_output = G(z)
    D_output = D(G_output)
    # 计算 cost
    G_loss = criterion(D_output, y) # Fool the discriminator :P

    # Only update G's parameters
    G_loss.backward()
    G_optimizer.step()
        
    return G_loss.data.item(), G_output

```

### 11.2.4. D 训练
对于每次 D 训练, 先输入一组 real image 再输入一组 fake image 作为一次训练流程  

```py
# Code for training the discriminator.
def D_train(x, D_optimizer):
    D_optimizer.zero_grad()
    b,c,h,w = x.size()

    # train discriminator on real image
    x_real, y_real = x.view(-1, mnist_dim), torch.ones(b, 1)
    x_real, y_real = Variable(x_real).cuda(), Variable(y_real).cuda()

    D_output = D(x_real)
    
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # train discriminator on fake
    # 生成一个随机 z
    z      = Variable(torch.randn(b, z_dim)).cuda()
    # 期望的输出是全 0 
    y_fake = Variable(torch.zeros(b, 1)).cuda()
    # 获取 fake image
    x_fake = G(z)

    # 获取 fake image 的 cost
    D_output = D(x_fake.detach()) # Detach the x_fake, no need grad. for Generator.
    D_fake_loss = criterion(D_output, y_fake)
    D_fake_score = D_output

    # 两个loss 相加
    # Only update D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()
        
    return  D_loss.data.item()
```

### 11.2.5. iteration

```py

D_epoch_losses, G_epoch_losses = [], []   # record the average loss per epoch.

# iteration 
for epoch in range(1, epo_size+1):
    D_losses, G_losses = [], []     

    # 对于一个 train batch 进行训练
    for iteration, (x, _) in enumerate(train_loader):
        # Train discriminator 
        D_loss = D_train(x, D_optimizer)
        D_losses.append(D_loss)
        # Train generator
        G_loss, G_output = G_train(bch_size, z_dim, G_optimizer)
        G_losses.append(G_loss)

    # Record losses for logging
    D_epoch_loss = torch.mean(torch.FloatTensor(D_losses))
    G_epoch_loss = torch.mean(torch.FloatTensor(G_losses))
    D_epoch_losses.append(D_epoch_loss)
    G_epoch_losses.append(G_epoch_loss)

    # Convert G_output to an image.
    G_output = G_output.detach().cpu()
    G_output = G_output.view(-1, 1, 28, 28)

    # Logging 
    Logging(G_output, G_epoch_losses, D_epoch_losses)
    print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
            (epoch), epo_size, D_epoch_loss, G_epoch_loss))
    
    # Save G/D models
    save_pth_G = save_root+'G_model.pt'
    save_pth_D = save_root+'D_model.pt'
    torch.save(G.state_dict(), save_pth_G)
    torch.save(D.state_dict(), save_pth_D)
print("Training is finished.")



# Logging  日志函数 , 用于保存loss 随 epoch 的变化
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output, display

def Logging(images, G_loss, D_loss):
    clear_output(wait=True)
    plt.clf()
    x_values = np.arange(0,len(G_loss), 1)
    fig, ax = plt.subplots()
    ax.plot(G_loss, label='G_loss')
    ax.plot(D_loss, label='D_loss')
    legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large')
    plt.grid(linestyle='-')
    plt.title("Training loss")
    plt.ylabel("Loss")
    plt.show()
    show_imgs = torchvision.utils.make_grid(G_output, nrow=10).numpy().transpose((1,2,0))
    plt.imshow(show_imgs)
    plt.show()
        
```

## 11.3. MINST USPS adversarial examples

### 11.3.1. dataset

```py
# Make MINIST dataloaders
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
mnist_train = datasets.MNIST('./data', train=True,  download=True, transform=transform)
mnist_test  = datasets.MNIST('./data', train=False, download=True, transform=transform)
mnist_trainloader = utils.data.DataLoader(mnist_train, batch_size=50, shuffle=True,  num_workers=2)
mnist_testloader  = utils.data.DataLoader(mnist_test,  batch_size=1,  shuffle=False, num_workers=2)

# Make USPS dataloaders
transform = transforms.Compose([transforms.Resize(size=(28, 28), interpolation=Image.BICUBIC),transforms.ToTensor(), 
            transforms.Normalize((0.1307,), (0.3081,))])
usps_train = datasets.USPS('./data', train=True,  download=True, transform=transform)
usps_test  = datasets.USPS('./data', train=False, download=True, transform=transform)
usps_trainloader = utils.data.DataLoader(usps_train, batch_size=50, shuffle=True,  num_workers=2)
usps_testloader  = utils.data.DataLoader(usps_test,  batch_size=1,  shuffle=False, num_workers=2)

```

### 11.3.2. general train and evaluate

```py
# Script for training a network
# 输入参数 网络,优化器,训练数据loader,loss func,训练次数,使用GPU
def train_model(net, optimizer, dataloader, loss_func, epoch_size=2, use_gpu=True):
    # loss 变化记录
    loss_rec    = []    # For plotting loss
    # epoch iteration
    for epoch in range(epoch_size):
        running_loss = 0.0
        ct_num = 0
        # 遍历 loader 数据
        for iteration, data in enumerate(dataloader):
            inputs, labels = data
            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()
            # 初始化优化器
            optimizer.zero_grad()
            # 前向传播
            outputs = net(inputs)
            # 计算 loss
            loss    = loss_func(outputs, labels)
            # 根据loss backward
            loss.backward()
            # 更新参数
            optimizer.step()

            # 格式化输出训练进度
            with torch.no_grad():
                running_loss+= loss.item()
                ct_num+= 1
                if iteration%10 == 9:
                    clear_output(wait=True)
                    print("Epoch ["+str(epoch+1)+"/"+str(epoch_size)+"], Iteration: "+str(iteration+1)+", Loss: "+str(running_loss/ct_num)+'.')
                    loss_rec.append(running_loss/ct_num)
    # 返回训练 loss 变化 array
    return loss_rec

# Script for testing a network
# 输如 网络,测试数据loader,网络名称,是否预训练
def evaluate_model(net, dataloader, tag, Pretrained=None, resize_to=None):
    print("Testing the network...")
    if not Pretrained is None:
        # 直接读取网络状态
        net.load_state_dict(torch.load(Pretrained))
    # 网络进入 eval 模式
    net.eval()
    total_num   = 0
    correct_num = 0

    # 遍历 test loader
    for test_iter, test_data in enumerate(dataloader):
        # Get one batch of test samples
        inputs, labels = test_data    
        if not resize_to is None:
            inputs = F.interpolate(inputs, size=resize_to, mode='bicubic')
        bch = inputs.size(0)

        # Move inputs and labels into GPU
        inputs = inputs.cuda()
        labels = torch.LongTensor(list(labels)).cuda()

        # Forward
        outputs = net(inputs)   

        # Get predicted classes
        _, pred_cls = torch.max(outputs, 1)

        # Record test resultplot_traininigloss
        correct_num+= (pred_cls == labels).float().sum().item()
        total_num+= bch

    # 返回训练模式
    # net.train()
    print(tag+"- accuracy: "+"%.2f"%(correct_num/float(total_num)))
```