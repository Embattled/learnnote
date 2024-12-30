- [1. Python API](#1-python-api)
- [2. torch](#2-torch)
  - [2.1. Tensors](#21-tensors)
    - [2.1.1. Creation Ops](#211-creation-ops)
    - [2.1.2. Indexing, Slicing, Joining, Mutating Ops 拼接与截取等](#212-indexing-slicing-joining-mutating-ops-拼接与截取等)
      - [2.1.2.1. Indexing](#2121-indexing)
      - [2.1.2.2. Joining 函数](#2122-joining-函数)
      - [2.1.2.3. Dimension - 维度操作函数](#2123-dimension---维度操作函数)
      - [2.1.2.4. Slicing - 切片函数](#2124-slicing---切片函数)
  - [2.2. Generators, Random Sampling - 随机采样](#22-generators-random-sampling---随机采样)
    - [2.2.1. Generator 相关](#221-generator-相关)
    - [2.2.2. Random Sampling - 默认使用全局 rng](#222-random-sampling---默认使用全局-rng)
  - [2.3. 序列化 Serialization](#23-序列化-serialization)
    - [2.3.1. torch.save](#231-torchsave)
    - [2.3.2. torch.load](#232-torchload)
  - [2.4. Parallelism  - 并行化信息](#24-parallelism----并行化信息)
  - [2.5. Locally disabling gradient computation - 局部禁用梯度计算](#25-locally-disabling-gradient-computation---局部禁用梯度计算)
  - [2.6. Math operations](#26-math-operations)
    - [2.6.1. Pointwise Ops 元素为单位的操作](#261-pointwise-ops-元素为单位的操作)
      - [2.6.1.1. special 特殊高精度计算](#2611-special-特殊高精度计算)
    - [2.6.2. Reduction Ops 元素之间的操作(降维)](#262-reduction-ops-元素之间的操作降维)
      - [2.6.2.1. 极值操作](#2621-极值操作)
    - [2.6.3. Comparison Ops](#263-comparison-ops)
    - [2.6.4. Spectral Ops - 光谱函数 在频域上操作信号](#264-spectral-ops---光谱函数-在频域上操作信号)
    - [2.6.5. Other Operations - 无法分类的其他函数](#265-other-operations---无法分类的其他函数)
- [3. torch.nn](#3-torchnn)
  - [3.1. Containers 网络容器](#31-containers-网络容器)
    - [3.1.1. torch.nn.Module 类](#311-torchnnmodule-类)
      - [3.1.1.1. 基础方法及应用](#3111-基础方法及应用)
      - [3.1.1.2. 网络参数以及存取](#3112-网络参数以及存取)
      - [3.1.1.3. 特殊参数](#3113-特殊参数)
    - [3.1.2. torch.nn.Sequential 系列](#312-torchnnsequential-系列)
  - [3.2. Convolution Layers 卷积层](#32-convolution-layers-卷积层)
  - [3.3. Pooling layers 池化层](#33-pooling-layers-池化层)
  - [3.4. Padding Layers](#34-padding-layers)
  - [3.5. Non-linear Activations (weighted sum, nonlinearity) 非线性激活函数](#35-non-linear-activations-weighted-sum-nonlinearity-非线性激活函数)
    - [3.5.1. nn.Softmax](#351-nnsoftmax)
  - [3.6. Normalization Layers 归一化层](#36-normalization-layers-归一化层)
  - [3.7. Linear Layers  线性层](#37-linear-layers--线性层)
    - [3.7.1. Identity](#371-identity)
    - [3.7.2. Linear](#372-linear)
  - [3.8. Loss Function 损失函数](#38-loss-function-损失函数)
    - [3.8.1. Binary Cross Entropy (BCE)](#381-binary-cross-entropy-bce)
  - [3.9. Vision Layrers - 与图像相关的网络层](#39-vision-layrers---与图像相关的网络层)
  - [3.10. ChannelShuffle - 重排列 Channel](#310-channelshuffle---重排列-channel)
- [4. torch.nn.functional](#4-torchnnfunctional)
  - [4.1. Convolution functions](#41-convolution-functions)
  - [4.2. Non-linear Activations](#42-non-linear-activations)
    - [4.2.1. F.glu](#421-fglu)
    - [4.2.2. F.gelu](#422-fgelu)
    - [4.2.3. F.sigmoid](#423-fsigmoid)
    - [4.2.4. F.batch\_norm](#424-fbatch_norm)
    - [4.2.5. F.instance\_norm](#425-finstance_norm)
    - [4.2.6. F.layer\_norm](#426-flayer_norm)
    - [4.2.7. F.normalize](#427-fnormalize)
  - [4.3. Vision functions](#43-vision-functions)
- [5. torch.Tensor 张量类](#5-torchtensor-张量类)
  - [5.1. torch.Tensor 的格式](#51-torchtensor-的格式)
  - [5.2. 类属性](#52-类属性)
  - [5.3. 类方法](#53-类方法)
    - [5.3.1. 类型转换](#531-类型转换)
    - [5.3.2. view 变形](#532-view-变形)
    - [5.3.3. transpose](#533-transpose)
  - [5.4. 创建操作 Creation Ops](#54-创建操作-creation-ops)
    - [5.4.1. 统一值 tensor](#541-统一值-tensor)
    - [5.4.2. 随机值 random](#542-随机值-random)
    - [5.4.3. like 类方法](#543-like-类方法)
    - [5.4.4. torch.from\_numpy](#544-torchfrom_numpy)
    - [5.4.5. tensor复制](#545-tensor复制)
    - [5.4.6. .new\_ 方法](#546-new_-方法)
- [6. torch.amp - Automatic Mixed Precision package](#6-torchamp---automatic-mixed-precision-package)
  - [6.1. Autocasting - torch 接口](#61-autocasting---torch-接口)
    - [6.1.1. Gradient Scaling](#611-gradient-scaling)
  - [6.2. Autocast Op Reference - 自动类型转换参考](#62-autocast-op-reference---自动类型转换参考)
- [7. torch.autograd - 梯度计算包](#7-torchautograd---梯度计算包)
  - [7.1. Locally disabling gradient computation](#71-locally-disabling-gradient-computation)
- [8. Torch Devices](#8-torch-devices)
  - [8.1. torch.cpu - 虚类实现](#81-torchcpu---虚类实现)
  - [8.2. torch.cuda - CUDA 计算](#82-torchcuda---cuda-计算)
    - [8.2.1. General - 通用接口](#821-general---通用接口)
    - [8.2.2. Memory management - CUDA 设备内存管理](#822-memory-management---cuda-设备内存管理)
- [9. torch.linalg - pytorch 的线性代数子库](#9-torchlinalg---pytorch-的线性代数子库)
  - [9.1. Matrix Properties](#91-matrix-properties)
  - [9.2. Decompositions](#92-decompositions)
- [10. torch.profiler](#10-torchprofiler)
  - [10.1. class torch.profiler.profile](#101-class-torchprofilerprofile)
  - [10.2. class torch.profiler.\_KinetoProfile](#102-class-torchprofiler_kinetoprofile)
- [11. torch.onnx](#11-torchonnx)
  - [11.1. TorchDynamo-based ONNX Exporter](#111-torchdynamo-based-onnx-exporter)
  - [11.2. TorchScript-based ONNX Exporter](#112-torchscript-based-onnx-exporter)
    - [11.2.1. API of TorchScript-based ONNX Exporter](#1121-api-of-torchscript-based-onnx-exporter)
- [12. torch.optim](#12-torchoptim)
  - [12.1. 预定义优化 Algorithm](#121-预定义优化-algorithm)
    - [12.1.1. optim 通用参数](#1211-optim-通用参数)
  - [12.2. Adjust learning rate - lr\_scheduler 动态 Learn Rate](#122-adjust-learning-rate---lr_scheduler-动态-learn-rate)
    - [12.2.1. Optim Scheduler 通用参数和方法](#1221-optim-scheduler-通用参数和方法)
    - [12.2.2. Optim Class](#1222-optim-class)
    - [12.2.3. 有序调整](#1223-有序调整)
  - [12.3. 定义自己的 optim](#123-定义自己的-optim)
    - [12.3.1. Optimizer 基类](#1231-optimizer-基类)
    - [12.3.2. optimization step](#1232-optimization-step)
    - [12.3.3. per-parameter options](#1233-per-parameter-options)
- [13. Autograd mechanics](#13-autograd-mechanics)
  - [13.1. Locally disabling gradient computation](#131-locally-disabling-gradient-computation)
    - [13.1.1. Setting requires\_grad](#1311-setting-requires_grad)
    - [13.1.2. Grad Modes](#1312-grad-modes)
    - [13.1.3. No-grad Mode](#1313-no-grad-mode)
    - [13.1.4. Inference Mode](#1314-inference-mode)
    - [13.1.5. Evaluation Mode (nn.Module.eval())](#1315-evaluation-mode-nnmoduleeval)


# 1. Python API

官方对于 分类感觉做的不是很好, 没办法细分成子文档


按照官方文档的顺序排序, 尽可能的把小节整合到一起




# 2. torch

* 作为最基础的包, torch 包括了tensors的数据格式以及对应的数学操作. 
* 还提供了许多工具来高效的序列化 tensors 以及其他任意数据格式   
* 该包有 CUDA 对应  

The torch package contains data structures for multi-dimensional tensors and defines mathematical operations over these tensors. Additionally, it provides many utilities for efficient serialization of Tensors and arbitrary types, and other useful utilities.

It has a CUDA counterpart, that enables you to run your tensor computations on an NVIDIA GPU with compute capability >= 3.0.

## 2.1. Tensors 

张量操作相关的接口  

### 2.1.1. Creation Ops

基本上所有创建 tensor 的函数都位于 `torch.` 下  

创建tensor 有许多通用的参数
* `dtype  =`
* `device =`


* 从 python list 或者 numpy array 来创建张量
* `torch.tensor()` 函数总是会进行数据拷贝
  * 防止拷贝可以使用 `torch.as_tensor()`
```py
torch.tensor(data, *, dtype=None, device=None, requires_grad=False, pin_memory=False) → Tensor
``` 


torch 的创建等间隔函数, 返回值都是 1-D 的
* torch.arange    : 不包含 end 的区间
* torch.range     : 包含 end 的区间, 这点和 python 内置的 range 不同
  * 因此使用该函数本身已经被附加了 deprecated 的警告
* torch.linspace  : 获取从区间等分的 N 个数, 指定 N 而不是指定 step
* torch.logspace  : 在给定的 base 上, 获取 指数从 `[start, end]` 上的 step 个函数


### 2.1.2. Indexing, Slicing, Joining, Mutating Ops 拼接与截取等

类似于 Numpy 中对序列进行各种操作的函数集  


#### 2.1.2.1. Indexing 

`torch.gather(input, dim, index, *, sparse_grad=False, out=None) → Tensor`
* 形容起来比较复杂, 提供两个**相同 维度数**的 Tensor input 和 index
  * `input.dim == index.dim` 
  * index 和 input 不会互相 broadcast, 且需要在dim 以外的所有维度上 `index.size(d) <= input.size(d)`
  * 输出的 shape 和 index 相同
* 根据指定的 dim, 参照 index 提供的数据在 input 的 dim 维度重新选择数据输出
  * 因此属于 Indexing 函数

```py
torch.gather(t, 1, torch.tensor([[0, 0], [1, 0]]))
tensor([[ 1,  1],
        [ 4,  3]])
```


高级索引方法, beta 函数: This function is in beta and may change in the near future.
* `index_reduce(input, dim, index, source, reduce, *, include_self=True, out=None)→ Tensor`
  * 对于 `index[i]=j`, `result[j]=f(result[j], source[i])`
    * 因此shape的要求是:
    * `source.shape[dim] == len(index)` index 每一个元素对应一个 source
    * source.shape 的 dim 以外的维度需要和 input 匹配
  * `self[index[i], :, :] *= src[i, :, :]  # if dim == 0`
  * 参数:
    * `dim` :  (int): index 的维度
    * `index` :  (Tensor): 索引 Tensor, 必须是整数
    * `source` :  (FloatTensor) : 值 Tensor
    * `reduce` : (str) : 降维方法
  * 看得出来 这里的 `index` 是一个1维 Tensor, 在执行 reduce 的时候, 索引访问的切片会作为一个整体, 而不是单个元素, 因此可以在reduce内部可以进行 broadcast
    * 对于 shape 的约束比较小
  * This function only supports floating point tensors.

* `scatter_reduce(input, dim, index, src, reduce, *, include_self=True)→ Tensor`
  * 从功能上看上去 index_reduce 一样
  * 需要所有输入的 dim 统一 `self, index and src` should all have the same number of dimensions
  * `self[index[i][j][k]][j][k] += src[i][j][k]  # if dim == 0`
  * 参数也基本上一样:
    * `dim`, `index`, `src` ``reduce`
    * index 支持多维度
  * 从shape上
    * 对于所有维度, 需要有 `index.size(d) <= src.size(d)`. 即每一个 index 元素 都有对应的 src 元素, index 和 src 不会进行 broadcast
    * 对于 `d!= dim`, 需要有 `index.size(d) <= self.size(d)`, 即除了索引应用的维度以外, 同样需要 index 每个元素有对应的 self 的位置.
    * 索引维度除外, 要考虑 index 元素值的范围是否在 self 对应维度之内
  * backward: 使用该函数的时候要考虑是否有反向传播的需求, 只有在 `src.shape==index.shape` 的时候才能够反向传播






#### 2.1.2.2. Joining 函数 


`torch.stack(tensors, dim=0, *, out=None) → Tensor`  

* 将多个 tensor 叠加到一起, 并产生一个新的 dimension
* Concatenates a sequence of tensors along a new dimension.
* 因此所有 tensor 必须相同大小
* dim 指定新的 dimension 的位置


`torch.cat(tensors, dim=0, *, out=None) → Tensor`  

* 将多个 tensor 链接, 沿着最外层 index 或者指定 dim 链接
* All tensors must either have the same shape (except in the concatenating dimension) or be empty.
* dim (int, optional) – the dimension over which the tensors are concatenated


`torch.tile(input, dims) → Tensor`
* 指定某个维度按照一定的格式进行平铺
* 类似于 numpy.tile()
* Tensor 类本身也有实现, 但是名字不统一 `Tensor.repeat()`

#### 2.1.2.3. Dimension - 维度操作函数 

`torch.squeeze(input, dim = None)`
* 把一个 tensor 的所有 size 为 1 的 维度移除
* 通过指定 dim 可以限定操作的维度
* input (Tensor) – the input tensor.
* dim (int or tuple of ints, optional).  

`torch.unsqueeze(input, dim) → Tensor`
* Returns a new tensor with a dimension of size one inserted at the specified position.
* 在指定位置添加一个维度
* dim(int) 为必须参数, 可以是  `[-input.dim() - 1, input.dim() + 1)` 中的值

#### 2.1.2.4. Slicing - 切片函数 


`torch.chunk`  `torch.tensor_split` : 将 tensor 按照某个维度切片


## 2.2. Generators, Random Sampling - 随机采样

### 2.2.1. Generator 相关

常识  `rng` 是 `Random Number Generator` 的缩写

`class torch.Generator(device='cpu')`   主要用于独立的管理随机数生成器的状态, 用法是作为参数传入其他 torch. 下的随机数生成
* `get_state() → Tensor`                返回随机数生成器的状态 as a `torch.ByteTensor`
* `set_state(new_state) → void`         设置生成器为一个状态  
* `initial_seed() → int`                获取该随机生成器的初始种子
* `manual_seed(seed) → Generator`       手动设置种子, 不需要接受返回值, in-place 实现
* `seed() → int`                        自动设置种子, 根据当前时间, 不需要接受返回值

torch. 下的针对全局 Generator 有同样名称的一组接口
* `torch.seed() ->int`
* `torch.manual_seed(seed) -> Generator`
* `torch.initial_seed() -> int`
* `torch.get_rng_state() -> Tensor`   接口的名称多了 `rng`
* `torch.set_rng_state(new_state)`


### 2.2.2. Random Sampling - 默认使用全局 rng

一些和 numpy 重名的函数反而效果不同, 需要多留意

* rand    : 随机 `[0,1)`


## 2.3. 序列化 Serialization

和存储相关, 将各种模型, 张量, 字典等数据类型序列化后存储到文件, 或者从文件中读取, 并不单只能用来存取网络  

pytorch 的load() 是基于 pickle 模组的, 不要轻易unpick不信任的序列数据  

Pytorch 的约定是 用 `.pt` 来保存 tensors

一共就两个函数
* torch.save()
* torch.load()

官方推荐的网络模型存取方式:
* `torch.save(model.state_dict(), PATH)`          ： 即只保存网络的参数, 不保存网络的格式
* `model.load_state_dict(torch.load(PATH))`       : 创建好网络之后, 再从文件中读取参数

完整模型的存取
* `torch.save(model, PATH)`    : 直接保存整个网络
* `model = torch.load(PATH)`

### 2.3.1. torch.save

```py
torch.save(
  obj, 
  f: Union[str, os.PathLike, BinaryIO], 
  pickle_module=pickle,
  pickle_protocol=DEFAULT_PROTOCOL, 
  _use_new_zipfile_serialization=True # 代表使用 pytorch 1.6 后的新的压缩格式
  ) → None
```

参数意思:
* `obj`   : 要保存的对象
* f       : a file-like object
* pickle_module   :   module used for pickling metadata and objects
* pickle_protocol :   can be specified to override the default protocol
* _use_new_zipfile_serialization : 如果要读取 pytorch 1.6 之前的旧数据, 传入 False

### 2.3.2. torch.load

```py
torch.load(
  f, 
  map_location=None, 
  pickle_module=pickle, 
  *,
  weights_only=False, 
  **pickle_load_args
  )

torch.load('tensors.pt')
# Load all tensors onto the CPU
torch.load('tensors.pt', map_location=torch.device('cpu'))
# Map tensors from GPU 1 to GPU 0
torch.load('tensors.pt', map_location={'cuda:1':'cuda:0'})
```

f   : a file-like object
map_location    : a function, torch.device, string or a dict specifying how to remap storage locations
pickle_module   :  
pickle_load_args: (Python 3 only) optional keyword arguments passed over to pickle_module.load() and pickle_module.Unpickler()


## 2.4. Parallelism  - 并行化信息



## 2.5. Locally disabling gradient computation - 局部禁用梯度计算  

梯度上下文管理有三个接口, 用于在局部对梯度计算进行启用, 停用:
* 在 `torch.autograd` 中有更详细的描述  
* 这些接口大部分都属于 Context-manager, 所以该种类接口一般情况下需要与 `with` 语法一起使用  
  * 该 Context-manager 影响的是线程 thread local, 即整个线程上的 torch 推论都会受到影响  


梯度管理方法 `torch.*`
* `no_grad()`  :          Context-manager that disables gradient calculation.
  * 关闭梯度计算, 中断计算图, 需要确保不会调用 `Tensor.backward()`
  * 注意, 对于 factory funcions, 或者在函数里创建的 new Tensor 并设置 requires_grad 为True, 则无法受此上下文管理器影响
  * 也可以作为 函数修饰器使用
* `enable_grad` :       Context-manager that enables gradient calculation


* inference_mode   :  Context-manager that enables or disables inference mode

* is_grad_enabled  :  Returns True if grad mode is currently enabled.
* is_inference_mode_enabled :  	Returns True if inference mode is currently enabled.

可以作为函数使用的接口:
* `class torch.set_grad_enabled(mode)` :  Context-manager that sets gradient calculation on or off.
  * Context-manager that sets gradient calculation on or off.
  * 这个接口同其他几个不同, 还可以作为函数使用用来全局的关闭梯度计算.  
  * 参数 : mode (bool)


```py

# 作为函数修饰器使用
@torch.no_grad()
def tripler(x):
     return x * 3

```


## 2.6. Math operations

提供对 tensor 的各种数学操作

### 2.6.1. Pointwise Ops 元素为单位的操作

* 三角函数
  * torch.sin

* 取倒数
  * torch.reciprocal


#### 2.6.1.1. special 特殊高精度计算  

有一部分函数是 torch.special 中接口的别名   
torch.special 则是仿照 scipy 的 special 模组  


对于非常接近 0 的微小值的计算, 如果直接实现计算数式的话会导致精度损失  
* `expm1`   : `exp(x) -1`, alias for `torch.special.expm1()`
* `log1p`   : `log(x+1)`, 


### 2.6.2. Reduction Ops 元素之间的操作(降维)

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


* 求范数
  * torch.norm()    : 求范数, 该函数已经被 deprecated. 使用 `torch.linalg` 中的对应函数




`torch.logsumexp(input, dim, keepdim=False, *, out=None)`
* 求 log sum exp , 先取自然数的幂, 再求和, 再取对数
* 直接应用了 Log-Sum-Exp Trick 避免数值溢出
* 作为降维方法使用


#### 2.6.2.1. 极值操作

* max/min : 返回最大值, 默认对全元素进行, 可以输入单个整数 dim 来对某一维度进行运算, 此时会返回两个值, 第二个返回值是索引位置
* argmax/argmin : 返回最大值的索引, 默认对全元素进行操作, 可以输入单个整数 dim 来对某一维度进行运算, 等同于 max/min 输入 dim 的第二个返回值
* amax/amin : 返回最大值, 专门用来对指定维度进行运算, dim 是必须参数且可以是 int or tuple, 即可以对多个维度进行运算, 不会返回索引

### 2.6.3. Comparison Ops

专门用来比较的函数 

获取指定维度的 k 个最大值, 同时还能获得对应的 索引, 用在分类任务上  
`torch.topk(input, k, dim=None, largest=True, sorted=True, *, out=None) -> (Tensor, LongTensor)`  
* dim, int 只能指定单独维度, 默认会对最后一个维度进行操作
* largest: bool, true 则选择最大的top, 否则选择最小的 top
* sorted: bool, 是否按对应的顺序返回这k个值


### 2.6.4. Spectral Ops - 光谱函数 在频域上操作信号

### 2.6.5. Other Operations - 无法分类的其他函数

参照 numpy 的相关实现

cum 累计系类:
* `torch.cumsum(input, dim, *, dtype=None, out=None) → Tensor`    : 将输入沿着某一维度 累加

broadcast 家族:
* `torch.broadcast_to(input, shape) → Tensor` : 手动 broadcast 
* `torch.broadcast_tensors(*tensors) → List of Tensors[source]` : 同时 broadcast 一整组张量



`torch.searchsorted(sorted_sequence, values, *, out_int32=False, right=False, side='left', out=None, sorter=None) → Tensor`
* 寻找有序插入位置
* 常被用来实现基于 torch 的线性插值, 因为 torch 似乎没有直接提供类似于 `numpy.interp` 用于非自然数索引的插值方法
* `right=False` 返回的插入位置是达成有序的最左边, 即会插入到相同值的元素的前面, 如果至 True, 则代表右边


`torch.meshgrid(*tensors, indexing=None)`
* 生成 `*tensors` 作为索引构成的坐标 grid
* indexing =`ij` 是默认索引顺序, 等同于先行后列


`torch.repeat_interleave(input, repeats, dim=None, *, output_size=None)`
* This is different from torch.Tensor.repeat() but similar to numpy.repeat.
* 指定每一个元素各自重复多少次, 类似于 numpy.repeat()


# 3. torch.nn

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

## 3.1. Containers 网络容器
 
Module            Base class for all neural network modules.  
Sequential        A sequential container.  
ModuleList        Holds submodules in a list.  
ModuleDict        Holds submodules in a dictionary.  
ParameterList     Holds parameters in a list.  
ParameterDict     Holds parameters in a dictionary.  

### 3.1.1. torch.nn.Module 类

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

#### 3.1.1.1. 基础方法及应用

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

# 定义好网络后, 可以直接通过成员网络层输出当前网络的参数
net = Model().cuda()

print(net.conv1.weight.size()) 
print(net.conv1.weight)
print(net.conv1.bias)

```

#### 3.1.1.2. 网络参数以及存取

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

#### 3.1.1.3. 特殊参数


`Module.register_buffer(name, tensor, persistent=True)`
* 向模型中添加一个 参数 buffer, 标准名称为 缓冲区张量
* 其属性很有意思, 可以参与前向传播但是不参与梯度计算  (not to be considered a model permeter) 
  * 从根本上该张量不属于模型参数, 不会被任何优化器更新
  * 和设置 requires_grad=False 的参数相比较, 后者仅仅是暂时冻结参数, 但本质上仍然属于模型参数, 甚至可以被优化器操作, 因此可能会被自定义的优化器更新
  * register_buffer 不会包含在  `Module.parameers()` 中 反而会出现在 `Module.buffer()` 中
  * register_buffer 的内容默认也会出现在 `state_dict()`中, 可以指定 `persistent=False` 来使得其不出现在 state_dict



### 3.1.2. torch.nn.Sequential 系列

A sequential container.
`class torch.nn.Sequential(*args: Module)`
`class torch.nn.Sequential(arg: OrderedDict[str, Module])`

sequential 也是最常用的模型容器, 接受 `*args` 和 `OrderedDict` 两种构造方法, 然而从结果上并没有不同  

sequential 的价值在于可以将整个容器是为单个模块
sequential 本身与 ModuleList 相似, 然后 ModuleList 更加偏向于以 list 的形式管理 Module, sequential 更偏向于整体


## 3.2. Convolution Layers 卷积层 

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
  * 将滑动窗口提出出来并扁平化
  * 对于输入为 `N,C, H,W` 的情况, 输出为 `N, C*k, L`
    * k 是滑动窗口的元素个数
    * L 是滑动窗口的采样次数
  * 采样次数的维度反而放在最后一维, 同时滑窗元素本身的维度和 C 合并了, 这一点要注意
  * `torch.nn.Unfold(kernel_size, dilation=1, padding=0, stride=1)`
    * dilation: 是一个很难描述的一个算法的一部分, 定义了 spacing between the kernel points, 可以参照 [URL](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md), 默认为1
    * kernel_size, dilation, padding or stride 都会在输入的维度只有 1 的时候进行全`空间维度`扩张
    * 若要自定义的话可以使用 F.functional.unfold

* nn.Fold
  * Combines an array of sliding local blocks into a large containing tensor.
  * 将一系列滑动窗口的各个值返回到原本的张量, 对于重叠的位置, 简单的将他们相加. 因此 unfold 和 fold 并不是完全相反的操作
  * `torch.nn.Fold(output_size, kernel_size, dilation=1, padding=0, stride=1)`
    * 参数基本上等同于 Unfold
    * output_size : 主要用于窗口个数的歧义情况, 通过给定输出的空间 shape 来消除歧义, 注意该参数仅仅输入空间shape. i.e., `output.sizes()[2:]`
  * only unbatched (3D) or batched (4D) image-like output tensors are supported.
    * Input: (N,C×∏(kernel_size),L)(N,C×∏(kernel_size),L) - > Output: (N,C,output_size[0],output_size[1],… )(N,C,output_size[0],output_size[1],…)
    * (C×∏(kernel_size),L) -> (C,output_size[0],output_size[1],…) 


## 3.3. Pooling layers 池化层

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
* 自适应池化层
  * nn.AdaptiveMaxPool1d
  * nn.AdaptiveMaxPool2d
  * nn.AdaptiveMaxPool3d
  * nn.AdaptiveAvgPool1d
  * nn.AdaptiveAvgPool2d
  * nn.AdaptiveAvgPool3d



1,2,3 D 都支持传入一个 Output_size 的参数, 用于指定输出的特征平面的大小
* 可以输入单个整数, 对于 2D 和 3D 等同于直接指定输出为 `H*H H*H*H`
* 1,2,3 D 的含义是指定最后的 n 维度为执行池化的对象
* 其他的维度都作为 输入的平面, 其输出会维持原本的 shape

## 3.4. Padding Layers

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

## 3.5. Non-linear Activations (weighted sum, nonlinearity) 非线性激活函数

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

大部分说明都定义在 F. 中

### 3.5.1. nn.Softmax

`class torch.nn.Softmax(dim=None)`
* 对 n-维 输入应用 softmax, 会将输入 Tensor 缩放到 `[0,1]` 范围同时求和为 1
* 如果指定了 dim, 则 dim 的每一个值的切片都会分别 求和等于 1

$$
Softmax(x_i) = \frac{\exp(x_i)}{\sum_j\exp(x_j)} \\
Softmax(x_i) = \frac{\exp(x_i/\tau)}{\sum_j\exp(x_j/\tau)} \\
$$

$\tau$ 是一个softmax 的改进变量, 成为 softmax 的温度, 通过进一步缩放输入数据, 会导致经过 exp 之后各个分量之间的距离改变
* 高温度 tau > 1, softmax 的输出概率会变得更平滑, 类之间的距离变小
* 地温度 0< tau <1 , softmax 的输出变得尖锐, 最有可能的类的概率变高, 其他类的概率更低

温度通常作为一个超参数, 在一些学习中也作为可优化的参数直接并入学习, 对一个初始值为 0.0 的标量执行 exp (即 tau 为1)
$$\tau = \exp(s)$$
直接对 tau 进行优化有可能导致 温度值低于等于 0, 是无意义的

pytorch 中没有对温度功能的直接实现

## 3.6. Normalization Layers 归一化层

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



## 3.7. Linear Layers  线性层

用于构筑网络的全连接层, 例如最早期的 MLP  

* nn.Identity
* nn.Linear
* nn.Bilinear
* nn.LazyLinear

### 3.7.1. Identity

用在网络构造定义中的占位符, 类似于 pass, 可以吞入任何参数
```py
class
torch.nn.Identity(*args, **kwargs)

```

### 3.7.2. Linear

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

## 3.8. Loss Function 损失函数

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





### 3.8.1. Binary Cross Entropy (BCE)



`torch.nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')`
* 二元交叉熵损失, 通常用于 GAN 网络的 discriminator 
  * 真实样本的目标值为 1
  * fake 的目标值为 0
* `l(x,y) = -w(y*log(x) + (1-y)*log(1-x))`
  * 目标值 y 为 1 的时候, 希望 x 尽可能逼近1, 使得 log(x) 从负值逼近于 0, 在 loss 的开头加上 负号使得 loss 始终为 正值
  * 目标值 y 为 0 的时候, 希望 x 尽可能逼近0, 效果相同
  * y 必须是在 0到1之间的数字
* 这种损失是 GAN 网络最原始的损失, 在 discriminator 的判别精准度很好的时候会导致梯度消失, 因此该损失在实际中很少使用
* pytorch 中的实现, 会对 log 函数的输出限制在 -100 以内, 确保不会在 x =0 或者 1 的时候出现梯度爆炸
* 参数含义:
  * `weight` : 对于 batch 输入的时候, 指定每一个 element 的权重
  * `size_average` bool, optional : deprecated.
  * `reduce=None`  bool, optional : deprecated. 
  * `reduction='mean'` : 对于 batch 输入的时候, 计算最终 loss 的降维方法
    *  `mean` : 取均值
    *  `sum`  : 求和
    *  `none` : 不进行降维
    *  该 loss 用于自动编码器的重建误差


`torch.nn.BCEWithLogitsLoss(weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None)`
* BCELoss 的改进版本, 相当于 Sigmoid 后接一个 BCELoss, 同时使用了 log-sum-exp trick 使得计算更加稳定
* 计算上和 BCELoss 相比, 就单纯的对输入的 x 先进行 Sigmoid 处理
  * `l(x,y) = -w(y*log( sig(x) ) + (1-y)*log(1- sig(x) ))`



`torch.nn.HingeEmbeddingLoss(margin=1.0, size_average=None, reduce=None, reduction='mean')`
* 二元 Loss 的一种
* 期望值 分别是 1和 -1
* pytorch 的实现方法优点奇怪, 
* chatgpt 给出的公式倒是比较合理 `HingeLoss(x,y) = max(0, 1-yixi)`
* 但是 y 值似乎指定 -1, 1 以外的值没有意义



## 3.9. Vision Layrers - 与图像相关的网络层


`class torch.nn.PixelShuffle(upscale_factor)`
* 等同于 depth to space, 将一部分 Channel 转换为 Space
  * $(*, C\times r^2, H,W) \rightarrow (*,C,H\times r, W\times r)$
  * `r` 就是参数里的 upscale_factor
* 对于实现步幅为 1/r 的子像素卷积很有用, 可以用来降低 channel 个数
* 主要的作用由 
  * Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network by Shi et. al (2016)
  * 被提出

`class torch.nn.PixelUnshuffle(downscale_factor)`  
* 等同于 PixelShuffle 的逆运算
* $C_{out}=C_{in}\times downscale_factor^2$


`class torch.nn.Upsample(size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None)`
* 上采样输入 Tensor, 该层支持 1,2,3D 数据, 对于输入Tensor, 会采样 `[2:]` 以后的维度, 对于 0-d (batch) 和 1-d (channel) 不执行上采用
* 输入大小由 size 或者 scale_factor 来指定, 但是不能同时指定这两个参数 
  * `size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int], optional)`
  * `scale_factor (float or Tuple[float] or Tuple[float, float] or Tuple[float, float, float], optional)`
* `mode` : str, 用于指定上采用的算法, 可选项有 
  * 'nearest'
  * 'linear'
  * 'bilinear'
  * 'bicubic'
  * 'trilinear'

## 3.10. ChannelShuffle - 重排列 Channel

`class torch.nn.ChannelShuffle(groups)`
* 将 tensor  (batch, channel, h, w) reshape 成 (batch, channel/g, g, h,w) 再调转顺序 (batch, g, channel/g, h,w)

# 4. torch.nn.functional

不是直接定义层, 而是把各个网络层的运算抽出来的包  

使用较为基础的运算进行网络定制化的时候需要用到, 和外部的接口的动作有些许不同


## 4.1. Convolution functions



` torch.nn.functional.unfold(input, kernel_size, dilation=1, padding=0, stride=1)`
* 滑窗提取函数, 参照 nn.Unfold
* 该函数本身目前只支持 4-D 输入 only 4-D input tensors (batched image-like tensors) are supported.
* 由于该函数是 view 方式, 因此元素本身会引用相同的地址, 因此在操作前需要克隆

` torch.nn.functional.fold(input, output_size, kernel_size, dilation=1, padding=0, stride=1)`
* 滑窗合并函数, 参照 nn.fold
* Currently, only unbatched (3D) or batched (4D) image-like output tensors are supported.

## 4.2. Non-linear Activations 

直接阅读 functional 部分的激活函数定义会有比较详细的说明

### 4.2.1. F.glu

`torch.nn.functional.glu(input, dim=-1) → Tensor`

在语言模型中目前被大量的使用  (论文: GLU Variants Improve Transformer)

$$GLU(a,b)= a \otimes \sigma(b)$$

这里激活函数的输入 input 本身只有一个, 然后将其 split in half 形成 a,b, 其中的 b 进行标准的激活函数 (在 pytorch 的这里 F.glu 里为 sigmoid), 最终的结果于 a 再进行点乘

在 NAFNet 中, 作者议论了 glu 本身可以被重构为两个 linear transformation layer 的点乘结果
* 此外, 计算量较轻的 channel-attention 也可以被解释为一种 GLU 的变体因为其类似 $$x * \Phi(x)$$

### 4.2.2. F.gelu

 Gaussian Error Linear Units (GELUs)
 基于论文 : Gaussian Error Linear Units (GELUs)

`torch.nn.functional.gelu(input, approximate='none')`
文档中没有指向源码的链接
* approximate 似乎是一个 string

对于 approximate 为 `'none'` 的时候
$$GELU(x) = x* \Phi(x)$$
$\Phi(x)$ 是高斯分布的累计分布函数, 所谓 none 就是不进行近似, 计算量很大

对于 approximate 为 `'tanh'` 的时候
$$GELU(x) =  0.5*x*(1+\tanh(\sqrt{2/\Pi}*(x+0.044615*x^3)))$$
即这是一个 GELU 的近似实现


整理来看, GELU 是一个特殊形式的 GLU
尽管他们的字符相近, 各自全称的单词并不相同, 但结果上却仍然是相似的

### 4.2.3. F.sigmoid

`torch.nn.functional.sigmoid(input) → Tensor`
最经典的激活函数, 对于输入的每一个值执行  
$$sigmoid(x) = \frac{1}{1+\exp(-x)}$$

### 4.2.4. F.batch_norm

`torch.nn.functional.batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-05)`

最经典的归一化: Apply Batch Normalization for each channel across a batch of data.

在整个 mini-batch 上计算均值和标准差, 对于输入为 B C H W 的 3通道 RGB 图像, 均值和标准差的 shape 都为 C.
即在整个 mini-batch 上按照通道进行归一化  

主要用于 CNN 和 FC

### 4.2.5. F.instance_norm

`torch.nn.functional.instance_norm(input, running_mean=None, running_var=None, weight=None, bias=None, use_input_stats=True, momentum=0.1, eps=1e-05)`

Apply Instance Normalization independently for each channel in every data sample within a batch.

之所以叫 instance 即不再是整个 Batch 进行归一化, 而是每一个输入实例本身内部进行归一化.
即 均值和标准差的 shape 为 B C, 在每个通道的每个实例上进行.

在风格迁移等图像生成任务上用的比较多  


### 4.2.6. F.layer_norm

`torch.nn.functional.layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-05)`

Apply Layer Normalization for last certain number of dimensions.

直接对最后几个维度进行归一化, 均值和标准差的 shape 为 (B,1,1,1)
即主要是在 instance 的基础上, 对通道维度也进行归一化

主要用于  RNN 和 FC

### 4.2.7. F.normalize

在指定维度上指定 $L_p$ 标准化, 

`torch.nn.functional.normalize(input, p=2.0, dim=1, eps=1e-12, out=None) -> Tensor`
* 默认针对 dim=1 (从左往右数第二个维度) 执行 p=2.0 (欧式) 标准化
* 实际上是 Tensor 的 norm 方法与 除法的结合
  * 然而 Tensor 的 norm 方法已经被 deprecated, 未来会怎么样还是谜团
  * 
```py
# 源码的定义
def normalize(input: Tensor, p: float = 2.0, dim: int = 1, eps: float = 1e-12, out: Optional[Tensor] = None) -> Tensor:
  # 此处省略 if 语句
        denom = input.norm(p, dim, keepdim=True).clamp_min(eps).expand_as(input)
        return input / denom
```

## 4.3. Vision functions

视觉相关的特殊函数

`torch.nn.functional.pad(input, pad, mode='constant', value=None) → Tensor`
* 填充函数
* 输入的 pad 会按照从对输入 x 后往前的顺序对每一个维度进行, 前后分别指定宽度的 pad
  * 即输入的顺序是按照 x 意思的从后往前
* 参数:
  * input (Tensor) – N-dimensional tensor
  * pad (tuple) – m-elements tuple, where `m/2 ≤ input dimensions` and m is even.
  * mode (str) – 'constant', 'reflect', 'replicate' or 'circular'. Default: 'constant'
  * `value (Optional[float])` – fill value for 'constant' padding. Default: 0


`torch.nn.functional.grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=None)`
* 计算 grid sample
* grid 在这里值得是 flow-field, 通过 input 和 grid 里指定的坐标获取 output
* 当前 4-D 和 5-D(volumetric, 3维空间 体) 数据可以作为 input 被输入



# 5. torch.Tensor 张量类

* `torch.Tensor` 是 pytorch 的张量的类名
* `torch.Tensor` is an alias for the default tensor type (torch.FloatTensor).
* `torch.` 里有许多便捷创建张量的函数
* Tensor 类里面也有许多转换格式的方法
* 几乎所有的类方法都有 torch.* 下的同名方法, 功能一样, 多一个参数是输入 tensor   

## 5.1. torch.Tensor 的格式

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

## 5.2. 类属性

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







## 5.3. 类方法
### 5.3.1. 类型转换

1. .item()   : Returns the value of this tensor as a standard Python number
   * 只在张量中只有一个元素时生效
   * 将该元素作为数字返回
2. .tolist() : Returns the tensor as a (nested) list.
   * 作为多层python原生列表返回,保留层次结构
3. .numpy()  : Returns self tensor as a NumPy ndarray
   * 共享内存, 非拷贝
  
### 5.3.2. view 变形

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

### 5.3.3. transpose

`torch.transpose(input, dim0, dim1) → Tensor`  
* transpose 可以解释为view的一种
* 返回一个 transposed 的 tensor
* The given dimensions dim0 and dim1 are swapped.
* 该方法是共享内存的, 不进行拷贝
  

注意该方法只能交换两个维度  
同 numpy 的 transpose 不同, numpy 的transpose 可以直接交换多个维度  


## 5.4. 创建操作 Creation Ops



### 5.4.1. 统一值 tensor


### 5.4.2. 随机值 random

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


### 5.4.3. like 类方法

需要获取一个不确定维度的 tensor, 即通过另一个 tensor 指定大小
* rand_like
* randint_like
* randn_like


### 5.4.4. torch.from_numpy

`torch.from_numpy` 接受一个 ndarray 并转换成 tensor 没有任何参数  
```py
torch.from_numpy(ndarray) → Tensor
```

### 5.4.5. tensor复制


* `torch.clone(input, *, memory_format=torch.preserve_format) → Tensor`
  * 返回一个完全相同的tensor,新的tensor开辟新的内存，但是仍然留在计算图中。
* `torch.Tensor.detach()`
  * detach() 属于 view 函数
  * 返回一个完全相同的tensor,新的tensor开辟与旧的tensor共享内存, 但会脱离计算图，不会牵扯梯度计算

一般彻底的复制并脱离可以使用  `tensor.clone().detach()`  这也是官方推荐的方法  


### 5.4.6. .new_ 方法

To create a tensor with similar type but different size as another tensor, use tensor.new_* creation ops.  

1. new_tensor
   * new_tensor可以将源张量中的数据复制到目标张量（数据不共享）
   * 提供了更细致的device、dtype和requires_grad属性控制
   * 默认参数下的操作等同于.clone().detach(), 但官方推荐后者
2. new_full
3. new_empty
4. new_ones
5. new_zeros

# 6. torch.amp - Automatic Mixed Precision package

`torch.amp` 提供了 方面的工具用于 mixed precision 混合精度?  

对于一些运算, 例如 linear layers and convolutions, float16 会比 float32 快得多.
amp 包用于在训练, 推理的时候方便的处理类型转换  

被弃用的接口:
* `torch.cuda.amp.autocast` `torch.cuda.amp.GradScaler`
* `torch.cpu.amp.autocast`  `torch.cpu.amp.GradScaler`

## 6.1. Autocasting - torch 接口

`class torch.autocast(device_type, dtype=None, enabled=True, cache_enabled=None)`
* autocast 的实例 会作为 context managers 或者 decorators, 允许特定区域的代码以混合精度来运行  
* 对应区域的 Tensor 可以是任何类型, 此时不应该再调用 Tensor 本身的 float() 或者 half() 接口
* `autocast` 只应该应用在 forward 以及 loss 计算,  不推荐在 backword 的部分应用 autocast


### 6.1.1. Gradient Scaling




## 6.2. Autocast Op Reference - 自动类型转换参考





# 7. torch.autograd - 梯度计算包

## 7.1. Locally disabling gradient computation

与该章节相关的内容还有两个  

一个是 torch 下的接口列表 Locally disabling gradient computation
另一个是介绍 Autograd mechanics 的 
* https://pytorch.org/docs/stable/notes/autograd.html#locally-disable-grad-doc


# 8. Torch Devices

torch.cpu  
torch.cuda

## 8.1. torch.cpu - 虚类实现

## 8.2. torch.cuda - CUDA 计算  

torch.cuda 主要实现了 CUDA 张量类型, 同 CPU 张量的各种接口都一样, 但是是以 GPU 来计算的  

torch.cuda 可以随时导入, 并通过  `is_available()` 来判断设备的 CUDA 设备可用情况  


### 8.2.1. General - 通用接口

主要是对设备的管理而非计算

可用性函数:
* `torch.cuda.is_available()`     : 表示 CUDA 是否可用  
* `torch.cuda.is_initialized()`   : Pytorch CUDA 是否初始化完成, 在交互式终端中直接闻讯该函数会得到 False
* `torch.cuda.init()`             : 手动进行 Pytorch CUDA 初始化
  * 需要手动调用的情况可能是, if you are interacting with PyTorch via its C API
  * 在初始化之前, Python 与 CUDA 功能的链接不会被建立

### 8.2.2. Memory management - CUDA 设备内存管理  

更多的是用来管理监视学习进程的内存情况   


内存监视函数:
* `torch.cuda.memory_allocated(device=None)`  : 获取当前设备的张量 GPU 使用情况, 返回 int of bytes
* `torch.cuda.max_memory_allocated(device=None)` : 获取程序自运行开始后的峰值 GPU 使用量, int of bytes
* `torch.cuda.reset_peak_memory_stats(device=None)` : 清空 峰值
  * 该函数主要与上一个 max 函数配合, 可以同来测量每次学习迭代的峰值 GPU 使用情况
* `torch.cuda.reset_max_memory_allocated(device=None)` : 重设 max 的计算起点
  * 该函数目前是 reset_peak 的封装, 本质上是一样的


# 9. torch.linalg - pytorch 的线性代数子库

`torch.linalg.*`

## 9.1. Matrix Properties

* `vector_norm`          Computes a vector norm.
  * `torch.linalg.vector_norm(x, ord=2, dim=None, keepdim=False, *, dtype=None, out=None)`
  * 默认是 2 即计算一个向量在欧氏空间中的距离
* `matrix_norm`          Computes a matrix norm.
  * `torch.linalg.matrix_norm(A, ord='fro', dim=(-2, -1), keepdim=False, *, dtype=None, out=None)`
* `norm`                 Computes a vector or matrix norm.  
  * `torch.linalg.norm(A, ord=None, dim=None, keepdim=False, *, out=None, dtype=None)`  
  * 属于上面两个函数的结合体, 根据输入的 shape 和 dim 来判断要执行的计算


## 9.2. Decompositions


# 10. torch.profiler

允许在训练和推理期间收集性能指标的工具, 了解模型哪些运算符昂贵, 检查输入形状和追踪

类似的量化性能 API 在 torch.autograd 中也有, 但是已被 deprecated

## 10.1. class torch.profiler.profile

核心的 性能指标收集 类, 常通过 context manager 格式来使用



## 10.2. class torch.profiler._KinetoProfile

核心类 profile 的父类, 定义了输出性能结果的方法  





# 11. torch.onnx

将 torch 模型转换成 ONNX graph 的 API

转换后的模型支持所有能够运行 ONNX 的 runtime.  

在 Pytorch 中, 定义了两种转化为 ONNX 模型的偏好

## 11.1. TorchDynamo-based ONNX Exporter

于 Torch 2.0 推出的最新的转换方法, 目前还是 Beta?
* 在转换中与 Python's frame evaluation API 链接, 动态的将字节码重写为 FX 图表
* 对FX 进行完善, 最终才将其转化为 ONNX
* 使用字节码分析捕获 FX graph, 能够保留模型的 动态特性, 而不是传统的静态跟踪技术

## 11.2. TorchScript-based ONNX Exporter

于 Pytorch 1.2.0 推出的 ONNX 转换器  
* 通过使用 TorchScript 来跟踪 模型, 并捕获静态计算图
* 因此, 该传统方法的限制为:
  * 不会记录任何控制流, 例如 if 语句 或者 loop 循环
  * 不会处理 training 和 eval 模型之间的细微差距 (例如 dropoff?)
  * 没有真正意义上处理动态输入的能力 Does not truly handle dynamic inputs

### 11.2.1. API of TorchScript-based ONNX Exporter

```py
torch.onnx.export(model, args, f, export_params=True, verbose=False, training=<TrainingMode.EVAL: 0>, input_names=None, output_names=None, operator_export_type=<OperatorExportTypes.ONNX: 0>, opset_version=None, do_constant_folding=True, dynamic_axes=None, keep_initializers_as_inputs=None, custom_opsets=None, export_modules_as_functions=False, autograd_inlining=True)


# args 是一个复杂的参数, 代表了模型的典型输入
args (tuple or torch.Tensor)

# 1. args can be structured either as:
args = (x, y, z)
# 对于模型 forward 需要多个输入内容的情况, 所有非 Tensor 的输入都会被硬编码

# 2. A TENSOR: 相当于只有一个输入的时候, tuple 的解包
args = torch.Tensor([1])

# 3. A TUPLE OF ARGUMENTS ENDING WITH A DICTIONARY OF NAMED ARGUMENTS:
args = ( x,{"y": input_y, "z": input_z} )
# 主要用于模型需要 命名的参数的情况, 这种时候可以给模型 forward 定义输入默认值, 如果在字典中不给 named arg 提供输入值的话, 默认输入值是 None
```

# 12. torch.optim

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

## 12.1. 预定义优化 Algorithm

各种参数优化方法, (参数各不相同, 一般都有 lr )
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


### 12.1.1. optim 通用参数

* `lr` (float, Tensor, optional), 学习速率
* `betas` (Tuple[float, float], optional), 用于计算 梯度和梯度平方的running averages 的系数
* `verbose` : 被弃用的参数, Deprecated since version 2.2: verbose is deprecated. Please use get_last_lr() to access the learning rate.

## 12.2. Adjust learning rate - lr_scheduler 动态 Learn Rate

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
# 运行完一个完整的 epoch 《后》 更新学习速率 是 pytorch 1.1 版本以后的约定
for epoch in range(100):
    for input, target in dataset:
      forward...
      loss=...
      loss.backward()
      optimizer.step()
  scheduler.step()
```

### 12.2.1. Optim Scheduler 通用参数和方法

scheduler 的通用参数:
* `gamma` : float, 乘法参数, 当前学习速率直接乘以该值
* `last_epoch` : (int), 主要用于学习再开, 设置最新的学习状态

scheduler 的通用成员方法:
* print_lr(is_verbose, group, lr, epoch=None)  : 打印当前的学习速率
* get_last_lr() : 根据输入的参数计算最终的学习率

### 12.2.2. Optim Class 

scheduler 的种类: `torch.optim.lr_scheduler.*`

`StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1, verbose='deprecated')`
* 每经过固定 step_size, 对 lr 乘以 gamma 系数

`MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1, verbose='deprecated')`
* `milestones` (list) – List of epoch indices. Must be increasing.
* 同 Step, 只不过 step 变为数组, 每达到 milestones 的某一个元素的时候, 应用 gamma 系数


`LinearLR(optimizer, start_factor=0.3333333333333333, end_factor=1.0, total_iters=5, last_epoch=-1, verbose='deprecated')`
* 线性调整 lr 的乘法系数, 直到最终的 total_iters. lr 本身的变化不是线性的 
* `start_factor` `end_factor`, lr 乘法系数的数值, 从一开始的 0.5 (减半) 到最终的 1.0 (lr保持不变)

`ExponentialLR(optimizer, gamma, last_epoch=-1, verbose='deprecated')`
* ExponentialLR : 对比 StepLR, 代表每一个 epoch 都进行学习速率衰减  
* 


`CosineAnnealingLR(optimizer, T_max, eta_min=0.0, last_epoch=-1, verbose='deprecated')`
* 余弦退火 (cosine annealing) 来调整学习速率  
* 学习速率是一个余弦曲线, 但不清楚具体在应用的时候需不需要使用超过一个周期的 数值
* `T_max` (int), 通常是 epoch_final
* `eta_min` (float), 最终的最小学习速率


`ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose='deprecated')`
* 模型在学习停滞 stagnates 的时候通常将 lr 降低 2-10 倍, 会从中受益, 根据 评价 metric, 如果 metric 停止改进, 则降低学习速度
* `mode`  : 设置目标 matric 的模式, 最小化还是最大化, loss通常是需要 min 最小化的
* `factor`: (float), 调整 lr 的系数, 乘法应用, 相当于其他 Scheduler 的 Gamma参数
* `patience` : (int), 所允许的没有改进的 epoch 数量, 在之后 lr 将会降低
* `threshold` : (float), 仅关注 阈值之上的改进
* `cooldown` :(int), lr 降低后, 回复 normal operation 所需要的时间, 没太懂
* `min_lr ` : (float or list), 学习率的下限
* 方法:
  * `step(metrics, epoch=None)`  : 传入 metrics






### 12.2.3. 有序调整


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


## 12.3. 定义自己的 optim

### 12.3.1. Optimizer 基类

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

### 12.3.2. optimization step

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

### 12.3.3. per-parameter options

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



#  13. Autograd mechanics


## 13.1. Locally disabling gradient computation

局部关闭梯度在 pytorch 中事实上有多种模式, 其中有些许的区别

禁用整个块的梯度可以用上下文管理器例如 no_grad 或者 inference 模式, 如果要细微的从梯度计算中排除某一个 子图 则可以设置 requires_grad

除此之外  nn.Module.eval() 虽然称为 评估模式, 但是并不是用来禁用梯度的, 这点常常会产生混淆



### 13.1.1. Setting requires_grad

手动设置某一个 Tensor 的梯度情况, 这是一个 Tensor 的 flag, 但是只有设置在 nn.Parameter 中的 Tensor 才会自动被设置为 requires_grad = True


在计算梯度的时候, 只有 require_grad = True 的张量才会将梯度累计到 .grad 字段中
* 要注意, 在一个计算图中, 这个 flag 只针对 叶子节点的张量有意义
* 非叶子节点的梯度需求是受 叶子节点影响的, 如果叶子节点需要梯度, 那么路径上的非叶子节点也必须有梯度, 因此 非叶子节点的 True 会被自动设置 


手动设置 require_grad 主要用于 在模型微调的时候冻结 pretrain 模型的参数  
这是一种最常见的模式, 因此可以使用 `nn.Module.requires_grad_()` 在模块级别设置 requires_grad


### 13.1.2. Grad Modes

除了 rquire_grad 的 flag 之外, 还有三种模式会影响 autograd 的计算方式
* 默认 grad
* no-grad
* inference
这三种模式的应用层级高于 requires_grad


grad 模式: 默认模式, 只有在该模式下 require_grad 才会生效, 之所以成为 grad 模式只是为了与其他两种没有梯度的模式做区别  


### 13.1.3. No-grad Mode

no-grad 从整体上彻底停止了一切梯度  

作为上下文的时候很方便

no-grad 应该被用于修改模型参数的时候, 例如自定义 optimizer 的时候对 parameter 进行更新

此外, 模型在初始化的时候对参数进行设置 也是在 no-grad 模式下的 


### 13.1.4. Inference Mode

比 no-grad 更进一步的优化, 推理速度更快  

区别在于 在 inference 模式下创建的 Tensor (例如推理结果), 在 inference 模式推出了后也不能作为 autograd 记录的计算  

如果推理的结果与 autograd 在推出 inference 模式后也 没有任何的交互的话, 可以使用 inference 模式



### 13.1.5. Evaluation Mode (nn.Module.eval())


该模式会出现在该章节仅仅是因为特别容易混淆  

module.eval() 和 module.train() 所影响的完全是模型的行为本身, 因为某些模型本身就对于推理和训练有不同的行为
* 最典型的就是 dropout
* 以及 BatchNorm2d : avoid updating your BatchNorm running statistics on validation data.
* 不管如何, 在 validation 的时候使用 module.eval() 总是正确的 

