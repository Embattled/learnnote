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
* 形容起来比较复杂, 提供两个相同 维度数的 Tensor input 和 index
  * `input.dim == index.dim` 
  * index 和 input 不会互相 broadcast, 且需要在每个维度上 `index.size(d) <= input.size(d)`
  * 输出的 shape 和 index 相同
* 根据指定的 dim, 参照 index 提供的数据在 input 的 dim 维度重新选择数据输出
  * 因此属于 Indexing 函数

```py
torch.gather(t, 1, torch.tensor([[0, 0], [1, 0]]))
tensor([[ 1,  1],
        [ 4,  3]])
```


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

### Generator 相关

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


### Random Sampling - 默认使用全局 rng

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

## 2.4. Locally disabling gradient computation - 局部禁用梯度计算  

梯度上下文管理有三个接口, 用于在局部对梯度计算进行启用, 停用:
* 在 `torch.autograd` 中有更详细的描述  
* 这些接口大部分都属于 Context-manager, 所以该种类接口一般情况下需要与 `with` 语法一起使用  
  * 该 Context-manager 影响的是线程 thread local, 即整个线程上的 torch 推论都会受到影响  


梯度管理方法 `torch.*`
* no_grad  :          Context-manager that disables gradient calculation.
* enable_grad :       Context-manager that enables gradient calculation

* inference_mode   :  Context-manager that enables or disables inference mode

* is_grad_enabled  :  Returns True if grad mode is currently enabled.
* is_inference_mode_enabled :  	Returns True if inference mode is currently enabled.

可以作为函数使用的接口:
* `class torch.set_grad_enabled(mode)` :  Context-manager that sets gradient calculation on or off.
  * Context-manager that sets gradient calculation on or off.
  * 这个接口同其他几个不同, 还可以作为函数使用用来全局的关闭梯度计算.  
  * 参数 : mode (bool)

## 2.5. Math operations

提供对 tensor 的各种数学操作

### 2.5.1. Pointwise Ops 元素为单位的操作

* 三角函数
  * torch.sin

### 2.5.2. Reduction Ops 元素之间的操作(降维)

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

#### 2.5.2.1. 极值操作

* max/min : 返回最大值, 默认对全元素进行, 可以输入单个整数 dim 来对某一维度进行运算, 此时会返回两个值, 第二个返回值是索引位置
* argmax/argmin : 返回最大值的索引, 默认对全元素进行操作, 可以输入单个整数 dim 来对某一维度进行运算, 等同于 max/min 输入 dim 的第二个返回值
* amax/amin : 返回最大值, 专门用来对指定维度进行运算, dim 是必须参数且可以是 int or tuple, 即可以对多个维度进行运算, 不会返回索引



### 2.5.3. Comparison Ops

专门用来比较的函数 

获取指定维度的 k 个最大值, 同时还能获得对应的 索引, 用在分类任务上  
`torch.topk(input, k, dim=None, largest=True, sorted=True, *, out=None) -> (Tensor, LongTensor)`  
* dim, int 只能指定单独维度, 默认会对最后一个维度进行操作
* largest: bool, true 则选择最大的top, 否则选择最小的 top
* sorted: bool, 是否按对应的顺序返回这k个值


### 2.5.4. Spectral Ops - 光谱函数 在频域上操作信号

### 2.5.5. Other Operations - 无法分类的其他函数


cum 累计系类:
* `torch.cumsum(input, dim, *, dtype=None, out=None) → Tensor`    : 将输入沿着某一维度 累加

broadcast 家族:
* `torch.broadcast_to(input, shape) → Tensor` : 手动 broadcast 
* `torch.broadcast_tensors(*tensors) → List of Tensors[source]` : 同时 broadcast 一整组张量



`torch.searchsorted(sorted_sequence, values, *, out_int32=False, right=False, side='left', out=None, sorter=None) → Tensor`
* 寻找有序插入位置
* 常被用来实现基于 torch 的线性插值, 因为 torch 似乎没有直接提供类似于 `numpy.interp` 用于非自然数索引的插值方法
* `right=False` 返回的插入位置是达成有序的最左边, 即会插入到相同值的元素的前面, 如果至 True, 则代表右边

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
#### 3.1.1.3. 残差结构


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
  * nn.AdaptiveMaxPool1d
  * nn.AdaptiveMaxPool2d
  * nn.AdaptiveMaxPool3d
  * nn.AdaptiveAvgPool1d
  * nn.AdaptiveAvgPool2d
  * nn.AdaptiveAvgPool3d

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


## Convolution functions

## Non-linear activation functions





### F.normalize

在指定维度上指定 $L_p$ 标准化, 

`torch.nn.functional.normalize(input, p=2.0, dim=1, eps=1e-12, out=None) -> Tensor`
* 默认针对 dim=1 (从左往右数第二个维度) 执行 p=2.0 (欧式) 标准化
* 实际上是 Tensor 的 norm 方法与 除法的结合
  * 然而 Tensor 的 norm 方法已经被 deprecated, 未来会怎么样还是谜团
  * 
```py
        denom = input.norm(p, dim, keepdim=True).clamp_min(eps).expand_as(input)
        return input / denom
```




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


# 6. torch.autograd - 梯度计算包

# 7. Torch Devices

torch.cpu  
torch.cuda

## 7.1. torch.cpu - 虚类实现

## 7.2. torch.cuda - CUDA 计算  

torch.cuda 主要实现了 CUDA 张量类型, 同 CPU 张量的各种接口都一样, 但是是以 GPU 来计算的  

torch.cuda 可以随时导入, 并通过  `is_available()` 来判断设备的 CUDA 设备可用情况  


### 7.2.1. General - 通用接口

主要是对设备的管理而非计算

可用性函数:
* `torch.cuda.is_available()`     : 表示 CUDA 是否可用  
* `torch.cuda.is_initialized()`   : Pytorch CUDA 是否初始化完成, 在交互式终端中直接闻讯该函数会得到 False
* `torch.cuda.init()`             : 手动进行 Pytorch CUDA 初始化
  * 需要手动调用的情况可能是, if you are interacting with PyTorch via its C API
  * 在初始化之前, Python 与 CUDA 功能的链接不会被建立

### 7.2.2. Memory management - CUDA 设备内存管理  

更多的是用来管理监视学习进程的内存情况   


内存监视函数:
* `torch.cuda.memory_allocated(device=None)`  : 获取当前设备的张量 GPU 使用情况, 返回 int of bytes
* `torch.cuda.max_memory_allocated(device=None)` : 获取程序自运行开始后的峰值 GPU 使用量, int of bytes
* `torch.cuda.reset_peak_memory_stats(device=None)` : 清空 峰值
  * 该函数主要与上一个 max 函数配合, 可以同来测量每次学习迭代的峰值 GPU 使用情况
* `torch.cuda.reset_max_memory_allocated(device=None)` : 重设 max 的计算起点
  * 该函数目前是 reset_peak 的封装, 本质上是一样的


# 8. torch.linalg - pytorch 的线性代数子库

`torch.linalg.*`

## Matrix Properties

* `vector_norm`          Computes a vector norm.
  * `torch.linalg.vector_norm(x, ord=2, dim=None, keepdim=False, *, dtype=None, out=None)`
  * 默认是 2 即计算一个向量在欧氏空间中的距离
* `matrix_norm`          Computes a matrix norm.
  * `torch.linalg.matrix_norm(A, ord='fro', dim=(-2, -1), keepdim=False, *, dtype=None, out=None)`
* `norm`                 Computes a vector or matrix norm.  
  * `torch.linalg.norm(A, ord=None, dim=None, keepdim=False, *, out=None, dtype=None)`  
  * 属于上面两个函数的结合体, 根据输入的 shape 和 dim 来判断要执行的计算


## Decompositions



# 9. torch.onnx

将 torch 模型转换成 ONNX graph 的 API

转换后的模型支持所有能够运行 ONNX 的 runtime.  

在 Pytorch 中, 定义了两种转化为 ONNX 模型的偏好

## 9.1. TorchDynamo-based ONNX Exporter

于 Torch 2.0 推出的最新的转换方法, 目前还是 Beta?
* 在转换中与 Python's frame evaluation API 链接, 动态的将字节码重写为 FX 图表
* 对FX 进行完善, 最终才将其转化为 ONNX
* 使用字节码分析捕获 FX graph, 能够保留模型的 动态特性, 而不是传统的静态跟踪技术

## 9.2. TorchScript-based ONNX Exporter

于 Pytorch 1.2.0 推出的 ONNX 转换器  
* 通过使用 TorchScript 来跟踪 模型, 并捕获静态计算图
* 因此, 该传统方法的限制为:
  * 不会记录任何控制流, 例如 if 语句 或者 loop 循环
  * 不会处理 training 和 eval 模型之间的细微差距 (例如 dropoff?)
  * 没有真正意义上处理动态输入的能力 Does not truly handle dynamic inputs

### 9.2.1. API of TorchScript-based ONNX Exporter

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

# 10. torch.optim

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

## 10.1. 预定义 Algorithm

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

## 10.2. torch.optim.lr_scheduler - 动态 Learn Rate

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


### 10.2.1. 有序调整


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


## 10.3. 定义自己的 optim

### 10.3.1. Optimizer 基类

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

### 10.3.2. optimization step

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

### 10.3.3. per-parameter options

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



