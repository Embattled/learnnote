# 1. Pytorch

## 1.1. install 

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
```

# 2. 建模
## 2.1. Model Authoring

A `Module` is the basic unit of composition in PyTorch.
  1. A constructor, which prepares the module for invocation.
  2. A set of Parameters and sub-Modules. These are initialized by the constructor and can be used by the module during invocation.
  3. A forward function. This is the code that is run when the module is invoked.


```py

# 创建一个被作为成员的子模型
class MyDecisionGate(torch.nn.Module):
    # 这里并没有定义构造器
    def forward(self, x):
        if x.sum() > 0:
            return x
        else:
            return -x


# 创建一个主模型, 从 torch.nn.Module 继承
class MyCell(torch.nn.Module):

    # 构造函数 网络的成员层都定义在这里面
    def __init__(self):
        # 定义网络构造器的必须的一条语句
        super(MyCell, self).__init__()

        # 定义一个自定义的网络层
        self.dg = MyDecisionGate()

        # 定义一个全连接层
        self.linear = torch.nn.Linear(4, 4)

    # 前向传播函数
    def forward(self, x, h):

        # 输入4X4 全连接后得到输出
        new_h = torch.tanh(self.linear(x) + h)
        return new_h, new_h

# 实例化模型
my_cell = MyCell()
x = torch.rand(3, 4)
h = torch.rand(3, 4)

# 打印网络成分
print(my_cell)
""" 
MyCell(
  (dg): MyDecisionGate()
  (linear): Linear(in_features=4, out_features=4, bias=True)
)
"""


# 输如网络使用 my_cell(输入值)  即可
print(my_cell(x, h))

```

流程总结:
1. 创建一个类 , 继承于 torch.nn.Module
2. 定义构造函数, 不需要做什么, 只需要: just calls the constructor for super.
3. 定义 `forward` 函数, 


# 3. API

至2020年11月  
torch 最新版是 1.7  
中文文档更新到1.4  

```py
# 开始使用
import torch
print(torch.__version__)

```

## 3.1. torch

作为最基础的包, torch 包括了tensors的数据格式以及对应的数学操作. 还提供了许多工具来高效的序列化 tensors以及其他任意数据格式   
该包有 CUDA 对应  

## 3.2. torch.nn

These are the basic building block for graphs  
用于定义网络中的各个层  

### 3.2.1. Module

* `class torch.nn.Module  `
  * 所有神经网络模块的基类 Base class for all neural network modules.
  * 您的模型也应该继承此类
  * 模块也可以包含其他模块，从而可以将它们嵌套在树形结构中。 您可以将子模块分配为常规属性


基础方法:
* cpu()                 : Moves all model parameters and buffers to the CPU.
* cuda(device: Union[int, torch.device, None] = None)   :Moves all model parameters and buffers to the GPU
* forward(*input: Any)  : Defines the computation performed at every call. Should be overridden by all subclasses.

模式
* training mode.
* evaluation mode.  `net.eval()`

```py

import torch.nn as nn
import torch.nn.functional as F

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

# 直接打印网络的信息
print(net)

# 进入 evaluation mode  等同于 
net.train(False)
net.eval()

# 进入 training mode
net.train()
```

### 3.2.2. Convolution Layers 卷积层 

用于在神经网络中定义卷积层  

* nn.Conv1d
  * Applies a 1D convolution over an input signal composed of several input planes.
* nn.Conv2d
  * Applies a 2D convolution over an input signal composed of several input planes.
* nn.Conv3d
  * Applies a 3D convolution over an input signal composed of several input planes.

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
### 3.2.3. Normalization Layers 归一化层

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
### 3.2.4. Pooling layers 池化层

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



### 3.2.5. Linear Layers  线性层

用于构筑网络的全连接层  

* nn.Identity
* nn.Linear
* nn.Bilinear

```py
class torch.nn.Linear(in_features, out_features, bias=True)

# in_features   :每个输入样本的大小
# out_features  :每个输出样本的大小
# bias          :如果设置为False，则该图层将不会学习加法偏差。 默认值：True

# 定义网络的全连接层

    # Fully connected layers
    self.fc1 = nn.Linear(800, 500)
    #self.bn3 = nn.BatchNorm1d(500)
    self.fc2 = nn.Linear(500, num_class)

```
### 3.2.6. 非线性激活函数

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


### 3.2.7. Loss Function 损失函数

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


## 3.3. torch.nn.functional

不是直接定义层, 而是把各个网络层的运算抽出来的包  
因为pooling运算没有参数, 所以定义在了这里  

## 3.4. torch.utils.data

* pytorch 核心的数据装载工具,  `torch.utils.data.DataLoader`  
* It represents a Python iterable over a dataset

### 3.4.1. torch.utils.data.DataLoader
Pytorch的核心数据读取器     :  `torch.utils.data.DataLoader`   是一个可迭代的数据装载器  包括了功能
* map-style and iterable-style datasets,
* customizing data loading order,
* automatic batching,
* single- and multi-process data loading,
* automatic memory pinning.

它可以接受两种类型的数据集
* map-style datasets,
  * implements the `__getitem__()` and `__len__()` protocols
  * represents a map from (possibly non-integral) indices/keys to data samples.
  *  when accessed with `dataset[idx]`, could read the idx-th image and its corresponding label from a folder on the disk.
* iterable-style datasets.
  * Is a subclass of IterableDataset that implements the `__iter__()` protocol
  * particularly suitable for cases where random reads are expensive or even improbable, and where the batch size depends on the fetched data.
  * when called `iter(dataset)`, could return a stream of data reading from a database, a remote server, or even logs generated in real time.



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

# 通过装载器获得一个 可迭代数据库  使用 iter 
iter_data = iter(trainloader)
#iter_data = iter(testloader)

# 对可迭代数据库使用 next 得到数据和label
images, labels = next(iter_data)
print(images.size())
# torch.Size([100, 1, 28, 28])   100 个数据 每个数据 1 channel , 高和宽都是28

print(labels)
# 打印labels
```

### 3.4.2. torch.utils.data.Dataset
表示数据集的抽象类          :  `torch.utils.data.Dataset`  

## 3.5. torch.optim

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

### 3.5.1. per-parameter options

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
### 3.5.2. optimization step

重点: 所有 optimizers 必须实现 step 方法, 用来更新要优化的参数  

1. optimizer.step()
   大部分优化器都实现了的简单的版本, 在 backward() 方法之后调用

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
### 3.5.3. Algorithm

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
* **SGD**

### 3.5.4. 动态 Learn Rate

`torch.optim.lr_scheduler` 提供了一些方法用来根据 epoch 或者其他计算来调整学习速率

```py
# 使用方法: 一般学习速度的调整应该放在 optimizer 更新之后
scheduler = ...
for epoch in range(100):
    train(...)
    validate(...)
    scheduler.step()
```



## 3.6. torchivision

包含用于计算机视觉的流行数据集，模型架构和常见图像转换  

### 3.6.1. torchvision.datasets

主流的数据集：  

    MNIST
    fashion MNIST
    KMNIST
    EMNIST
    QMNIST
    FakeData
    coco
    LSUN
    ImageFolder
    DatasetFolder
    ImageNet
    CIFAR
    STL10
    SVHN
    PhotoTour
    SBU
    Flickr
    VOC
    城市景观
    SBD
    USPS
    Kinetics-400
    HMDB51
    UCF101


### 3.6.2. torchvision.transforms

1. transforms包含了一些常用的图像变换，这些变换能够用 `Compose` 串联组合起来  
2. `torchvision.transforms.functional` 模块供了一些更加精细的变换，用于搭建复杂的变换流水线(例如分割任务）  

分类: 
    Scriptable transforms
    Compositions of transforms
    Transforms on PIL Image and torch.*Tensor
    Transforms on PIL Image only
    Transforms on torch.*Tensor only
    Conversion Transforms
    Generic Transforms
    Functional Transforms

大部分的 Transformation 接受 PIL Image, Tensor Image 和 batch of Tensor Images 作为输入  
* Tensor Image is a tensor with (C, H, W) shape, where C is a number of channels
* Batch of Tensor Images is a tensor of (B, C, H, W) shape, where B is a number of images in the batch. 


#### 3.6.2.1. Scrpit and Compositions


```py
# In order to script the transformations, please use torch.nn.Sequential instead of Compose.
transforms = torch.nn.Sequential(
    transforms.CenterCrop(10),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
)
# Make sure to use only scriptable transformations, i.e. that work with torch.Tensor, does not require lambda functions or PIL.Image.
scripted_transforms = torch.jit.script(transforms)



# 用 `Compose` 串联组合变换

class torchvision.transforms.Compose(transforms)
# transforms    :  (Transform对象的list）  一系列需要进行组合的变换。

transforms.Compose( [  transforms.CenterCrop(10), transforms.ToTensor(),  ])



```

#### 3.6.2.2. Transforms on PIL Image and torch.*Tensor 通用变换函数

* class torchvision.transforms.CenterCrop(size)
* class torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
* class torchvision.transforms.FiveCrop(size)
* class torchvision.transforms.Grayscale(num_output_channels=1)
* class torchvision.transforms.Pad(padding, fill=0, padding_mode='constant')
* class torchvision.transforms.RandomAffine(degrees, translate=None, scale=None, shear=None, resample=0, fillcolor=0)
* class torchvision.transforms.RandomApply(transforms, p=0.5)
* class torchvision.transforms.RandomCrop(size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant')
* class torchvision.transforms.RandomGrayscale(p=0.1)
* class torchvision.transforms.RandomHorizontalFlip(p=0.5)
* class torchvision.transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=2, fill=0)
* class torchvision.transforms.RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2)
* class torchvision.transforms.RandomRotation(degrees, resample=False, expand=False, center=None, fill=None)
* class torchvision.transforms.RandomSizedCrop(*args, **kwargs)
* class torchvision.transforms.Resize(size, interpolation=2)
* class torchvision.transforms.Scale(*args, **kwargs)
* class torchvision.transforms.TenCrop(size, vertical_flip=False)
* class torchvision.transforms.GaussianBlur(kernel_size, sigma=(0.1, 2.0))
  

#### 3.6.2.3. Transforms on only 特定函数


**Transforms on PIL Image Only**
* class torchvision.transforms.RandomChoice(transforms)
* class torchvision.transforms.RandomOrder(transforms)

**Transforms on torch.*Tensor only**
* class torchvision.transforms.LinearTransformation(transformation_matrix, mean_vector)
* class torchvision.transforms.Normalize(mean, std, inplace=False)
* class torchvision.transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
* class torchvision.transforms.ConvertImageDtype(dtype: torch.dtype)


#### 3.6.2.4. Conversion and Generic Transforms 格式转换 和 通用变化

* class torchvision.transforms.ToPILImage(mode=None)
* class torchvision.transforms.ToTensor


* class torchvision.transforms.Lambda(lambd)
    * 将用户定义的 lambda 作为变换函数




#### 3.6.2.5. Functional Transforms 用于更精细化的变换

`import torchvision.transforms.functional as TF`  函数名称省略前缀  


# 4. TorchScript


对于一个从Pytorch创建的一个可优化和串行的模型, 使其可以运行在其他非Python的平台上  
  * From a pure Python program to a TorchScript program that can be run independently from Python.
  * An intermediate representation of a PyTorch model (subclass of nn.Module) that can then be run in a high-performance environment such as C++.

TorchScript provides tools to capture the definition of your model, even in light of the flexible and dynamic nature of PyTorch.  

1. TorchScript 有自己的解释器, 这个解释器类似于受限制的 Python 解释器, 该解释器不获取全局解释器锁，因此可以在同一实例上同时处理许多请求。
2. TorchScript 可以是我们把整个模型保存到磁盘上, 并在另一个运行环境中载入, 例如非Python的运行环境
3. TorchScript 可以进行编译器优化并以此获得更高的执行效率
4. TorchScript 可以允许与许多后端设备运行接口, 这些运行环境往往需要比单独的操作器更广泛的程序视野.


## 4.1. Tracing Modules

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

## 4.2. Convert Modules

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

## 4.3. Mixing Scripting and Tracing

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

## 4.4. Saving and Loading models

save and load TorchScript modules  
这种形式的存储 包括了代码,参数,性质还有Debug信息

```py

traced.save('wrapped_rnn.zip')

loaded = torch.jit.load('wrapped_rnn.zip')

print(loaded)
print(loaded.code)

```

## 4.5. API torch.jit

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

# 5. Pytorch C++ API

## 5.1. ATen



# 6. 例程


## 6.1. MNIST LeNet 例程
### 6.1.1. Network structure
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


### 6.1.2. dataset


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
### 6.1.3. iteration

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



### 6.1.4. evaluate

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

## 6.2. MINST GAN 例程

### 6.2.1. dataset

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

### 6.2.2. 网络

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

### 6.2.3. G 训练

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

### 6.2.4. D 训练
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

### 6.2.5. iteration

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

## 6.3. MINST USPS adversarial examples

### 6.3.1. dataset

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

### 6.3.2. general train and evaluate

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