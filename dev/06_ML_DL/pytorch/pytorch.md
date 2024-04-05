- [1. Pytorch Documentation](#1-pytorch-documentation)
  - [1.1. install](#11-install)
  - [1.2. verification](#12-verification)
  - [1.3. API](#13-api)
- [4. Tensor views](#4-tensor-views)
- [9. torch.jit - TorchScript](#9-torchjit---torchscript)
  - [9.1. Introduction to TorchScript](#91-introduction-to-torchscript)
    - [9.1.1. Tracing Modules](#911-tracing-modules)
    - [9.1.2. Using Scripting to Convert Modules](#912-using-scripting-to-convert-modules)
    - [9.1.3. Mixing Scripting and Tracing](#913-mixing-scripting-and-tracing)
    - [9.1.4. Saving and Loading models](#914-saving-and-loading-models)
  - [9.2. API torch.jit](#92-api-torchjit)
- [13. 例程](#13-例程)
  - [13.1. MNIST LeNet 例程](#131-mnist-lenet-例程)
    - [13.1.1. Network structure](#1311-network-structure)
    - [13.1.2. dataset](#1312-dataset)
    - [13.1.3. iteration](#1313-iteration)
    - [13.1.4. evaluate](#1314-evaluate)
  - [13.2. MINST GAN 例程](#132-minst-gan-例程)
    - [13.2.1. dataset](#1321-dataset)
    - [13.2.2. 网络](#1322-网络)
    - [13.2.3. G 训练](#1323-g-训练)
    - [13.2.4. D 训练](#1324-d-训练)
    - [13.2.5. iteration](#1325-iteration)
  - [13.3. MINST USPS adversarial examples](#133-minst-usps-adversarial-examples)
    - [13.3.1. dataset](#1331-dataset)
    - [13.3.2. general train and evaluate](#1332-general-train-and-evaluate)

# 1. Pytorch Documentation


https://pytorch.org/docs/stable/index.html

Pytorch 是一个优化后的张量库, 用于 GPU和CPU进行深度学习  

在官方文档中 Pytorch 文档分为了几类
* Community
* Developer Notes
* Language Bindings
* Python API

主要接触的最多的还是 Python API

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

至2023年8月  
torch 最新版是 2.0


```py
# 开始使用
import torch
print(torch.__version__)

```




# 4. Tensor views

View 是一个数据的映射共享, 可以避免多余的数据拷贝  







# 9. torch.jit - TorchScript


对于一个从Pytorch创建的一个可优化和串行的模型, 使其可以运行在其他非Python的平台上  
  * From a pure Python program to a TorchScript program that can be run independently from Python.
  * An intermediate representation of a PyTorch model (subclass of nn.Module) that can then be run in a high-performance environment such as C++.

TorchScript provides tools to capture the definition of your model, even in light of the flexible and dynamic nature of PyTorch.  

1. TorchScript 有自己的解释器, 这个解释器类似于受限制的 Python 解释器, 该解释器不获取全局解释器锁，因此可以在同一实例上同时处理许多请求
2. TorchScript 可以是我们把整个模型保存到磁盘上, 并在另一个运行环境中载入, 例如非Python的运行环境
3. TorchScript 可以进行编译器优化并以此获得更高的执行效率
4. TorchScript 可以允许与许多后端设备运行接口, 这些运行环境往往需要比单独的操作器更广泛的程序视野.

TorchScript 是 Pytorch model (nn.Module 的子类) 的中间表示形式, 可以在 C++ 等高性能环境中运行

## 9.1. Introduction to TorchScript

TorchScript 的全局教学
https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html


要点:
* `torch.jit.trace`
* `torch.jit.script`

### 9.1.1. Tracing Modules

首先, Pytorch 本身是使用了 autograd 思想, 在计算的过程中动态的生成计算图, 这样可以不必再一开始就为整个网络构造显式的定义导数  

那么 TorchScript 自然也要兼容该方法 , 提供了能够捕获模型定义的工具, 首先就是 `tracing`

Trace:
1. invoked the Module
2. recorded the operations that occured when the Module was run
3. created an instance of torch.jit.ScriptModule (of which TracedModule is an instance)
   根据输入模型的运算流 创建出对应的 TorchScript 模型对象

简要的说, 创建了一个解析后的称为 `torch.jit.ScriptModule` 的模型, 该模型也可以和原本的 nn.Module 一样 forward  
可以通过 `ScriptModule.graph` 来检查生成的计算图, 但是没什么可读性  
可以通过 `ScriptModule.code` 来检查从计算图中还原出来的  Python-syntax 的计算流程

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

### 9.1.2. Using Scripting to Convert Modules

* 对于一个带有控制流的子模型 (带有 if 的 Module), 直接使用 Trace 不能正确的捕捉整个程序流程  
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

# 这里使用 script 方法
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

### 9.1.3. Mixing Scripting and Tracing

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

### 9.1.4. Saving and Loading models

save and load TorchScript modules  
这种形式的存储 包括了代码,参数,性质还有Debug信息

```py

traced.save('wrapped_rnn.zip')

loaded = torch.jit.load('wrapped_rnn.zip')

print(loaded)
print(loaded.code)

```

## 9.2. API torch.jit

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




# 13. 例程


## 13.1. MNIST LeNet 例程
### 13.1.1. Network structure
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


### 13.1.2. dataset


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
### 13.1.3. iteration

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



### 13.1.4. evaluate

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

## 13.2. MINST GAN 例程

### 13.2.1. dataset

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

### 13.2.2. 网络

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

### 13.2.3. G 训练

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

### 13.2.4. D 训练
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

### 13.2.5. iteration

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

## 13.3. MINST USPS adversarial examples

### 13.3.1. dataset

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

### 13.3.2. general train and evaluate

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