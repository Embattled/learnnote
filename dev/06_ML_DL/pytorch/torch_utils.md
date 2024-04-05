# 1. torch.utils

`torch.util` 的核心主要是其下的各种细化的 util,  `torch.util` 下面没有什么函数  

## 1.1. torch.utils.data
Pytorch 自定义数据库中最重要的部分  
提供了对 `dataset` 的所种操作模式  

### 1.1.1. 数据集类型

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

### 1.1.2. torch.utils.data.Dataset

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


### 1.1.3. torch.utils.data.DataLoader

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


## 1.2. torch.utils.tensorboard

完整的说明在  
https://www.tensorflow.org/tensorboard/

只有一个类, 即 SummaryWriter
`torch.utils.tensorboard.writer.SummaryWriter(log_dir=None, comment='', purge_step=None, max_queue=10, flush_secs=120, filename_suffix='')`

