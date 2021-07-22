- [1. torchvision](#1-torchvision)
  - [1.1. torchvision.datasets](#11-torchvisiondatasets)
- [2. torchvision.io](#2-torchvisionio)
  - [2.1. Image part 图像部分](#21-image-part-图像部分)
- [3. torchvision.models](#3-torchvisionmodels)
  - [全局参数](#全局参数)
  - [3.1. image classification](#31-image-classification)
- [4. torchvision.utils](#4-torchvisionutils)
  - [4.1. make_grid](#41-make_grid)
  - [4.2. save_image](#42-save_image)
- [5. torchvision.transforms](#5-torchvisiontransforms)
  - [5.1. Compositions of transforms](#51-compositions-of-transforms)
  - [5.2. 通用变换函数](#52-通用变换函数)
    - [5.2.1. 切割函数](#521-切割函数)
    - [5.2.2. 颜色](#522-颜色)
    - [5.2.3. 噪点](#523-噪点)
    - [5.2.4. 几何变换](#524-几何变换)
    - [5.2.5. 调整大小](#525-调整大小)
    - [5.2.6. 随机应用](#526-随机应用)
    - [5.2.7. 还没看的](#527-还没看的)
  - [5.3. 数据格式转换 Conversion Transforms](#53-数据格式转换-conversion-transforms)
  - [5.4. 一般性自定义函数 Generic Transforms](#54-一般性自定义函数-generic-transforms)
- [6. 基础性变化 Functional Transforms](#6-基础性变化-functional-transforms)
  - [6.1. 类型转换](#61-类型转换)
    - [6.1.2. convert_image_dtype](#612-convert_image_dtype)
  - [6.2. Scrpit and Compositions](#62-scrpit-and-compositions)
  - [6.3. Transforms on only 特定函数](#63-transforms-on-only-特定函数)

# 1. torchvision

包含用于计算机视觉的流行数据集，模型架构和常见图像转换   

The torchvision package consists of :
1. popular datasets
2. model architectures
3. common image transformations for computer vision

## 1.1. torchvision.datasets


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
# 2. torchvision.io


* The torchvision.io package provides functions for performing IO operations. 
* They are currently specific to reading and writing video and images

## 2.1. Image part 图像部分

1. read_image   (path: str) → torch.Tensor
2. decode_image (input: torch.Tensor) → torch.Tensor
3. encode_jpeg  (input: torch.Tensor, quality: int = 75) → torch.Tensor
4. write_jpeg   (input: torch.Tensor, filename: str, quality: int = 75)
5. encode_png   (input: torch.Tensor, compression_level: int = 6) → torch.Tensor
6. write_png    (input: torch.Tensor, filename: str, compression_level: int = 6)


```py

# 标准读取图片函数
torchvision.io.read_image(path: str) → torch.Tensor
# 读出的图片为 3 channel RGB, 数据类型是 uint8  [3,height,width]


# 将一个一维 uint8 字节流 tensor 解压成 RGB 图像 tensor
torchvision.io.decode_image(input: torch.Tensor) → torch.Tensor
# 输入的 tensor 是一维的


# 使用 jpeg 压缩图片
torchvision.io.encode_jpeg(input: torch.Tensor, quality: int = 75) → torch.Tensor
# quality : jpeg 压缩的质量,  1~100 的整数
# 输出 一维 tensor, 可以被 decode_image 解压


# 使用 png 压缩图片
torchvision.io.encode_png(input: torch.Tensor, compression_level: int = 6) → torch.Tensor
# compression_level  : 压缩等级  0~9
# 输出 一维 tensor, 可以被 decode_image 解压


# 图像tensor写入文件并压缩
torchvision.io.write_jpeg(input: torch.Tensor, filename: str, quality: int = 75)
torchvision.io.write_png(input: torch.Tensor, filename: str, compression_level: int = 6)
# filename  : 输出的 path
# input     : [c,h,w] 的tensor, c 必须是 1 或者 3 

```


# 3. torchvision.models

保存了预定义的模型用于不同任务
* image classification
* video classification
* pixelwise semantic segmentation
* object detection, instance segmentation , person keypoint detection

Pytorch 的所有预定义 model 都继承自  `nn.Module` , 注意类的相关构造参数

相关模型的源代码非常有参考意义, 保存到了 [这里](torchvisionmodel.py)

## 全局参数

* 类的定义为首字母大写, 相关模型获取函数为小写
* 函数的通用参数
  * `pretrained: bool = False`  使用预训练模型
  * `progress: bool = True`     显示模型参数下载进度条

## 3.1. image classification

主要用于图像识别, Classification 的模型
* 通过直接调用构造函数, 可以直接生成一个拥有随机 weights 的模型对象
* 通过传入参数 `pretrained=True` 可以直接获得基于 `model_zoo` 训练好的模型
  * 模型参数会下载并存入一个 `cache` 文件
  * pre-trained 的模型则需要:
    * 输入图像的尺寸至少有 224 且 shape (3 H W)
    * 图像需要被 normalized, 范围 0~1 
    * 
* 注意模型的 train 和 test 可能会有不同的动作
  * train 中可能会有 batch normalization
  * 使用 .train()  和 .eval() 确保模型正常工作



| Network       | 函数名称                                        |
| ------------- | ----------------------------------------------- |
| Alexnet       | alexnet                                         |
| VGG           | vgg11 vgg11_bn vgg13 vgg13_bn *16 *19           |
| ResNet        | resnet18 resnet34 resnet50 resnet101  resnet152 |
| squeezeNet    | squeezenet1_0 squeezenet1_1                     |
| DenseNet      | densenet121 densenet169 densenet161 densenet201 |
| Inception v3  | inception_v3                                    |
| GoogLeNet     | googlenet                                       |
| ShuffleNet v2 | ShuffleNet_v2_*                                 |
| MobileNet V2  | mobilenet_v2                                    |
| MobileNet V3  | mobilenet_v3_large  mobilenet_v3_small          |
| ResNext       | resnext50_32x4d resnext101_32x8d                |
| Wide ResNet   | wide_resnet50_2 wide_resnet101_2                |
| MNASNet       | mnasnet0_5 mnasnet0_75 mnasnet1_0 mnasnet1_3    |
 

# 4. torchvision.utils

vision 的 utils 里只有两个函数, 比较简单

## 4.1. make_grid

* 制作一个网状图形 可用于填入多个样本后进行样本展示
* 返回值是一个 Tensor

参数说明:
* tensor    : 具有相同大小的多个图像 , 4D Tensor(B,C,H,W) 或者一个images列表
* nrow      : 网格打印后每行的小图像个数, 会决定最终图像的宽高
* padding   : Amount of padding, 每个小图像留一个边框
* pad_value : Value for the padded pixels. , 即边框的颜色, 灰度值, 255 为白色

标准化相关的参数:
* normailize: 是否使用标准化, 使用的话则根据 range 的值将图像shift到 (0,1)
* range     : Used to normalize the image, 默认是根据 tensor 来自动计算
* scale_each: 是否各自标准化, 开启的话相当于自动根据每个图像计算各自的 range

```py
torchvision.utils.make_grid(
    tensor: Union[torch.Tensor,List[torch.Tensor]],
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    range: Union[Tuple[int,int],NoneType] = None,
    scale_each: bool = False,
    pad_value: int = 0)
→ torch.Tensor
```

## 4.2. save_image

将一个 Tensor 保存到文件中

保存参数:
* fp        : 文件名 或者 `a file object`
* format    : 保存格式, 可以从文件名中推测出来, 但是如果使用 `file object` 则需要手动赋值

重复参数:
* tensor    : 如果传入了 4D tensor 则相当于执行了 make_grid
* padding
* pad_value
* normalize
* range
* scale_each

```py
torchvision.utils.save_image(
    fp: Union[str, pathlib.Path, BinaryIO], 
    format: Union[str, NoneType] = None

    tensor: Union[torch.Tensor, List[torch.Tensor]],
    padding: int = 2, 
    pad_value: int = 0, 
    normalize: bool = False, 
    range: Union[Tuple[int, int], NoneType] = None, 
    scale_each: bool = False, 
    nrow: int = 8, 
) → None

```
无返回值

# 5. torchvision.transforms

1. transforms包含了一些常用的图像变换，这些变换能够用 `Compose` 串联组合起来  
2. `torchvision.transforms.functional` 模块供了一些更加精细的变换，用于搭建复杂的变换流水线(例如分割任务）  


大部分的 Transformation 接受 :
1. PIL Image
2. Tensor Image
3. batch of Tensor Images

例如
* Tensor Image is a tensor with (C, H, W) shape, where C is a number of channels
* Batch of Tensor Images is a tensor of (B, C, H, W) shape, where B is a number of images in the batch. 



分类: 
* Compositions of transforms
* Transforms on PIL Image and torch.*Tensor
* Transforms on PIL Image only
* Transforms on torch.*Tensor only
* Conversion Transforms
* Generic Transforms
* Scriptable transforms

## 5.1. Compositions of transforms

* 将多个变换组合起来方便调用, 但是这个变换不支持 torchscript  
* 除了 torchvision.transforms.functional module 里的函数都可以进行 compose

```py
class torchvision.transforms.Compose(transforms)

# transforms (list of Transform objects)
# 注意是list, 即需要 Compose([ transform1,transform2]) 带方括号的写法

# 标准化
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
# 串联组合起来， 其中还包含了图像转 Tensor的过程
preprocess = transforms.Compose([
    # transforms.Scale(256),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

# 加入到图像loader的过程中
def sscd_loader(path):
    img_pil = Image.open(path)
    img_pil = img_pil.resize((64,64))
    img_tensor = preprocess(img_pil)
    return img_tensor
```


## 5.2. 通用变换函数 

* 最简单的函数, 比较通用, 自定义能力较低
* 可以应用在  PIL Image, Tensor Image or batch of Tensor Images
* 随机变换应用在 batch 上的时候, 会对全部图片应用同一参数的变换
* 以下省略 `torchvision.transforms.*`

### 5.2.1. 切割函数

会将切割下来的图像部分返回, 原图像不受改变  
* CenterCrop(size)      : 中心
* FiveCrop(size)        : 经典的 4个角落加中心 
* TenCrop(size, vertical_flip=False)
* RandomCrop

```py
torchvision.transforms.CenterCrop(size)
    return 	PIL Image or Tensor

torchvision.transforms.FiveCrop(size)
    return tuple of 5 images. Image can be PIL Image or Tensor
# size : 整数或者 (h,w)的列表元组  整数的话代表正方形  指定切割的大小
```

### 5.2.2. 颜色

随机改变图像的 亮度, 对比度, 饱和度, 色度:
* ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
  * 四个参数都接受 float 或者 (min,max) 值的元组
  * 前三个都会 chosen uniformly from [max(0, 1 - 输入值), 1 + 输入值] or the given [min, max]
  * hue 会 chosen uniformly from [-hue, hue] or the given [min, max]

### 5.2.3. 噪点

* GaussianBlur(kernel_size, sigma=(0.1, 2.0))
* 参数:
    - kernel_size : int or (kx,ky) 高斯模糊的核的大小
    - sigma       : float or (min,max), 指定要被用于模糊的标准差, 

### 5.2.4. 几何变换

* RandomAffine(degrees, translate=None, scale=None, shear=None, resample=0, fillcolor=0)

仿射变换：除了角度是必须参数, 其他的都默认为空

* degrees   : ( sequence float or int)  指定 rotations 的角度, 可以是一个元组指定 (min,max), 也可以是单个数字代表 范围是(-n,n), 0 代表不使用旋转
* translate : 指定平移, None代表不平移, 输入元组 (a,b) 代表随机水平平移 %a, 或者垂直%b, a和b的值小于等于1
* scale     : 指定缩放, 同样的是 (a,b) 代表水平和垂直的 百分比
* shear     : ( sequence float or int) 输入类型和 degrees 类似但是有不同
    * 一个值n  代表水平随机 shear (-n,n)
    * 两个值   代表水平随机 shear (n1,n2)
    * 只有四个值的时候 才有垂直 shear (n3,n4)

* RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=2, fill=0)
投影变换, 这个变换可以设置应用概率

* destortion_scale : (float) 0~1 , 代表扭曲的程度
* inerploation     : 一个整数代表插值的类型, 例如 `Nearest Bilinear`
* fill             : 填入空白位置的像素, 这里的 fill 可以是多通道的元组



### 5.2.5. 调整大小
* Resize(size, interpolation=2)

### 5.2.6. 随机应用  

* RandomApply(transforms, p=0.5)
写法和 Compose 有点类似, 但是多了一个 p 参数  


### 5.2.7. 还没看的

* Grayscale(num_output_channels=1)
* Pad(padding, fill=0, padding_mode='constant')
* RandomGrayscale(p=0.1)
* RandomHorizontalFlip(p=0.5)
* RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2)
* RandomRotation(degrees, resample=False, expand=False, center=None, fill=None)
* RandomSizedCrop(*args, **kwargs)

## 5.3. 数据格式转换 Conversion Transforms 

定义到 compose 里, 将 PIL 转化成 Tensor 或者反向变化
* class torchvision.transforms.ToTensor
  * 注意这个转化后的数据类型变化和标准化
  * Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] 
  * to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] 

* class torchvision.transforms.ToPILImage(mode=None)
  * 反向转化格式
  * Converts a torch.*Tensor of shape C x H x W or a numpy ndarray 
  * of shape H x W x C to a PIL Image while preserving the value range.

## 5.4. 一般性自定义函数 Generic Transforms

* class torchvision.transforms.Lambda(lambd)
    * 将用户定义的 lambda 作为变换函数


# 6. 基础性变化 Functional Transforms 

* `torchvision.transforms.functional`  算是一个独立出来的 module `as TF`
* 和 torchvision 其他变换函数相反
* 这里的函数都不包含随机数, 意味着更手动, 但是可以控制所有变化
* 需要输入 image 的 tensor , 意味着不能用 compose 组合起来


## 6.1. 类型转换

并不是变换函数, 但是是找了很久的 pytorch 中的图像类型转换函数

* to_tensor
  * `torchvision.transforms.functional.to_tensor(pic)`
  * 将 PIL 或者 ndarray 转换成 tensor
  * 注意如果输入的像素值是8位255整数, 这个函数会将像素值转为浮点数 0~1, 即颜色缩放
* pil_to_tensor
  `torchvision.transforms.functional.pil_to_tensor(pic)`  
  * 如名称, 将 `PIL Image` 转换成相同类型  
  * 只有一个参数, 就是 PIL Image 对象
  * 这个函数不会进行浮点数缩放
* to_pil_image
  * `torchvision.transforms.functional.to_pil_image(pic, mode=None)`
  * 将 tensor 或者 ndarray 转换成 PIL 图片
  * `mode` 可以指定目标类型
* 


### 6.1.2. convert_image_dtype

Convert a tensor image to the given dtype and scale the values accordingly  
* 利用 torchvision.io 读取的数据虽然是`Tensor`, 但是数值是 `uint8` 类型
* 可以用该函数转换成标准 float32 类型的 tensor 或者其他类型
* dtype 可以指定需要转换成的数据类型


```py
torchvision.transforms.functional.convert_image_dtype(
    image: torch.Tensor, 
    dtype: torch.dtype = torch.float32
)→ torch.Tensor

```


## 6.2. Scrpit and Compositions


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

## 6.3. Transforms on only 特定函数


**Transforms on PIL Image Only**
* class torchvision.transforms.RandomChoice(transforms)
* class torchvision.transforms.RandomOrder(transforms)

**Transforms on torch.*Tensor only**
* class torchvision.transforms.LinearTransformation(transformation_matrix, mean_vector)
* class torchvision.transforms.Normalize(mean, std, inplace=False)
* class torchvision.transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
* class torchvision.transforms.ConvertImageDtype(dtype: torch.dtype)