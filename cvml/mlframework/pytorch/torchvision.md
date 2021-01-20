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

## 2.1. Image part

1. torchvision.io.read_image(path: str) → torch.Tensor
2. torchvision.io.decode_image(input: torch.Tensor) → torch.Tensor
3. torchvision.io.encode_jpeg(input: torch.Tensor, quality: int = 75) → torch.Tensor
4. torchvision.io.write_jpeg(input: torch.Tensor, filename: str, quality: int = 75)
5. torchvision.io.encode_png(input: torch.Tensor, compression_level: int = 6) → torch.Tensor
6. torchvision.io.write_png(input: torch.Tensor, filename: str, compression_level: int = 6)




# 3. torchvision.models

保存了预定义的模型用于不同任务
* image classification
* video classification
* pixelwise semantic segmentation
* object detection, instance segmentation , person keypoint detection

## 3.1. image classification

* 通过调用构造函数可以直接获得网络模型对象  
* 通过传入参数 `pretrained=True` 可以直接获得基于 `model_zoo` 训练好的模型
* pre-trained 的模型则需要输入图像的尺寸至少有 224






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
* Functional Transforms
* Scriptable transforms

## 5.1. Compositions of transforms

将多个变换组合起来方便调用, 但是这个变换不支持 torchscript  

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
* RandomCrop

```py
torchvision.transforms.CenterCrop(size)
    return 	PIL Image or Tensor

torchvision.transforms.FiveCrop(size)
    return tuple of 5 images. Image can be PIL Image or Tensor
# size : 整数或者 (h,w)的列表元组  整数的话代表正方形  指定切割的大小
```

### 5.2.2. 颜色噪点

随机改变图像的 亮度, 对比度, 饱和度, 色度:
* ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
  * 四个参数都接受 float 或者 (min,max) 值的元组
  * 前三个都会 chosen uniformly from [max(0, 1 - 输入值), 1 + 输入值] or the given [min, max]
  * hue 会 chosen uniformly from [-hue, hue] or the given [min, max]





* Grayscale(num_output_channels=1)
* Pad(padding, fill=0, padding_mode='constant')
* RandomAffine(degrees, translate=None, scale=None, shear=None, resample=0, fillcolor=0)
* RandomGrayscale(p=0.1)
* RandomHorizontalFlip(p=0.5)
* RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=2, fill=0)
* RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2)
* RandomRotation(degrees, resample=False, expand=False, center=None, fill=None)
* RandomSizedCrop(*args, **kwargs)

* TenCrop(size, vertical_flip=False)
* GaussianBlur(kernel_size, sigma=(0.1, 2.0))

### 5.2.3. 调整大小
* Resize(size, interpolation=2)


### 5.2.4. 随机应用  

* RandomApply(transforms, p=0.5)
写法和 Compose 有点类似, 但是多了一个 p 参数  


## 5.3. Transforms on only 特定函数


**Transforms on PIL Image Only**
* class torchvision.transforms.RandomChoice(transforms)
* class torchvision.transforms.RandomOrder(transforms)

**Transforms on torch.*Tensor only**
* class torchvision.transforms.LinearTransformation(transformation_matrix, mean_vector)
* class torchvision.transforms.Normalize(mean, std, inplace=False)
* class torchvision.transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
* class torchvision.transforms.ConvertImageDtype(dtype: torch.dtype)


## 5.4. Conversion and Generic Transforms 格式转换 和 通用变化

* class torchvision.transforms.ToPILImage(mode=None)
* class torchvision.transforms.ToTensor


* class torchvision.transforms.Lambda(lambd)
    * 将用户定义的 lambda 作为变换函数




## 5.5. Functional Transforms 用于更精细化的变换

`import torchvision.transforms.functional as TF`  函数名称省略前缀  


## 5.6. Scrpit and Compositions


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