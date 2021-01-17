# 1. torchvision


包含用于计算机视觉的流行数据集，模型架构和常见图像转换   

The torchvision package consists of :
1. popular datasets
2. model architectures
3. common image transformations for computer vision


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



# 4. torchvision.utils

vision 的 utils 里只有两个函数

1. make_grid() 制作一个网状图形 可用于填入多个样本
2. save_image  将一个 Tensor 保存到文件中



# 5. torchvision.transforms

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


## 5.1. Scrpit and Compositions


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

## 5.2. Transforms on PIL Image and torch.*Tensor 通用变换函数

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