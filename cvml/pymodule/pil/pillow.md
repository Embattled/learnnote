- [1. Pillow](#1-pillow)
  - [1.1. 基础概念](#11-基础概念)
    - [1.1.1. mode](#111-mode)
    - [1.1.2. Filters](#112-filters)
- [2. Image](#2-image)
  - [2.1. 定义 Image 对象实例](#21-定义-image-对象实例)
    - [2.1.1. PIL.Image.open](#211-pilimageopen)
    - [2.1.2. PIL.Image.new](#212-pilimagenew)
    - [2.1.3. PIL.Image.fromarray](#213-pilimagefromarray)
  - [2.2. Image Class](#22-image-class)
    - [2.2.1. Attribute](#221-attribute)
    - [2.2.2. 基本操作](#222-基本操作)
    - [2.2.3. 简单方法](#223-简单方法)
    - [2.2.4. Image.transpose](#224-imagetranspose)
    - [2.2.5. Image.transform](#225-imagetransform)
    - [2.2.6. Image.filter](#226-imagefilter)
  - [2.3. Generating images 图片生成](#23-generating-images-图片生成)
    - [2.3.1. PIL.Image.effect_*](#231-pilimageeffect_)
    - [2.3.2. PIL.Image.*_gradient](#232-pilimage_gradient)
- [3. ImageDraw](#3-imagedraw)
  - [基础知识](#基础知识)
  - [3.1. 类方法](#31-类方法)
    - [3.1.1. .text](#311-text)
- [4. ImageOps](#4-imageops)
  - [4.1. 黑白彩色转换](#41-黑白彩色转换)
- [5. ImageFont](#5-imagefont)
  - [5.1. truetype](#51-truetype)
- [6. Other 小包](#6-other-小包)
  - [6.1. ImageMorph](#61-imagemorph)
  - [6.2. ImageFilter](#62-imagefilter)
    - [6.2.1. 预定义filter](#621-预定义filter)
    - [6.2.2. size区域定义filter](#622-size区域定义filter)
    - [6.2.3. radius区域定义filter](#623-radius区域定义filter)
    - [6.2.4. 自定义Kernel](#624-自定义kernel)
  - [6.3. ImageTransform  <span id="transform"></span>](#63-imagetransform--)
# 1. Pillow 

Pillow is the friendly PIL fork

所有子模块都定义在 `PIL` 里  

## 1.1. 基础概念

* 有一些预定义的图像模式或者颜色字符串

### 1.1.1. mode

* Image 的 mode 是一个字符串, 用来代表图像的通道以及色宽
* 标准模式, 完整支持:
  * 1   : 1-bit 每像素, 只有黑白两色
  * L   : 8-bit 每像素, 灰度图
  * P   : 8-bit 每像素, 没太看懂, 好像是一个灰度映射
  * RGB : 标准8-bit 三通道真彩
  * RGBA: 带有 transparency 的RGB
  * CMYK: 4X8-bit
  * YCbCr: 3X8-bit, 有色视频的格式
  * HSV : Hue, Saturation, Value color space
  * LAB : `L*a*b` color space, JPEG的像素格式
  * I   : (32-bit signed integer pixels)
  * F   : (32-bit floating point pixels)

### 1.1.2. Filters

* 对于图像的几何变换, 用于指定像素的映射方法

| 代码:`PIL.Image.*` | 功能                                |
| ------------------ | ----------------------------------- |
| NEAREST            | 最近邻像素                          |
| BILINEAR           | 线性插值                            |
| HAMMING            | 相比线性插值更加锐化                |
| BICUBIC            | 立方插值, 没学过                    |
| LANCZOS            | high-quality Lanczos, 1.1.3新加入的 |

# 2. Image

`from PIL import Image`  
* 整个库中最核心的包
* 包含了所有常用的函数, 其他的包相对来说都是小众功能
* 包含了PIL最核心的类 `class PIL.Image.Image`


## 2.1. 定义 Image 对象实例

* Image 类是PIL中最重要的类, 可以通过多种方法定义一个实例

### 2.1.1. PIL.Image.open

* `PIL.Image.open(fp, mode='r', formats=None)` 打开一个给定的图像文件
  * fp : 指定文件路径  filename (string), pathlib.Path object or a file object.
  * mode : 对该函数来说只能给定 'r', 搞不懂为啥还要独立出来, 表示读取
  * formats : list or tuple, 指定用指定的格式打开文件, None 的话就是尝试所有格式
  * 返回一个 Image object
* 异常
  * FileNotFoundError : 文件未找到
  * ValueError : 如果 mode 未指定成 'r', 或者 fp 被别的 IO 占用
  * TypeError : format 传入异常值
  * PIL.UnidentifiedImageError : 无法打开或者无法识别的图像文件


### 2.1.2. PIL.Image.new

* `PIL.Image.new(mode, size, color=0)` 凭空创建一片内存用来写入图像
  * 注意该函数的参数不是关键字参数
  * mode : 用于新图像的格式
  * size : 2-tuple, 用于指定 (width, height)
  * color: 指定该图像的底色, 默认是0 黑色
    * 如果传入 None, 则该图像不会进行初始化
    * a single integer or floating point value for single-band modes
    * a tuple for multi-band modes (one value per band).
    * 如果图像格式是 RGB, 那么可以用PIL支持的颜色字符串

### 2.1.3. PIL.Image.fromarray

* `PIL.Image.fromarray(obj, mode=None)`
* 提供了与 Numpy 的相互转换功能
  * obj , numpy 的实例
  * mode , 可以手动指定图像模式
* PIL转为 np 的方法定义在了 np中 `np.asarray(PILimage)`



## 2.2. Image Class

* `PIL.Image.Image` PIL 中的唯一最核心的类
* 一些基础的图像操作也作为了类的方法加入
* 类的方法默认都是创建一个新的Image实例作为返回值

### 2.2.1. Attribute

Image class 拥有的成员属性:
* 图像信息
  * format    : str or None, 图像文件的格式, 如果是从内存中从各种函数中创建的图像则为 `None`
  * mode      : str, 喜闻乐见
  * size      : 2-tuple , 记住是 (width,height), 但是 torch tensor 的图像tensor一般是 C H W,
  * width     : int
  * height    : int
* 文件信息
  * filename: str, 如果是从文件中读取的图像, 会自动保留文件路径, 但是使用 file object获取的图像不会得到路径

* 看不懂
  * palette
  * info
  * is_animated
  * n_frames

### 2.2.2. 基本操作 

`im 为 image 实例对象`:
* im.show()
  * 该函数会调用 `PIL.ImageShow.show(im, title=None, **options)`
  * 可以通过 title 指定显示的窗口标题, 但并不是所有的 viewer都可以正确的将该标题显示
  * 一般不建议用 PIL 中的 show, 因为它不效率, 而且难以 debug

* `Image.save(fp, format=None, **params)`
  * fp, 保存的目标, 可以是 str 也可以是 file object
  * 第二个参数是格式, 默认通过文件名推断保存的格式

* `Image.copy()`
  * 返回一个赋值的 Image class 对象


除了成员属性外, 还有一些方法也可以获得Image的属性:  
* `im.getbands()`
  * 获得成员的各个通道名称, 如果是RBG图像就会返回 `(“R”, “G”, “B”)`
  * 源代码是直接返回 `ImageMode.getmode(self.mode).bands`
  * 因为成员 size 不包括通道数, 因此可以通过 `len(im.getbands())` 获取图像的通道数


### 2.2.3. 简单方法

* `Image.paste(im, box=None, mask=None)`
  * 将另一个图片 im 粘贴到该图像的 box 位置
  * im  : images, 或者 pixel value (integer or tuple).
  * box : 2-tuple or 4-tuple
    * 传入 2-tuple 表示被粘贴图片的左上角的目标坐标
    * 传入 4-tuple 表示左上角和右下角的坐标, 需要被粘贴图片的大小匹配, 但是好像并不会自动缩放
    * 传入 None 就是默认的左上角
  * mask: 另一个 image, 可以是 1 L RGBA, RGBA的话只会使用 A通道
    * 用值来代表拷贝的mask, mask附在被拷贝图像上
    * 0 代表保留目标图像原本的值
    * 255 代表完整赋予被拷贝图像的像素值, 中间的值代表 mix

### 2.2.4. Image.transpose

`Image.transpose(method)` 对图像进行反转, method传入几个预定义常量
1. PIL.Image.FLIP_LEFT_RIGHT,
2. PIL.Image.FLIP_TOP_BOTTOM,
3. PIL.Image.ROTATE_90,
4. PIL.Image.ROTATE_180,
5. PIL.Image.ROTATE_270,
6. PIL.Image.TRANSPOSE
7. PIL.Image.TRANSVERSE.

### 2.2.5. Image.transform

* Image.transform 作为一个类方法, 却整合了多种变换方法
* 作为一个 core 方法, 文档中并没有详细描写使用方法
* 具体的接口以及具体参数定义在了 [ImageTransform](#transform) 中
* `Image.transform(size, method, data=None, resample=0, fill=1, fillcolor=None)`
  * size : 输出的大小
  * method : 变换方法
  * data=None : 对一些变换需要的额外数据
  * resample : 对一些变换需要的平滑重采样方法, 根据 Filter 的不同有不同的速度性能和效果性能, 默认是效果最差的 Nearest
  * fill : 只有 `ImageTransformHandler` 才使用, 一个额外参数
  * fillcolor : 一些变换对于超出区域的填充值

method 变换方法:
1. PIL.Image.EXTENT       : 切选出一个矩形子区域
2. PIL.Image.AFFINE       : Affine 变换
   * 传入 `data[0:6]`
3. PIL.Image.PERSPECTIVE  : 投影变换
   * 传入 `data[0:8]`
4. PIL.Image.QUAD         : 将一个四边形映射成矩阵
5. PIL.Image.MESH         : 在一次操作中映射多个源四边形
6. 可以是自定义的 `ImageTransformHandler` 类对象
7. 可以是自定义的任意类对象, 但是必须要有 拥有`getdata()` 方法

### 2.2.6. Image.filter

* `.filter(filter)` 用于应用一个图像滤波器, 返回应用后的图像
* 可用的滤波器定义在了 `ImageFilter` module 中  

## 2.3. Generating images 图片生成

* 可以生成一些特殊的图片, 用于和其他图片结合实现一些处理
* 返回一个 Image 实例

### 2.3.1. PIL.Image.effect_*

* `PIL.Image.effect_noise(size,sigma)`
  * 生成一个高斯噪声图
  * size : 2-tuple, (width, height)
  * sigma: noise 的标准差(deviation) sigma
* `PIL.Image.effect_mandelbrot(size, extent, quality)`
  * 生成一个 Mandelbrot set covering the given extent.
  * size : 2-tuple, (width, height)
  * extent – The extent to cover, as a 4-tuple: (x0, y0, x1, y2).
  * quality – Quality.

### 2.3.2. PIL.Image.*_gradient

这个函数会生成一个固定大小(256,256)的正方形图

* 输入 mode
* `PIL.Image.linear_gradient(mode)`
  * 从上黑到下白的渐变图
* `PIL.Image.radial_gradient(mode)`
  * 从中心黑到边缘白的渐变图


# 3. ImageDraw

* Provides simple 2D graphics for Image objects.
* Use this module to :
  * create new images, 
  * annotate or retouch existing images
  * generate graphics on the fly for web use.

## 基础知识

* 坐标系: 标准 (0,0) 代表左上角的坐标系
* 颜色  : 同创建图像的时候相同, 整数或者 3-tuple 或者浮点数
* 

## 3.1. 类方法
### 3.1.1. .text

```py
ImageDraw.text(
    xy, 
    text, 
    fill=None, 
    font=None, 
    anchor=None, 
    spacing=4, 
    align='left', 
    direction=None, 
    features=None, 
    language=None, 
    stroke_width=0, 
    stroke_fill=None, 
    embedded_color=False)
```


# 4. ImageOps

* Most operators only work on L and RGB images.
* Pillow 最主要的图片操作函数, 主要是像素值处理

## 4.1. 黑白彩色转换

* 彩色转黑色, 黑色转彩色

* `PIL.ImageOps.grayscale(image)`
  * 将一个图片转换成 GrayScale
  * 该函数源代码就是直接将 mode 更改为 `("L")`

* `PIL.ImageOps.colorize(image, black, white, mid=None, blackpoint=0, whitepoint=255, midpoint=127)`
  * 将一个 GrayScale 转成 RBG彩色
  * black white mid 分别是一个 RBG tuples
  * 该颜色赋予是有梯度的, 不是单纯的二值赋值, 将灰度值按照比例复制到参数 black-white 中间的颜色
  * mid 可是可选的, 指定了 mid 代表这是一个三区间 mapping



# 5. ImageFont

定义了一个同名的 ImageFont 类, 可以保存 bitmap 类型的 fonts  

## 5.1. truetype

```py

PIL.ImageFont.truetype(font=None, size=10, index=0, encoding='', layout_engine=None)
```


# 6. Other 小包
## 6.1. ImageMorph

提供对图片的形态学变化

* 创建变换
  * A class for building a MorphLut from a descriptive language
  * `class PIL.ImageMorph.LutBuilder(patterns=None, op_name=None)`
  * `patterns` 是一个单独的描述性语言, 用来说明要进行的变换
  * `op_name`  可以使用预定义的变换
  * 类成员函数
    * `add_patterns( patterns )`
    * `build_lut()` 编译这个变换, 返回 morphology lut.
    * `get_lut()` 获取 lut

```py
# patterns 是一个列表包裹着的字符串, 类似于
p=["4:(....1.111)->1"]
"""
Operations:
    - 4 - 4 way rotation
    - N - Negate
    - 1 - Dummy op for no other operation (an op must always be given)
    - M - Mirroring

Kernel:
    - . or X - Ignore
    - 1 - Pixel is on
    - 0 - Pixel is off

->0 : 代表该 kernel 匹配时, 输出的像素值
"""
known_patterns = {
              "corner": ["1:(... ... ...)->0", "4:(00. 01. ...)->1"],
              "dilation4": ["4:(... .0. .1.)->1"],
              "dilation8": ["4:(... .0. .1.)->1", "4:(... .0. ..1)->1"],
              "erosion4": ["4:(... .1. .0.)->0"],
              "erosion8": ["4:(... .1. .0.)->0", "4:(... .1. ..0)->0"],
              "edge": [
                  "1:(... ... ...)->0",
                  "4:(.0. .1. ...)->1",
                  "4:(01. .1. ...)->1",
              ],
          }

```

* 应用变换
  * A class for binary morphological operators  
  * `class PIL.ImageMorph.MorphOp(lut=None, op_name=None, patterns=None)`


## 6.2. ImageFilter

用于产生可以被用于 `Image.filter()`的参数的 filter  
* 可传参数的filter的定义都是作为类出现的  

### 6.2.1. 预定义filter

* ImageFilter.BLUR                : 就是模糊
* ImageFilter.CONTOUR             : 轮廓提取
* ImageFilter.DETAIL              : 增强细节
* ImageFilter.EDGE_ENHANCE        : 增强边缘
* ImageFilter.EDGE_ENHANCE_MORE   : 深度边缘增强
* ImageFilter.EMBOSS              : 浮雕滤波
* ImageFilter.FIND_EDGES          : 边界提取
* ImageFilter.SHARPEN             : 锐化
* ImageFilter.SMOOTH              : 平滑
* ImageFilter.SMOOTH_MORE         : 深度平滑

### 6.2.2. size区域定义filter

传入的整数参数 `size`  代表filter 的感受野是 (size,size) 的正方形大小  
* class PIL.ImageFilter.RankFilter(size, rank)  : 等级滤波, 选择区域中 rank's小的像素值作为新值, 0 代表最小值, `size*size-1` 代表最大值, 可以替代实现下方三种滤波
* class PIL.ImageFilter.MedianFilter(size=3)    : 中值滤波
* class PIL.ImageFilter.MinFilter(size=3)       : 最小值
* class PIL.ImageFilter.MaxFilter(size=3)       : 最大值滤波
* class PIL.ImageFilter.ModeFilter(size=3)      : 模式滤波, 使用区域中出现次数最多的像素值, 如果出现次数都是1则选择原本的像素值

### 6.2.3. radius区域定义filter

* radius 支持小数
* radius = 0 代表无操作
* radius = 1 代表 9 pixels in total

区域定义filter:
* class PIL.ImageFilter.BoxBlur(radius)           : 均值模糊
* class PIL.ImageFilter.GaussianBlur(radius=2)    : 高斯模糊
* class PIL.ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3)
  * 线性反锐化掩膜, 提高图像的高频部分, 保持低频部分不变, 可以增强图片的轮廓细节
  * `percent`  Unsharp strength, in percent
  * `threshold` 阈值, Threshold controls the minimum brightness change that will be sharpened

### 6.2.4. 自定义Kernel

`class PIL.ImageFilter.Kernel(size, kernel, scale=None, offset=0)`  
* 完全的自定义卷积核, 只能用在 `RGB` 和 `L` 图形上
* `size` 卷积核的大小, 只能是 33 或者 55
* `kernel` 一个序列, 传入卷积核的 weight
* `scale` 均一化的值, 像素的最终值是卷积的结果以scale, 默认即可(卷积weight的sum)
* `offset` 偏差, 用加法添加到像素的最终值

## 6.3. ImageTransform  <span id="transform"></span>

* 定义了的图像变换方法接口
* Pillow官方文档还没有补充完整
* 基本上源代码就是 Transform 的说明文档, 代码里没什么内容

* `class Transform(Image.ImageTransformHandler):`
  * 定义了几个经典变化的骨架

1. Affine: 接受一个 6-tuple, 用来组成 affine 变换矩阵的 前两行
   - 对于坐标点 x,y的数据, 从原图像的 (ax+by+c, dx+ey+f ) 来获取
2. Extent: 接受一个 4-tuple, x0,y0,x1,y1, 将原图像的该区域选中并扩大到 size 参数的大小
3. Perspective: 传入的是投影变换的8个代数参数
4. Quad: 接受一个 8-tuple, x0,y0~x3,y3, 代表左上,左下,右下,右上的四个角, 将指定的四边形扩展到 size 参数的大小


```py
class Transform(Image.ImageTransformHandler):
    def __init__(self, data):
        self.data = data

    def getdata(self):
        return self.method, self.data

    def transform(self, size, image, **options):
        # can be overridden
        method, data = self.getdata()
        return image.transform(size, method, data, **options)

class AffineTransform(Transform):
  method = Image.AFFINE

class ExtentTransform(Transform):
  method = Image.EXTENT

class QuadTransform(Transform):
  method = Image.QUAD

class MeshTransform(Transform):
  method = Image.MESH
```
