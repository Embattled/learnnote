- [1. Pillow](#1-pillow)
  - [1.1. 基础概念](#11-基础概念)
    - [1.1.1. mode](#111-mode)
    - [Filters](#filters)
- [2. Image](#2-image)
  - [2.1. 定义 Image 对象实例](#21-定义-image-对象实例)
    - [2.1.1. open](#211-open)
    - [2.1.2. new](#212-new)
    - [2.1.3. fromarray](#213-fromarray)
  - [2.2. Image Class](#22-image-class)
    - [2.2.1. 基本操作](#221-基本操作)
    - [2.2.2. Attribute](#222-attribute)
    - [2.2.3. 简单方法](#223-简单方法)
    - [2.2.4. transform](#224-transform)
- [3. ImageDraw](#3-imagedraw)
  - [3.1. 类方法](#31-类方法)
    - [3.1.1. .text](#311-text)
- [4. ImageOps](#4-imageops)
  - [4.1. 黑白彩色转换](#41-黑白彩色转换)
- [5. Other 小包](#5-other-小包)
  - [5.1. ImageMorph](#51-imagemorph)
  - [5.2. ImageTransform  <span id="transform"></span>](#52-imagetransform--)
- [6. ImageFont](#6-imagefont)
  - [6.1. truetype](#61-truetype)
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

### Filters

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


## 2.1. 定义 Image 对象实例

* Image 类是PIL中最重要的类, 可以通过多种方法定义一个实例

### 2.1.1. open

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


### 2.1.2. new

* `PIL.Image.new(mode, size, color=0)` 凭空创建一片内存用来写入图像
  * 注意该函数的参数不是关键字参数
  * mode : 用于新图像的格式
  * size : 2-tuple, 用于指定 (width, height)
  * color: 指定该图像的底色, 默认是0 黑色
    * 如果传入 None, 则该图像不会进行初始化
    * a single integer or floating point value for single-band modes
    * a tuple for multi-band modes (one value per band).
    * 如果图像格式是 RGB, 那么可以用PIL支持的颜色字符串

### 2.1.3. fromarray

* `PIL.Image.fromarray(obj, mode=None)`
* 提供了与 Numpy 的相互转换功能
  * obj , numpy 的实例
  * mode , 可以手动指定图像模式
* PIL转为 np 的方法定义在了 np中 `np.asarray(PILimage)`



## 2.2. Image Class

* `PIL.Image.Image` PIL 中的唯一最核心的类
* 一些基础的图像操作也作为了类的方法加入

### 2.2.1. 基本操作 

`im 为 image 实例对象`:
* im.show()
  * 该函数会调用 `PIL.ImageShow.show(im, title=None, **options)`
  * 可以通过 title 指定显示的窗口标题, 但并不是所有的 viewer都可以正确的将该标题显示
  * 一般不建议用 PIL 中的 show, 因为它不效率, 而且难以 debug

* `Image.save(fp, format=None, **params)`
  * fp, 保存的目标, 可以是 str 也可以是 file object
  * 第二个参数是格式, 默认通过文件名推断保存的格式

### 2.2.2. Attribute

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

### 2.2.4. transform

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



# 3. ImageDraw

* Provides simple 2D graphics for Image objects.
* Use this module to :
  * create new images, 
  * annotate or retouch existing images
  * generate graphics on the fly for web use.


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


# 5. Other 小包
## 5.1. ImageMorph

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


## 5.2. ImageTransform  <span id="transform"></span>

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


# 6. ImageFont

定义了一个同名的 ImageFont 类, 可以保存 bitmap 类型的 fonts  

## 6.1. truetype

```py

PIL.ImageFont.truetype(font=None, size=10, index=0, encoding='', layout_engine=None)
```

