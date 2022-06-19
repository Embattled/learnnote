# 1. OpenCV

OpenCV (Open Source Computer Vision Library) :
* An open-source library that includes several hundreds of computer vision algorithms.

OpenCV 的历史:
1. C-based (OpenCV 1.x API) has been deprecated. since OpenCV 2.4 release
2. OpenCV 2.x API is essentially a C++ API.


因为 Opencv 4 开始对多语言的支持也较好, 所以从C++的笔记目录中独立出来, 目前支持
* C++
* Python
* js

整个库的功能实现仍然是 C++, 但是提供了对Python 的语言接口

## 1.1. Modular Structure

OpenCV 的整个模组结构:

旧版笔记:
1. Core functionality `core`        : Basic data structures.
2. Image Processing `imgproc`       : Image processing module.
3. Video Analysis `video`           : Motion estimation, tracking.
4. Video I/O `videoio`
5. Camera Calibration and 3D Reconstruction `calib3d`
6. 2D Features Framework `features2d`
7. Object Detection `objdetect`
8. High-level GUI `highgui` 


Opencv4, Main Modules: 除此之外还有 other 
1. `core`.           Core functionality
2. `imgproc`.        Image Processing
3. `imgcodecs`.      Image file reading and writing
4. `videoio`.        Video I/O
5. `video`.          Video Analysis
6. `highgui`.        High-level GUI
7. `calib3d`.        Camera Calibration and 3D Reconstruction
8. `features2d`.     2D Features Framework
9. `objdetect`.      Object Detection
10. `dnn`.           Deep Neural Network module
11. `ml`.            Machine Learning
12. `flann`.         Clustering and Search in Multi-Dimensional Spaces
13. `photo`.         Computational Photography
14. `stitching`.     Images stitching
15. `gapi`.          Graph API* 

## 1.2. opencv C++

### 1.2.1. API Concepts 

* 尽管从文档上, OpenCV 分成了数个模组
* 事实上整个 OpenCV 库都放在了 `cv` 的命名空间中
* 有些函数会和 STL 冲突, 因此有时候需要特殊指定

` #include "opencv2/core.hpp" `

### 1.2.2. Install on linux

只能通过自己编译并安装的方法安装OpenCV, apt 中没有索引  

```sh
# 主要版本在 github 上的 url 是固定的
wget -O opencv.zip https://github.com/opencv/opencv/archive/master.zip

# 解压
unzip opencv.zip

# 创建一个 build 文件夹保存编译文件
mkdir -p build

# 在 build 文件夹中调用 cmake 命令 输入源文件夹
cmake opencvfolder

# 正常编译
make -j4

# 检查编译结果
ls bin
ls lib

# Root 安装, 会将所有文件安装到 /usr/local 对应的位置下

    /usr/local/bin - executable files
    /usr/local/lib - libraries (.so)
    /usr/local/cmake/opencv4 - cmake package
    /usr/local/include/opencv4 - headers
    /usr/local/share/opencv4 - other files (e.g. trained cascades in XML format)

# 如果非管理者, 最好不要这么安装
sudo make install
# 修改默认安装位置
cmake -DCMAKE_INSTALL_PREFIX=$HOME/.local

```

## 1.3. opencv python

* OpenCV-Python is a Python wrapper for the original OpenCV C++ implementation.
  * 代码易读
  * 实际的执行计算在后台仍然是C++实现的

依赖库:
* Numpy. OpenCV-Python makes use of Numpy. And OpenCV-Python requires only Numpy.
* All the OpenCV array structures are converted to and from Numpy arrays. 
* This also makes it easier to integrate with other libraries that use Numpy. such as `SciPy` and `Matplotlib`



opencv-python 的 API 都定义在了 cv2 包中, 为了保证代码的可执行性  
`import cv2 as cv`  常用于opencv 包的导入



# 2. core

定义了 OpenCV 中最核心的类 (图像), 以及一些对于 array 的数学上的操作 
* 有些图像处理的功能可能没有定义在图像部分, 而是在 array 对象的更原始的层面里实现了


## 2.1. mat  core/mat.hpp

同 numpy 一样, opencv并没有特地的图片类, 而是用矩阵来表示 -> `cv::Mat`  

## 2.2. Operations on arrays

所有和 array 相关的基础处理方法
* 该模组在 C++ 中比较有意义
* Opencv-Python 因为是基于 Numpy 实现的, 因此有很多函数其实和 Numpy 的功能重复了

### 2.2.1. addWeighted 权重加

```cpp
void cv::addWeighted 	( 	InputArray  	src1,
		double  	alpha,
		InputArray  	src2,
		double  	beta,
		double  	gamma,
		OutputArray  	dst,
		int  	dtype = -1 
	) 		
Python:
	cv.addWeighted(	src1, alpha, src2, beta, gamma[, dst[, dtype]]	) -> 	dst

```

类似于 PIL 的 paste 函数在opencv 里是以 array 对象实现的
* 这里的权重可以是RGBA的 alpha 通道
* 因为函数不会意识到图像, 所以要求输入图像都是相同维度以及相同大小 `src1.shape==src2.shape`
* dst(I)=saturate(src1(I)∗alpha+src2(I)∗beta+gamma)
* 参数
  * alpha , beta 分别是 src1 src2 的透明度通道
  * gamma 是最后的标量
  * 由此可见对于 RGBA 图像, 需要先将 alpha 通道独立出来

# 3. imgcodecs  Image file reading and writing 

同 Image Process 不同, 该模组中的函数都是图像 IO 相关的  

`#include <opencv2/imgcodecs.hpp>` 

## 3.1. Flags used for image file reading and writing

用作图片 IO 的参数, 一般都是枚举类型, 用于指定图片的规格等

* 在C++下, OpenCV 会以内建的 Mat 类来处理图像
* 在 Python OpenCV 中则使用 numpy.ndarry 来处理, 因此 Mat 类的方法在 Python 下不能使用

### 3.1.1. ImreadModes

用于读取时候的模式

EXIF : 图像格式标记, 包括了该图像正确的展示方向, OpenCV 读取的时候会默认参考该标记得到正确的显示方向

此处省略 `IMREAD_`
* UNCHANGED			: 原汁原味, 返回原本的图像 (其实区别就是对待 png 图片时会有 alpha 通道). Ignore EXIF orientation. 
* GRAYSCALE			: 灰度 , 读取的时候会自动进行色彩转换
* COLOR				: 默认 , 总是自动转换成 BGR 三通道
* ANYDEPTH			: 高深度图像, 支持输入 16-bit 或 32bit 色深图像
* ANYCOLOR			: 任意颜色格式 (不太懂, 是不进行BGR转换的意思?)
* LOAD_GDAL 		: 使用 gdal driver for loading the image (不懂)
* IGNORE_ORIENTATION: 忽视 Exif 的方向标签
* REDUCE 系列, 在读取的时候自动将图片的大小降低对应系数
  * REDUCED_GRAYSCALE_2 
  * REDUCED_COLOR_2 
  * REDUCED_GRAYSCALE_4 
  * REDUCED_COLOR_4 
  * REDUCED_GRAYSCALE_8 
  * REDUCED_COLOR_8 


位表记
| 位  | 对应 Flag          |
| --- | ------------------ |
| 全0 | GRAYSCALE          |
| 1   | COLOR              |
| 2   | ANYDEPTH           |
| 4   | ANYCOLOR           |
| 8   | LOAD_GDAL          |
| 16  | REDUCE_2           |
| 32  | REDUCE_4           |
| 64  | REDUCE_8           |
| 128 | IGNORE_ORIENTATION |
| 255 | UNCHANGED          |

## 3.2. 基础图像读写

`imread()` 用于从一个文件路径中读取图像
* filename  :Name of file to be loaded.
* flags     :Flag that can take values of `cv::ImreadModes`
  * 默认的读取模式是 `IMREAD_COLOR`

`imwrite()` 用于把一个 `cv::Mat` 类写入到硬盘中
* filename	:Name of the file.
* img	      :(Mat or vector of Mat) Image or Images to be saved.
* params	  :Format-specific parameters encoded as pairs
  * (paramId_1, paramValue_1, paramId_2, paramValue_2, ... .)
  * see `cv::ImwriteFlags`


```cpp
Mat cv::imread 	(
  const String &  	filename,
	int  	flags = IMREAD_COLOR )

bool cv::imwrite 	(
  const String &  	filename,
	InputArray  	img,
	const std::vector< int > &  	params = std::vector< int >() 
)


// python
cv.imread(filename[, flags]	) -> 	retval

cv.imwrite(	filename, img[, params]	) -> 	retval
```

# 4. imgproc  Image Processing

`#include <opencv2/imgproc.hpp>`  

对读取的图像进行处理

注意:
* 对于图形处理函数
  * C++ 的 dst 一般都在参数里
  * Python 的 dst 可以是参数也可以是返回值


## 4.1. 色彩空间转换 Color Space Conversions

opencv 支持近乎所有的图形格式之间·`互相`转换  

### 4.1.1. cvtColor

```cpp
void cv::cvtColor 	( 	InputArray  	src,
		OutputArray  	dst,
		int  	code,
		int  	dstCn = 0 
	) 		
Python:
	cv.cvtColor(	src, code[, dst[, dstCn]]	) -> 	dst
```

参数:
* src	: 输入图像
* code	: 转换模式
* dstCn	: dst图像的通道数. 一般置零用来自动判定


### 4.1.2. ColorConversionCodes

通过 `enum cv::ColorConversionCodes` 枚举类型来表示转换的模式, 以下列出所有支持的图片格式的名称, 转换模式即为 `<A>2<B>` 
* BGR RGB LRGB LBGR
  * 565
  * 555
  * 
* BGRA
* RGBA
* GRAY
* XYZ
* YCrCb
* Lab
* Luv
* HSV
  * HSV_FULL
* HLS
  * HLS_FULL
* Bayer
* 太多了算了

## 4.2. Image Filtering 滤波函数

用于对 2D 图像进行各种 线性/非线性 的filtering 操作
* 对于图像中的每个像素 (x,y), 通过其周围像素的值来决定该像素新的值
* 通道之间将会分开来计算, 即可以保证输出图像具有和输入图像相同的通道数


通用参数
* src
* dst
* ddepth
`when ddepth=-1, the output image will have the same depth as the source. `

### 4.2.1. Smoothing Filtering

平滑图片, 即 smoothing, 常被用来:
* reduce noise


* `cv.blur()`				: 均值滤波
* `cv.GaussianBlur()`		: 高斯滤波
* `cv.medianBlur()`			: 中值滤波
* `cv.bilateralFilter()`	: 双边滤波, 是一种结合了高斯滤波和密度的滤波, 可以模糊区域但是保留图像边缘, 非线性
  * 核函数是空间域核与像素范围域核的综合结果
  * 空间域核: 根据边缘像素和中心像素的欧氏距离来确定权重, 可以是 mean 也可以是 二维Gaussian
  * 像素范围域核: 根据边缘像素和中心像素的像素值差来确定权重, 是一个 f(0)=1 的二次凸函数, 差值越大权重越接近0, 可以是 一维Gaussian
  * 整体权重是 两个权重的乘积, 即如果区域内像素值差距小, 则会被应用很强的 smoothing
  * Sigma values: CV里两个域核应该都是高斯函数, 推荐范围是 (10~150)
  * d : Large filters (d > 5) are very slow, so it is recommended to use d=5 for real-time applications, and perhaps d=9 for offline applications that need heavy noise filtering.


### 4.2.2. Morphological Transformation 形态学变化

OpenCV 的形态学变化也放在了 Filtering 模块中

通用参数:
* src		: input image, 支持任意 channel
* dst		: (大概是C++用的)output image of the same size and type as src.
* kernel	: structuring element. 可以用函数 `getStructuringElement()` 创建
* anchor	: 应用 kernel 的锚点, 默认值(-1,-1)不代表任何具体坐标, 会被转换成 kernel 的中心
* iterations: 可以直接在函数参数中设置重复执行多次  


#### 4.2.2.1. 基础 膨胀腐蚀

* dilate() 	膨胀
* erode()	腐蚀
* morphologyEx()

```cpp
void cv::dilate 	( 	InputArray  	src,
		OutputArray  	dst,
		InputArray  	kernel,
		Point  	anchor = Point(-1,-1),
		int  	iterations = 1,
		int  	borderType = BORDER_CONSTANT,
		const Scalar &  	borderValue = morphologyDefaultBorderValue() 
	) 		
Python:
	cv.dilate(	src, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]]	) -> 	dst

void cv::erode 	( 	InputArray  	src,
		OutputArray  	dst,
		InputArray  	kernel,
		Point  	anchor = Point(-1,-1),
		int  	iterations = 1,
		int  	borderType = BORDER_CONSTANT,
		const Scalar &  	borderValue = morphologyDefaultBorderValue() 
	) 		
Python:
	cv.erode(	src, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]]	) -> 	dst
```

#### 4.2.2.2. 高级组合操作

Performs advanced morphological transformations. 

```cpp
void cv::morphologyEx 	( 	InputArray  	src,
		OutputArray  	dst,
		int  	op,
		InputArray  	kernel,
		Point  	anchor = Point(-1,-1),
		int  	iterations = 1,
		int  	borderType = BORDER_CONSTANT,
		const Scalar &  	borderValue = morphologyDefaultBorderValue() 
	) 		
Python:
	cv.morphologyEx(	src, op, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]]	) -> 	dst
```

op 是一个枚举类型, 除了组合的形态学变化也把基础的变化加入进去了
* MORPH_ERODE 
* MORPH_DILATE 
* MORPH_OPEN 			: `dilate(erode(src,element))`
* MORPH_CLOSE 			: `erode(dilate(src,element))`
* MORPH_GRADIENT 		: `dilate(src,element)−erode(src,element)`
* MORPH_TOPHAT 			: `src−open(src,element)`
* MORPH_BLACKHAT 		: `close(src,element)−src`
* MORPH_HITMISS 		: 只支持 CV_8UC1 binary images 

### 4.2.3. 通用自定义 Filtering

...

## 4.3. Geometric Image Transformations

存放了重要的几何变换函数, 包括最基础的 resize
* 一些更加泛用的操作函数在文档中没有放在这里, 而是放在了 core/operations on arrays 中
* 凡是需要意识到图像的, 都放在这里, 例如 resize 需要进行插值所以


### 4.3.1. InterpolationFlags

插值方法标志 


### 4.3.2. resize

更改图像的尺寸

```cpp
void cv::resize 	( 	InputArray  	src,
		OutputArray  	dst,
		Size  	dsize,
		double  	fx = 0,
		double  	fy = 0,
		int  	interpolation = INTER_LINEAR 
	) 		
Python:
	cv.resize(	src, dsize[, dst[, fx[, fy[, interpolation]]]]	) -> 	dst
```
* src 		: 原图像
* dsize 	: 输出的大小, 如果为0 (None in python), 则使用 fx, fy 来缩放
  * 注意输入是 (width, height)
  * 与 opencv-python 下 img.shape 得到的顺序相反
* fx, fy	: 代表 横, 纵 的缩放比例


### 4.3.3. Affine 和 Perspective

经典几何变换
* cv::getAffineTransform
* cv::warpAffine
* cv::getPerspectiveTransform 
* cv::warpPerspective

注意: Python 下, 传入的点坐标需要是 `numpy.array([4,2], dtype='float32')` 类型, 否则会报错

```cpp

// OpenCV 的 Affine 可以通过三个点对来计算投影矩阵
Mat cv::getAffineTransform 	( 	const Point2f  	src[],
		const Point2f  	dst[] 
	) 		
Python:
	cv.getAffineTransform(	src, dst	) -> 	retval

// 应用Affine变换
void cv::warpAffine 	( 	InputArray  	src,
		OutputArray  	dst,
		InputArray  	M,
		Size  	dsize, // 输出图像的解析度, 这里传入的是一个 pair
		int  	flags = INTER_LINEAR,
		int  	borderMode = BORDER_CONSTANT,
		const Scalar &  	borderValue = Scalar() 
	) 		
Python:
	cv.warpAffine(	src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]]	) -> 	dst

// OpenCV 的投影则是通过 4 个点对计算
Mat cv::getPerspectiveTransform 	( 	InputArray  	src,
		InputArray  	dst,
		int  	solveMethod = DECOMP_LU 
	) 		
Python:
	cv.getPerspectiveTransform(	src, dst[, solveMethod]	) -> 	retval

```



## 4.4. 杂项 Miscellaneous Image Transformations 

### 4.4.1. 二值化 threshold


#### 4.4.1.1. 阈值类型

```cpp
enum  	cv::ThresholdTypes {
  cv::THRESH_BINARY = 0,
  cv::THRESH_BINARY_INV = 1,
  cv::THRESH_TRUNC = 2,
  cv::THRESH_TOZERO = 3,
  cv::THRESH_TOZERO_INV = 4,
  cv::THRESH_MASK = 7,
  cv::THRESH_OTSU = 8,			// 目前只能应用在 8-bit 灰度图像
  cv::THRESH_TRIANGLE = 16 	 	// 目前只能应用在 8-bit 灰度图像
}

```

AdaptiveThresholdTypes

```cpp
enum  	cv::AdaptiveThresholdTypes {
  cv::ADAPTIVE_THRESH_MEAN_C = 0,
  cv::ADAPTIVE_THRESH_GAUSSIAN_C = 1
}

```
#### 4.4.1.2. threshold 和 自适应 adaptive


`double cv::threshold`
* 固定 level 的阈值函数
* 从一个 grayscale 获取一个 bi-level (二值) 图像

`cv::adaptiveThreshold`
* 应用自适应的 threshold 到一个数列上, 每个像素 (x,y) 的阈值都是单独通过 `T(x,y)` 计算的
* 从一个 grayscale 获取一个 bi-level (二值) 图像

函数原型  
```cpp
double cv::threshold 	( 	InputArray  	src,
		OutputArray  	dst,
		double  	thresh,   // 指定的阈值
		double  	maxval,
		int  	type 
	) 	
// Returns : the computed threshold value if Otsu's or Triangle methods used.

// 注意, python 中 dst 作为返回值是第二个, 直接用返回值接受的话 需要用两个变量接受 
Python:
	cv.threshold(	src, thresh, maxval, type[, dst]	) -> 	retval, dst


// 自动计算阈值
void cv::adaptiveThreshold 	( 	InputArray  	src,
		OutputArray  	dst,
		double  	maxValue,
		int  	adaptiveMethod,
		int  	thresholdType,
		int  	blockSize,
		double  	C 
	) 		
Python:
	cv.adaptiveThreshold(	src, maxValue, adaptiveMethod, thresholdType, blockSize, C[, dst]	) -> 	dst
```

参数:
* src	    : Source 8-bit single-channel image. 
* dst	    : Destination image of the `same size` and the `same type` as src. 
* thresh	: 指定的阈值
  * 如果使用了 OTSU 或者 TRIANGLE 阈值类型, 阈值会自动计算, 该输入置零即可
* maxval  : 像素值高于阈值后更新的像素值
* type/thresholdType : 查看 `cv::ThresholdTypes` 枚举类型
* 自适应参数
  * blockSize: 自适应函数要参照的区块大小, 该参数需要是 奇数
  * adaptiveMethod: 根据区块内的像素值来计算当前像素的阈值
    * `ADAPTIVE_THRESH_MEAN_C` : 区块内像素的均值
    * `ADAPTIVE_THRESH_GAUSSIAN_C ` : 区块内像素的高斯权重和



# 5. highgui  High-level GUI

独立于具体图像处理之外的 GUI 功能, 可以用来显示图像等调试功能

`#include <opencv2/highgui.hpp>`

## 5.1. imshow() 显示图片在窗口中

```cpp
void cv::imshow 	( 	const String &  	winname,
		InputArray  	mat 
	) 		
Python:
	cv.imshow(	winname, mat	) -> 	None
```

参数:
* winname  : Name of the window. 可以是提前创建好的. 如果没有提前创建, 则会应用 ` cv::WINDOW_AUTOSIZE`
* mat      : 要显示的图片

特点:
* 会根据图片规格的不同, 用对应的方式将像素值映射到 RGB888 里并显示

使用:
* 该函数必须后接一个 `cv::waitKey or cv::pollKey` 来保证窗口可操作
* `waitKey(0)` will display the window infinitely until any keypress

# 6. videoio Video I/O

除了面向视频文件的API, 与摄像头设备有关的接口也定义在了该模组中



## 6.1. OpenCV VideoIO 构成

视频文件的处理还算简单, 但是与摄像头的交互因为涉及到系统接口层面, 有很多底层内容

这里列出 OpenCV IO的构成方法
* 最底层- 源
  * 视频文件
  * 摄像头
  * 网络流
* 操作系统
  * 制造商库文件
  * Backends 后端库文件
  * CODECS(fourcc)
  * O.S. 库文件
* 中层
  * OpenCV API
    * VideoCaputre
    * VideoWriter
  * 其他制造商驱动 (C/C++ 等API)
* 用户软件

## 6.2. class VideoCaputre

不论是视频文件还是摄像头或者网络提供的图像流, 在OpenCV的上层接口里都被认为是相同的类型, 使用相同的类来处理  

### 6.2.1. 构造函数 VideoCapture () open()

构造函数, 共有五个重载, 同理在 python 中则是不同的输入形式  

* 空构造函数 `VideoCapture ()` 用于实现定义好一个对象, 之后再通过方法 `open()` 来打开设备
* `VideoCapture.open()` 的4个重载的参数和剩下的四个构造函数完全一致, 因此不赘述
* `open()` 也可以用来 reinitializes , 返回的是初始化是否成功


```cpp
cv::VideoCapture::VideoCapture 	( 		) 	
cv::VideoCapture::VideoCapture 	( 	
		const String &  	filename,
		int  	apiPreference = CAP_ANY 
	) 		
cv::VideoCapture::VideoCapture 	( 	
		const String &  	filename,
		int  	apiPreference,
		const std::vector< int > &  	params 
	)
cv::VideoCapture::VideoCapture 	( 	
		int  	index,
		int  	apiPreference = CAP_ANY 
	)
cv::VideoCapture::VideoCapture 	( 	
		int  	index,
		int  	apiPreference,
		const std::vector< int > &  	params 
	)

Python:
	cv.VideoCapture(		) -> 	<VideoCapture object>
	cv.VideoCapture(	filename[, apiPreference]	) -> 	<VideoCapture object>
	cv.VideoCapture(	filename, apiPreference, params	) -> 	<VideoCapture object>
	cv.VideoCapture(	index[, apiPreference]	) -> 	<VideoCapture object>
	cv.VideoCapture(	index, apiPreference, params	) -> 	<VideoCapture object>
```

参数:  
* filename		: 虽然参数名叫做 filename, 事实上也包含了网络流
  * 视频文件的地址 `video.avi`
  * 一连串的图片文件, 需要加入通配符 `img_%02d.jpg`, 代表了 `img_00.jpg` 开始的一系列图片
  * 视频流的URL	 `protocol://host:port/script_name?script_params|auth`
  * GStreamer pipeline string in gst-launch tool format 看不懂
* apiPreference : 可以从几个OpenCV实现的图像流 reader 中选择
  * 定义在了 `VideoCaptureAPIs` 中, 如果有需要再去单独参考
  * 默认是自动检测, 应该足够用了
  * OpenCV定义了几十种 reader , 针对了不同的设备
* params		: 用于附加的配置参数
  * 尽管该参数的接受类型是 `vector<int>`, 但实际传入的时候是键值对顺序排列的形式
    * `{key1,value1,key2,value2,...}`
  * 代码中的名称是 `VideoCaptureProperties` 
  * 该参数的配置内容都是非常底层的, 是否有效都是根据硬件的条件以及底层驱动相关的
* index			: 主要用于摄像头模式
  * 该参数用于选择当前计算机已连接的摄像头设备
  * 默认值0 , 代表跟随当前系统的默认摄像头


### 6.2.2. 读取帧

读取下一帧图像在 OpenCV 中也有多层实现
* `grab()` 	: 仅仅只是读取, 返回值是 bool, 代表读取是否成功
* `retrieve()`	: decode 刚才读取的帧
  * 注意返回值是解码是否成功, 图像信息在另外的参数
  * C++和 python 的返回形式不同
* `read()`	: 上述两个函数的结合, 抓取并解码
  * 

```cpp
virtual bool cv::VideoCapture::grab 	( 		) 	
virtual bool cv::VideoCapture::read 	( 	OutputArray  	image	) 	
virtual bool cv::VideoCapture::retrieve 	( 	OutputArray  	image,
		int  	flag = 0 
	) 		

Python:
	cv.VideoCapture.grab(		) -> 	retval
	cv.VideoCapture.retrieve(	[, image[, flag]]	) -> 	retval, image
	cv.VideoCapture.read(	[, image]	) -> 	retval, image
```


### 6.2.3. 配置参数管理

* `get()` 方法不是取帧, 而是获取对应的配置参数
* `set()` 用于设置配置参数

```cpp
virtual double cv::VideoCapture::get 	( 	int  	propId	) 	const

Python:
	cv.VideoCapture.get(	propId	) -> 	retval

virtual bool cv::VideoCapture::set 	( 	int  	propId,
		double  	value 
	) 		
Python:
	cv.VideoCapture.set(	propId, value	) -> 	retval
```


## 6.3. class VideoWriter

Video writer class.  
The class provides C++ API for writing video files or image sequences. 



Default constructors.  
The constructors/functions initialize video writers.

    On Linux FFMPEG is used to write videos;
    On Windows FFMPEG or MSWF or DSHOW is used;
    On MacOSX AVFoundation is used.


### 6.3.1. 构造函数  open()

类的构造函数  
* 同 VideoCapture 的形式一样, 通过空构造函数预先定义对象, 再通过 `open()` 进行初始化
* `open()` 方法的参数也是完全一样, 因此不赘述, 返回的是初始化是否成功
* `open()` 也可以用来重新初始化一个对象 reinitializes 

```cpp
cv::VideoWriter::VideoWriter 	( 		) 	
cv::VideoWriter::VideoWriter 	( 	const String &  	filename,
		int  	fourcc,
		double  	fps,
		Size  	frameSize,
		bool  	isColor = true 
	) 	
cv::VideoWriter::VideoWriter 	( 	const String &  	filename,
		int  	apiPreference,
		int  	fourcc,
		double  	fps,
		Size  	frameSize,
		bool  	isColor = true 
	) 	
cv::VideoWriter::VideoWriter 	( 	const String &  	filename,
		int  	fourcc,
		double  	fps,
		const Size &  	frameSize,
		const std::vector< int > &  	params 
	) 	
cv::VideoWriter::VideoWriter 	( 	const String &  	filename,
		int  	apiPreference,
		int  	fourcc,
		double  	fps,
		const Size &  	frameSize,
		const std::vector< int > &  	params 
	) 	


Python:
	cv.VideoWriter(		) -> 	<VideoWriter object>
	cv.VideoWriter(	filename, fourcc, fps, frameSize[, isColor]	) -> 	<VideoWriter object>
	cv.VideoWriter(	filename, apiPreference, fourcc, fps, frameSize[, isColor]	) -> 	<VideoWriter object>
	cv.VideoWriter(	filename, fourcc, fps, frameSize, params	) -> 	<VideoWriter object>
	cv.VideoWriter(	filename, apiPreference, fourcc, fps, frameSize, params	) -> 	<VideoWriter object>
```

参数:
* filename		: Name of the output video file.
  * 该类除了老老实实的写入视频之外, 还可以以一连串图片的格式输出 video, 类似于与 VideoCapture 相对应
    * filename 带通配符, eg. `img_%02d.jpg` `img_%02d.BMP`
    * `fourcc=0` or `fps=0`
* fourcc		: Four Character Code, 用于指代视频流压缩方法
  * fource=-1 代表通过系统诊断来自动选择方法
* fps			: 视频帧率
* frameSize		: 视频的分辨率
* isColor		: 如果该值为 false, 则输出是灰度影片
* apiPreference	: 同 VideoCapture
* params 		: 同 VideoCapture


### 6.3.2. 写入 write

```cpp
virtual void cv::VideoWriter::write 	( 	InputArray  	image	) 	

Python:
	cv.VideoWriter.write(	image	) -> 	None
```

# 7. video Video Analysis

涉及到视频的分析在OpenCV中实装了两个方向
* Motion Analysis 动作(运动)分析
* Object Tracking 物体追踪

# 8. objdetect Object Detection 最常用的物体检测模型

## 8.1. Cascade 模型 Cascade Classifier for Object Detection

一个经典的级联传统机器学习模型

`class  	cv::CascadeClassifier`  

官方示例: https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html

预训练模型的下载地址 : https://github.com/opencv/opencv/blob/4.x/samples/python/tutorial_code/objectDetection/cascade_classifier/objectDetection.py



# 9. photo Computational Photography 计算图像处理

包括了几个基于计算的图像处理领域的算法实现

该模组下实现的算法列表, 方便后续单独查找学习:
* inpainting
  * Navier-Stokes based method
  * Alexandru Telea
    * Alexandru Telea. An image inpainting technique based on the fast marching method. Journal of graphics tools, 9(1):23–34, 2004.


## 9.1. inpainting

图像补全, 该分类下只有一个函数


```cpp
void cv::inpaint 	( 	InputArray  	src,
		InputArray  	inpaintMask,
		OutputArray  	dst,
		double  	inpaintRadius,
		int  	flags 
	) 		
Python:
	cv.inpaint(	src, inpaintMask, inpaintRadius, flags[, dst]	) -> 	dst
```

参数介绍:
* src 			: 8-bit 3-channel image, or 8-bit, 16-bit unsigned or 32-bit float 1-channel
* inpaintMask 	: 用于标记哪些像素要被补全 
  * 8-bit 1-channel image.
  * Non-zero pixels indicate the area that needs to be inpainted.
* dst			: Output image with the same size and type as src . 
* inpaintRadius	: Radius of a circular neighborhood of each point inpainted that is considered by the algorithm.
  * 补全算法的响应半径
* flags			: 补全算法选择
  * `cv::INPAINT_NS`
  * `cv::INPAINT_TELEA`
