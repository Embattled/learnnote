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


Extra modules: 拓展模组
* `ximgproc`. 		Extended Image Processing

按照 main modules 为一级标题来管理该笔记

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

安装:
* `pip install opencv-python` 
* `pip install opencv-contrib-python`

依赖库:
* Numpy. OpenCV-Python makes use of Numpy. And OpenCV-Python requires only Numpy.
* All the OpenCV array structures are converted to and from Numpy arrays. 
* This also makes it easier to integrate with other libraries that use Numpy. such as `SciPy` and `Matplotlib`


opencv-python 的 API 都定义在了 cv2 包中, 为了保证代码的可执行性  
`import cv2 as cv`  常用于opencv 包的导入



# 2. core

定义了 OpenCV 中最核心的类 (图像), 以及一些对于 array 的数学上的操作 
* 有些图像处理的功能可能没有定义在图像部分, 而是在 array 对象的更原始的层面里实现了

## 2.1. Basic structures OpenCV 的基础数据结构

### 2.1.1. mat  core/mat.hpp

同 numpy 一样, opencv并没有特地的图片类, 而是用矩阵来表示 -> `cv::Mat`  


### 2.1.2. cv::KeyPoint

用于保存显著点的数据结构 (salient point)

存储通过关键点检测器找到的点特征, 找到的点会通过 descriptor 来分析其邻域并最终生成特征    


```cpp
// float cv::KeyPoint::angle
// 一些特征点本身会有方向情报, 该值存储的为 [0,360) 的角度信息. 以 image coordinate system , 即 clockwise
float angle;

// int cv::KeyPoint::class_id
// object class (if the keypoints need to be clustered by an object they belong to)
// 关键点本身可能有不同的种类
int class_id;
 
// octave (pyramid layer) from which the keypoint has been extracted.
// 表示该特征点是在图像金字塔的哪一层 (哪个尺度) 下被采集到的, 表示该特征点的层数
int octave;

// Point2f cv::KeyPoint::pt
// coordinates of the keypoints More...
Point2f pt;
 
// float cv::KeyPoint::response
// the response by which the most strong keypoints have been selected. Can be used for the further sorting or subsampling.
// 代表该特征点的相应强度, 可能被用来进行特征点筛选
float response;

// float cv::KeyPoint::size
// diameter of the meaningful keypoint neighborhood.
// 一些检测算法可能会包含特征点的邻域半径大小
float size;


```

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

### 2.2.2. mixChannels 通道重排列

`mixChannels()`  将 src 指定的通道拷贝到 dst 指定的通道里 
* 非常 tools 的函数
* 很多其他函数就是基于该函数实现的 partial cases of `cv::mixChannels`.
  * `cv::split`
  * `cv::merge`
  * `cv::extractChannel`
  * `cv::insertChannel`
  * some forms of `cv::cvtColor`  

* `fromTo` : 一个一维数列, 按着拷贝顺序存储通道index, `s1,d1,s2,d2...`


# 3. imgproc  Image Processing

`#include <opencv2/imgproc.hpp>`  

对读取的图像进行处理

注意:
* 对于图形处理函数
  * C++ 的 dst 一般都在参数里
  * Python 的 dst 可以是参数也可以是返回值

全局的 enum 常量
* cv::InterpolationFlags	: 定义插值方法
* cv::InterpolationMasks 	: 不懂
* cv::WarpPolarMode			: 不懂


## 3.1. Image Filtering 滤波函数

用于对 2D 图像进行各种 线性/非线性 的filtering 操作
* 对于图像中的每个像素 (x,y), 通过其周围像素的值来决定该像素新的值
* 通道之间将会分开来计算, 即可以保证输出图像具有和输入图像相同的通道数


通用参数
* src
* dst
* ddepth
`when ddepth=-1, the output image will have the same depth as the source. `

### 3.1.1. Smoothing Filtering

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


* `cv::boxFilter()` : 均值盒滤波
  * call `blur(src, dst, ksize, anchor, borderType)` 
  * is equivalent to `boxFilter(src, dst, src.type(), ksize, anchor, true, borderType)`.
  * 其中的参数 `bool normalize` 用于指定滤波核是否经过归一化, 如果是 true 的话就等同于 blur
  * 如果 normalize 是 false 的话, 相当于计算了各个位置上的 窗口核, 主要用于计算其他 Filter 的中间量
  * 因此该函数主要用于辅助别的函数实现

### 3.1.2. Morphological Transformation 形态学变化

OpenCV 的形态学变化也放在了 Filtering 模块中

通用参数:
* src		: input image, 支持任意 channel
* dst		: (大概是C++用的)output image of the same size and type as src.
* kernel	: structuring element. 可以用函数 `getStructuringElement()` 创建
* anchor	: 应用 kernel 的锚点, 默认值(-1,-1)不代表任何具体坐标, 会被转换成 kernel 的中心
* iterations: 可以直接在函数参数中设置重复执行多次  


#### 3.1.2.1. 基础 膨胀腐蚀

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

#### 3.1.2.2. 高级组合操作

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

### 3.1.3. edge detection

OpenCV实现了如下几种边缘检测算子:
* Sobel		: Kernel 是 1 2 1
* Scharr	: sober 的 kernel 改进, 为 3 10 3, 参数与 Sober 完全一致
* Laplacian : 由二阶导数来计算梯度
* Canny		: 细致的完整边缘检测算法
  * 1. 平滑图像, 使用高斯滤波
  * 2. 计算梯度和 edge , 使用 Sober
  * 3. Non-maximum suppression, 移除非线条的Sober结果
  * 4. 阈值

```cpp
void cv::Sobel 	(
	 	InputArray  	src,
		OutputArray  	dst,
		int  	ddepth,
		int  	dx,
		int  	dy,
		int  	ksize = 3,
		double  	scale = 1,
		double  	delta = 0,
		int  	borderType = BORDER_DEFAULT 
	)
void cv::Scharr	(Same with above)
void cv::Laplacian 	(
	 	InputArray  	src,
		OutputArray  	dst,
		int  	ddepth,
		int  	ksize = 1,
		double  	scale = 1,
		double  	delta = 0,
		int  	borderType = BORDER_DEFAULT 
	) 		
// 完整的 Canny 
void cv::Canny 	( 	
		InputArray  	image,
		OutputArray  	edges,
		double  	threshold1,
		double  	threshold2,
		int  	apertureSize = 3,
		bool  	L2gradient = false 
	) 	
// 使用经过其他途径计算的 x, y 方向梯度
void cv::Canny 	( 	
		InputArray  	dx,
		InputArray  	dy,
		OutputArray  	edges,
		double  	threshold1,
		double  	threshold2,
		bool  	L2gradient = false 
	) 	

Python:
	cv.Scharr(	src, ddepth, dx, dy[, dst[, scale[, delta[, borderType]]]]	) -> 	dst
	cv.Sobel(	src, ddepth, dx, dy[, dst[, ksize[, scale[, delta[, borderType]]]]]	) -> 	dst
	cv.Laplacian(	src, ddepth[, dst[, ksize[, scale[, delta[, borderType]]]]]	) -> 	dst


	cv.Canny(	image, threshold1, threshold2[, edges[, apertureSize[, L2gradient]]]	) -> 	edges
	cv.Canny(	dx, dy, threshold1, threshold2[, edges[, L2gradient]]	) -> 	edges
```

### 3.1.4. 通用自定义 Filtering

...



## 3.2. Geometric Image Transformations

存放了重要的几何变换函数, 包括最基础的 resize
* 一些更加泛用的操作函数在文档中没有放在这里, 而是放在了 core/operations on arrays 中
* 凡是需要意识到图像的, 都放在这里, 例如 resize 需要进行插值所以


### 3.2.1. Flags

插值方法标志 - InterpolationFlags

```cpp
enum  	cv::InterpolationFlags {
  cv::INTER_NEAREST = 0,
  cv::INTER_LINEAR = 1,
  cv::INTER_CUBIC = 2,
  cv::INTER_AREA = 3,
  cv::INTER_LANCZOS4 = 4,
  cv::INTER_LINEAR_EXACT = 5,
  cv::INTER_NEAREST_EXACT = 6,
  cv::INTER_MAX = 7,
  cv::WARP_FILL_OUTLIERS = 8,
  cv::WARP_INVERSE_MAP = 16
}


```






### 3.2.2. resize

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


### 3.2.3. Affine 和 Perspective

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


### remap - 畸变矫正的基础

```cpp
void cv::remap 	( 	
		InputArray  	src,
		OutputArray  	dst,
		InputArray  	map1,
		InputArray  	map2,
		int  	interpolation,
		int  	borderMode = BORDER_CONSTANT,
		const Scalar &  	borderValue = Scalar() 
	) 		
Python:
	cv.remap(	src, map1, map2, interpolation[, dst[, borderMode[, borderValue]]]	) -> 	dst
```

对图像应用通用几何变换, 用两个 map 来表示两个图像之间的坐标对应
$$dst(x,y) = src(map_x(x,y), map_y(x,y))$$

dst 的大小取决于 map 的大小, 对应的想获取多大的 dst 就要提供多大的 map





## 3.3. 色彩空间转换 Color Space Conversions

opencv 支持近乎所有的图形格式之间·`互相`转换  

### 3.3.1. cvtColor

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
* src,dst	: 输入输出图像
* code	: 转换模式
* dstCn	: dst图像的通道数. 一般置零用来自动判定

注意:
* 该函数是最顶端的接口, 即支持任意形式的图像转换
* 该函数是(部分)支持输入图像是小数格式的 (0,1) , 在默写转换模式下甚至推荐使用小数格式
  * 对于 Luv 转换 e.g. COLOR_BGR2Luv , 是应该使用标准化来将图像放到 (0,1) 中的
  * 对于 Demosaic 转换, 即 Bayer 的转换只支持整数类型 `Assertion failed) depth == CV_8U || depth == CV_16U in function 'demosaicing'`
* OpenCV 只支持32 bit 小数, 即单精度, 因此对于 numpy 来说不要使用 float64
* 通过阅读源码, 发现尽管ColorConversionCodes 里有Bayer2RGB的模式, 但是函数的源代码并没有对应的分支, 通过阅读头文件, 发现 Bayer2RGB 的模式通过等效变换成了 2BGR 的格式来进行处理  


### 3.3.2. demosaicing

专门用于 CFA 图像的去马赛克函数, 属于 cvtColor 的一部分功能独立出来的接口

```c++
void cv::demosaicing 	( 	InputArray  	src,
		OutputArray  	dst,
		int  	code,
		int  	dstCn = 0 
	) 		
Python:
	cv.demosaicing(	src, code[, dst[, dstCn]]	) -> 	dst
```

对比于 cvtColor , 所支持的 code 被限制为一部分, 可以分为四类
* Demosaicing using bilinear interpolation
  * Bayer2BGR
  * Bayer2Gray
* Demosaicing using Variable Number of Gradients.
  * Bayer2BGR_VNG
* Edge-Aware Demosaicing.
  * Bayer2BGR_EA
* Demosaicing with alpha channel
  * Bayer2BGRA
* 通过阅读源码, 确认了该函数支持直接转为 RGB 的各种 code


### 3.3.3. ColorConversionCodes

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

## 3.4. 色彩图 - ColorMaps in OpenCV

专门用于对 灰度图 进行可视化增强的灰度图映射函数

## 3.5. 杂项 Miscellaneous Image Transformations 

### 3.5.1. 二值化 threshold


#### 3.5.1.1. 阈值类型

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
#### 3.5.1.2. threshold 和 自适应 adaptive


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


# 4. imgcodecs  Image file reading and writing 

同 Image Process 不同, 该模组中的函数都是图像 IO 相关的  

`#include <opencv2/imgcodecs.hpp>` 

## 4.1. Flags used for image file reading and writing

用作图片 IO 的参数, 一般都是枚举类型, 用于指定图片的规格等

* 在C++下, OpenCV 会以内建的 Mat 类来处理图像
* 在 Python OpenCV 中则使用 numpy.ndarry 来处理, 因此 Mat 类的方法在 Python 下不能使用

### 4.1.1. ImreadModes

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

## 4.2. 基础图像读写

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



# 5. videoio Video I/O

除了面向视频文件的API, 与摄像头设备有关的接口也定义在了该模组中



## 5.1. OpenCV VideoIO 构成

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

# 6. highgui  High-level GUI

独立于具体图像处理之外的 GUI 功能, 可以用来显示图像等调试功能
* 用于快速尝试功能并可视化结果
* 用于直接基于 OpenCV 开发完整的应用程序

由于窗口环境属于比较特殊的环境, 因此是操作系统, 开发平台高度相关的, 大概有以下几种情况
* OpenGL
* Qt
* WinRT

`#include <opencv2/highgui.hpp>`

## 6.1. 键盘响应

* waitKey : Waits for a pressed key. 
  * delay : Delay in milliseconds. 如果小于等于0 意为无限期等待
  * 因为OS有最小线程切换时间, 所以该时间不会特别准确,  it will wait at least delay ms
  * returns:
    * the code of the pressed key
    * `-1` if no key was pressed before the specified time had elapsed
* waitKeyEx :
  * Similar to `waitKey`, but returns `full key code`. 
* pollKey : Polls for a pressed key. 
  * check for a key press but not wait for it.
  * The function pollKey polls for a key event without waiting.
  * returns :
    * the code of the pressed key
    * -1 if no key was pressed since the last invocation. 

键盘处理的两个函数是:
* only methods in HighGUI that can fetch and handle GUI events
* one of them needs to be called periodically for normal event processing
  * unless HighGUI is used within an environment that takes care of event processing.
* The function only works if there is at least one HighGUI window created and the window is active. 
* If there are several HighGUI windows, any of them can be active.


```cpp
int cv::waitKey 	( 	int  	delay = 0	) 	
int cv::waitKeyEx 	( 	int  	delay = 0	) 	
int cv::pollKey 	( 		) 	
Python:
	cv.waitKey(	[, delay]	) -> 	retval
	cv.waitKeyEx(	[, delay]	) -> 	retval
	cv.pollKey(		) -> 	retval
```

## 6.2. 简易窗口

### 6.2.1. imshow() 显示图片在窗口中

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

## 6.3. windows 操作

1. 创建窗口 `cv.namedWindow(winname, flags)`
	* as a placeholder for images and trackbars
	* windows are referred to by their names.
	* the function does nothing when a window with the same name already exists.
2. 移动窗口 `cv.moveWindow(winname, x, y)`
	* 无任何特别, x y 是坐标
3. 重命名窗口 `cv.setWindowTitle(winname, title)`
	* Updates window title. 
	* 并不会更改窗口的 name, 即各种操作的引用名字
4. 修改窗口属性 `cv::setWindowProperty(winname, key, value)`
	* 动态修改窗口的属性
	* key : `cv::WindowPropertyFlags`
	* value: `cv::WindowFlags` 同创建窗口时候的 flags 为同一枚举类型
5. 修改窗口大小
	* `cv.resizeWindow(winname, width, height)`
	* `cv.resizeWindow(winname, size)`
	* 只有窗口不是 `cv::WINDOW_AUTOSIZE` 才能调整
	* 只能调整图像的区域, Toolbars 区域不受影响
6. 手动关闭窗口 
	* `cv::destroyWindow()`
	* `cv::destroyAllWindows()`

完整函数原型
```cpp
void cv::namedWindow 	(
	const String &  	winname,
	int  	flags = WINDOW_AUTOSIZE 
	)
// Winname	: Name of the window in the window caption that may be used as a window identifier. 
// flags	: Flags of the window.  (cv::WindowFlags) 

void cv::setWindowTitle 	( 	
	const String &  	winname,
	const String &  	title 
	) 	

	
void cv::setWindowProperty 	( 	
	const String &  	winname,
	int  	prop_id,
	double  	prop_value 
	) 		
// prop_id	: cv::WindowPropertyFlags
// prop_value: New value of the window property.cv::WindowFlags


Python:

cv.namedWindow(	winname[, flags]	) -> 	None
cv.setWindowTitle(	winname, title	) -> 	None
cv.setWindowProperty(	winname, prop_id, prop_value	) -> 	None
```
### 6.3.1. cv::WindowFlags 创建窗口时候的 flag

Flags for cv::namedWindow  枚举类型

| 代码                | 功能                                             |
| ------------------- | ------------------------------------------------ |
| WINDOW_NORMAL       | 用户可以自己调整窗口大小                         |
| WINDOW_AUTOSIZE     | 用户不能控制窗口大小, 根据图片的大小自动调整窗口 |
| WINDOW_OPENGL       | OpenGL 后端的窗口                                |
| WINDOW_FULLSCREEN   | 全屏幕窗口                                       |
| WINDOW_FREERATIO    | 自由比例                                         |
| WINDOW_KEEPRATIO    | 固定比例                                         |
| WINDOW_GUI_EXPANDED | status bar and tool bar                          |


## 6.4. Trackbar 


1. 创建  `cv::createTrackbar()`
2. 修改属性:
	* `cv::setTrackbarMax()`
	* `cv::setTrackbarMin()`
	* `cv::setTrackbarPos()`
3. 获取值 ` getTrackbarPos()`


函数原型
```cpp
int cv::createTrackbar 	( 	
	// trackbar的名称
	const String &  	trackbarname,
	// 要添加到的窗口, 如果是 Qt 后端, 则可以为空
	const String &  	winname,
	// 用以存储滑条值的变量
	int *  	value,
	// 滑条的最大值, 滑条的最小值0且无法在初始时更改
	int  	count,
	// 函数指针, 用以每次滑块的值被更改时自动进行回调
	// 函数原型必须是 void Foo(int,void*); 
	// 第一个值是 value , 第二个值是 userdata
	TrackbarCallback  	onChange = 0,
	// 用以方便操作 trackBar 行为的一个用户自定义变量
	// 使用该变量可以省去通过全局变量来调控回调函数的行为
	// Python 下没有这个参数所以无法使用
	void *  	userdata = 0 
	) 	

void cv::setTrackbarMin 	( 	const String &  	trackbarname,
		const String &  	winname,
		int  	minval 
	)
void cv::setTrackbarMax 	( 	const String &  	trackbarname,
		const String &  	winname,
		int  	maxval 
	) 		
void cv::setTrackbarPos 	( 	const String &  	trackbarname,
		const String &  	winname,
		int  	pos 
	) 	
int cv::getTrackbarPos 	( 	const String &  	trackbarname,
		const String &  	winname 
	) 		

Python:
	cv.setTrackbarMin(	trackbarname, winname, minval	) -> 	None
	cv.setTrackbarMax(	trackbarname, winname, maxval	) -> 	None
	cv.setTrackbarPos(	trackbarname, winname, pos	) -> 	None
	cv.getTrackbarPos(	trackbarname, winname	) -> 	retval	

```

## 6.5. mouse 鼠标事件

OpenCV GUI模块也提供了和鼠标的交互, 相比于 trackBar 等模块, 鼠标的管理函数更少, 但是对应的 flags 更复杂

* getMouseWheelDelta()	: Gets the mouse-wheel motion delta. 鼠标滚轮
* setMouseCallback()	: 设置鼠标的回调函数

```cpp
// flag The mouse callback flags parameter. 
int cv::getMouseWheelDelta 	( 	int  	flags	) 	

// 设置鼠标事件回调函数
// 回调函数的原型则必须是 : 
// void(* cv::MouseCallback) (int event, int x, int y, int flags, void *userdata)
// 			event  : cv::MouseEventTypes constants
// 			flags  : cv::MouseEventFlags constants
void cv::setMouseCallback 	( 	
		const String &  	winname,
		MouseCallback  	onMouse,
		void *  	userdata = 0 
	) 	
```

鼠标事件 flags :
* `enum cv::MouseEventTypes`
  * EVENT_MOUSEMOVE 		: pointer has moved over the window. 
  * 单击按下
  * EVENT_LBUTTONDOWN 		: left mouse button is pressed. 
  * EVENT_RBUTTONDOWN 		: right mouse button is pressed.
  * EVENT_MBUTTONDOWN 		: middle mouse button is pressed. 
  * 双击
  * EVENT_LBUTTONDBLCLK 	: left mouse button is double clicked. 
  * EVENT_RBUTTONDBLCLK 	: right mouse button is double clicked. 
  * EVENT_MBUTTONDBLCLK 	: middle mouse button is double clicked.   
  * 松开
  * EVENT_LBUTTONUP 		: left mouse button is released. 
  * EVENT_RBUTTONUP 		: right mouse button is released.
  * EVENT_MBUTTONUP 		: middle mouse button is released. 
  * 滚轮
  * EVENT_MOUSEWHEEL 		: positive and negative values mean forward and backward scrolling
  * EVENT_MOUSEHWHEEL 		: positive and negative values mean right and left scrolling
* `enum cv::MouseEventFlags`: 代表鼠标操作时候的其他特殊 flags (多键组合)
  * EVENT_FLAG_LBUTTON		: left mouse button is down
  * EVENT_FLAG_RBUTTON		: right mouse button is down
  * EVENT_FLAG_MBUTTON		: middle mouse button is down
  * EVENT_FLAG_CTRLKEY 		: CTRL Key is pressed
  * EVENT_FLAG_SHIFTKEY 	: SHIFT Key is pressed.
  * EVENT_FLAG_ALTKEY 		: ALT Key is pressed


## 6.6. class VideoCaputre

不论是视频文件还是摄像头或者网络提供的图像流, 在OpenCV的上层接口里都被认为是相同的类型, 使用相同的类来处理  

### 6.6.1. 构造函数 VideoCapture () open()

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


### 6.6.2. 读取帧

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


### 6.6.3. 配置参数管理

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


## 6.7. class VideoWriter

Video writer class.  
The class provides C++ API for writing video files or image sequences. 



Default constructors.  
The constructors/functions initialize video writers.

    On Linux FFMPEG is used to write videos;
    On Windows FFMPEG or MSWF or DSHOW is used;
    On MacOSX AVFoundation is used.


### 6.7.1. 构造函数  open()

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


### 6.7.2. 写入 write

```cpp
virtual void cv::VideoWriter::write 	( 	InputArray  	image	) 	

Python:
	cv.VideoWriter.write(	image	) -> 	None
```

# 7. video Video Analysis

涉及到视频的分析在OpenCV中实装了两个方向
* Motion Analysis 动作(运动)分析
* Object Tracking 物体追踪

# 8. calib3d - Camera Calibration and 3D Reconstruction

涉及 3D 重构和相机标定的主要模组  

目前实现了普通的 pinhole 单目, 多目
* 在 主包中的 fisheye 命名空间下实现了鱼眼相机的模型
* 在 追加包 ccalib3d 下实现了 omnidir 相机的模型



## 8.1. camera, fisheye, omnidir 相机标定相关

两种独特相机的专用函数接口名称是相同的

### 8.1.1. calibrate - 相机标定

```cpp
double cv::omnidir::calibrate 	( 	
		InputArrayOfArrays  	objectPoints,
		InputArrayOfArrays  	imagePoints,
		Size  	size,
		InputOutputArray  	K,
		InputOutputArray  	xi,
		InputOutputArray  	D,
		OutputArrayOfArrays  	rvecs,
		OutputArrayOfArrays  	tvecs,
		int  	flags,
		TermCriteria  	criteria,
		OutputArray  	idx = noArray() 
	) 		

double cv::fisheye::calibrate 	( 	
		InputArrayOfArrays  	objectPoints,
		InputArrayOfArrays  	imagePoints,
		const Size &  	image_size,
		InputOutputArray  	K,
		InputOutputArray  	D,
		OutputArrayOfArrays  	rvecs,
		OutputArrayOfArrays  	tvecs,
		int  	flags = 0,
		TermCriteria  	criteria = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 100, DBL_EPSILON) 
	) 	

Python:
	cv.omnidir.calibrate(
		objectPoints, imagePoints, size, K, xi, D, flags, criteria[, rvecs[, tvecs[, idx]]]	) ->
		 	retval, K, xi, D, rvecs, tvecs, idx
			
	cv.fisheye.calibrate(	
		objectPoints, imagePoints, image_size, K, D[, rvecs[, tvecs[, flags[, criteria]]]]	) ->
		 	retval, K, D, rvecs, tvecs

```

执行相机标定

参数:
* InputArrayOfArrays	: objectPoints, (?) 从其他函数获取的 calibration pattern 的 points 世界坐标
  * `Vector of vector of Vec3f`, `Mat with size 1xN/Nx1 and type CV_32FC3`
* InputArrayOfArrays	: imagePoints, 与 objectPoints 长度一致, 对应的 calibration pattern points 的图像坐标
* Size					: size, image_size,  calibration image 的图像大小, 仅仅用来初始化内部参数矩阵
* InputOutputArray K	: 标准相机内部参数矩阵. fx,fy,cx,cy 等, 如果是使用 GUESS 方法来标定, 则传入的矩阵本身应该赋予了对应的 GUESS 值
* InputOutputArray D 	: 相机扭曲参数, 对于 omnidir 是 k1,k2,p1,p2, 对于 opencv fisheye 是 k1,k2,k3,k4
* InputOutputArray xi 	: omni 相机所独有的单位球球心和投影原点的距离
* TermCriteria criteria	: 用于标定算法的迭代终止控制


### 8.1.2. undistort - 逆畸变

用于反向计算相机的畸变过程, 获得畸变前的图像/点的坐标


```cpp

void cv::omnidir::undistortImage 	(
	 	InputArray  	distorted,
		OutputArray  	undistorted,
		InputArray  	K,
		InputArray  	D,
		InputArray  	xi,
		int  	flags,
		InputArray  	Knew = cv::noArray(),
		const Size &  	new_size = Size(),
		InputArray  	R = Mat::eye(3, 3, CV_64F) 
	) 		
void cv::fisheye::undistortImage 	(
		InputArray  	distorted,
		OutputArray  	undistorted,
		InputArray  	K,
		InputArray  	D,
		InputArray  	Knew = cv::noArray(),
		const Size &  	new_size = Size() 
	) 
		
Python:
	cv.omnidir.undistortImage(
			distorted, K, D, xi, flags[, undistorted[, Knew[, new_size[, R]]]]	) -> 	undistorted
	cv.fisheye.undistortImage(
			distorted, K, D[, undistorted[, Knew[, new_size]]]	) -> 	undistorted
```

将一张有畸变的图片反变换补偿为没有畸变的图片, 该函数是一个懒人函数, 就是 `initUndistorRectifyMap` 和 `remap` 的结合而已  

从结果上, 如果原本的相机是鱼眼摄像的, 那么用 fisheye 下的 undistort 能有更好的效果

参数:
* K, D, xi 不再赘述  : 根据相机模型不同而内容不同, 分别是 k1,k2,p1,p2 和 k1,k2,k3,k4
* distorted			: 即输入图像, 
* newsize  			: 反畸变的输出图像大小, 默认和输入相同
* Knew				: 输出的矫正的图像内部参数, 默认为 单位矩阵, 某些时候仍然需要自定义
* 对于 omnidir 的接口, 还附加的实现了
  * R : 输出图像和 输入图像的旋转矩阵
  * flags : 用于调整函数行为, 在 omnidir 的接口中独有而且是必须参数  



```cpp

void cv::omnidir::undistortPoints 	(
	 	InputArray  	distorted,
		OutputArray  	undistorted,
		InputArray  	K,
		InputArray  	D,
		InputArray  	xi,
		InputArray  	R 
	) 		
void cv::fisheye::undistortPoints 	(
	 	InputArray  	distorted,
		OutputArray  	undistorted,
		InputArray  	K,
		InputArray  	D,
		InputArray  	R = noArray(),
		InputArray  	P = noArray(),
		TermCriteria  	criteria = TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, 10, 1e-8) 
	) 		

Python:
	cv.omnidir.undistortPoints(	distorted, K, D, xi, R[, undistorted]	) -> 	undistorted
Python:
	cv.fisheye.undistortPoints(	distorted, K, D[, undistorted[, R[, P[, criteria]]]]	) -> 	undistorted

```

参数
* 输入输出格式和 image 模式相同, 都是 Array, 但不在是图像而是点的序列 `vector of Vec2f`
  * or `1xN/Nx1 2-channel Mat of type CV_32F, 64F depth is also acceptable`
* K, D, xi : 相机参数
* R : 旋转矩阵, 3x3
* P : fisheye 实现了传入新相机的内部参数 3x3 或者整个 projection matrix 3x4


## stereo 双目相机

* cv::stereoCalibrate
* cv::stereoRectify
* cv::stereoRectifyUncalibrated


## PnP 求解

全局指导: 
`Perspective-n-Point (PnP) pose computation`  
求解 PnP 问题  
* cv::solveP3P
* cv::solvePnP
* cv::solvePnPGeneric	: 检索 所有可能的方案  
* cv::solvePnPRansac	: 使用 RANSAC 来处理异常值
* Pose refinement: 姿态细化, 使用非线性最小化方法, 从解的初始值开始估计旋转和平移, 最小化重投影误差
  * cv::solvePnPRefineLM
  * cv::solvePnPRefineVVS

`enum cv::SolvePnPMethod`  




通用参数:
* `objectPoints`	: 空间点坐标数组, 传入 (N, 3) 或者 cv 里的 `vector<Point3d>`
* `imagePoints`		: 空间点对应的图像点的数组, (N, 2), 传入 cv `vector<Point2d>`
* `cameraMatrix`	: 提供预定义的相机内参 `3x3` 矩阵, 包括 主点信息和焦距信息共4个值
* `distCoeffs`		: 相机的 扭曲系数, 可以是 `4, 5, 8, 12 or 14` 长度, 需要参考 OpenCV 的扭曲模型
* `iterationsCount`	: 迭代求解的迭代次数  
* `reprojectionError`	: 求解过程中的距离阈值, 用于管理某个点是否被视作 `inlier`
* 
* `flags`			: 求解 PnP 的解法标志





## Rodrigues - 旋转矩阵

将旋转矩阵转为旋转向量, 或者执行相反操作.  



## 三角标定 

* cv::triangulatePoints


## 验证函数 

* cv::validateDisparity


# 9. features2d - 2D Features Framework 2D图像的传统特征检测

## 9.1. Feature Detection and Description - 特征检测和描述

基本上有名的特征检测都在 OpenCV 中实现了
而且为了实现实例复用, 所有特征检测器都是以 class 来实现的, 这点和 GIF 类似

这里包括了 Detection 和 Description  

### 9.1.1. Detection 简易函数

关键点检测部分包含了两个算法, 以函数形式实现 
函数实现的接口不存在对应的 Python API

```cpp
void cv::FAST 	( 	InputArray  	image,
		std::vector< KeyPoint > &  	keypoints,
		int  	threshold,
		bool  	nonmaxSuppression,
		FastFeatureDetector::DetectorType  	type 
	) 	
// threshold 为关键点检测的响应阈值
// bool  	nonmaxSuppression 代表是否对关键点采用 局部非最大值抑制
// FastFeatureDetector::DetectorType  	type  原论文中提供了三种 邻域的定义, 这里可选

void cv::FAST 	( 	InputArray  	image,
		std::vector< KeyPoint > &  	keypoints,
		int  	threshold,
		bool  	nonmaxSuppression = true 
	) 	
// 快速调用的重载, 减少了必要参数

// AGAST
void cv::AGAST 	( 	InputArray  	image,
		std::vector< KeyPoint > &  	keypoints,
		int  	threshold,
		bool  	nonmaxSuppression,
		AgastFeatureDetector::DetectorType  	type 
	) 	
// 参数的意思基本相同

void cv::AGAST 	( 	InputArray  	image,
		std::vector< KeyPoint > &  	keypoints,
		int  	threshold,
		bool  	nonmaxSuppression = true 
	) 	
// 便捷的函数接口
```

### 9.1.2. Abstract base class - cv::Feature2D

Feature 2D, 是其他 特征检测类 的基类, 可以作为类接口的通用参考  
* `detect`		:
* `compute`	:  输入图像和 keypoints 计算描述符列表
```cpp

// 特征点检测
virtual void cv::Feature2D::detect 	( 	InputArray  	image,
		std::vector< KeyPoint > &  	keypoints,
		InputArray  	mask = noArray() 
	) 	
cv.Feature2D.detect(	image[, mask]	) -> 	keypoints
// 输入图像计算特征点
// mask 用于对输入图像加掩码, 必须是 8bit 非0 的整数矩阵

virtual void cv::Feature2D::detect 	( 	InputArrayOfArrays  	images,
		std::vector< std::vector< KeyPoint > > &  	keypoints,
		InputArrayOfArrays  	masks = noArray() 
	) 	
cv.Feature2D.detect(	images[, masks]	) -> 	keypoints
// 同时对系列图像计算的形式


// 特征点描述计算  
virtual void cv::Feature2D::compute 	( 	InputArray  	image,
		std::vector< KeyPoint > &  	keypoints,
		OutputArray  	descriptors 
	) 	
cv.Feature2D.compute(	image, keypoints[, descriptors]	) -> 	keypoints, descriptors
// 对单张图片检测到的特征点进行描述计算
// keypoints, 对于无法计算描述特征的关键点将会从列表被删除

virtual void cv::Feature2D::compute 	( 	InputArrayOfArrays  	images,
		std::vector< std::vector< KeyPoint > > &  	keypoints,
		OutputArrayOfArrays  	descriptors 
	) 	
cv.Feature2D.compute(	images, keypoints[, descriptors]	) -> 	keypoints, descriptors
// 对一系列图像的处理, 提前了解一下 ArrayOfArrays 的构造会比较好


// 同时进行特征点检测和特征计算的接口
// 该接口下 mask 成为了关键参数
// 该接口不支持多张图片作为序列输入  
virtual void cv::Feature2D::detectAndCompute 	( 	InputArray  	image,
		InputArray  	mask,
		std::vector< KeyPoint > &  	keypoints,
		OutputArray  	descriptors,
		bool  	useProvidedKeypoints = false 
	) 		
Python:
	cv.Feature2D.detectAndCompute(	image, mask[, descriptors[, useProvidedKeypoints]]	) -> 	keypoints, descriptors
```

## 9.2. Descriptor Matchers - 特征匹配

特征检测和匹配分开在不同的小章节来实现  

OpenCV Main 里实现了 2 种 Matcher, 以及一个基类
* cv::BFMatcher : 暴力 Brute-force
* cv::FlannBasedMatcher : Flann-based
Matcher 类的接口是统一的, 因此可以方便的进行切换   


### 9.2.1. Abstract base class - cv::DescriptorMatcher

cv::DescriptorMatcher 的接口
* create() 类静态函数
* match()  点匹配


```cpp

void cv::DescriptorMatcher::match 	( 	InputArray  	queryDescriptors,
		InputArray  	trainDescriptors,
		std::vector< DMatch > &  	matches,
		InputArray  	mask = noArray() 
	) 		const
void cv::DescriptorMatcher::match 	( 	InputArray  	queryDescriptors,
		std::vector< DMatch > &  	matches,
		InputArrayOfArrays  	masks = noArray() )

Python:
	cv.DescriptorMatcher.match(	queryDescriptors, trainDescriptors[, mask]	) -> 	matches
	cv.DescriptorMatcher.match(	queryDescriptors[, masks]	) -> 	matches

```


## 9.3. Drawing Function of Keypoints and Matches

<!-- 完 -->
在软件开发中需要对匹配的特征点进行 Debug, 可以利用该模组来方便的实现匹配结果可视化, 该部分只有 4 个函数  


枚举 DrawMatchesFlags
```cpp
enum  	cv::DrawMatchesFlags {
// 默认行为, output image 会被重新初始化. 会绘制  两个源图像, 匹配的点, 单个点(未匹配的点). 对于每一个点, 只标注中心, 不会附加圆圈和方向
  cv::DrawMatchesFlags::DEFAULT = 0,
// 不会创建对应的输出矩阵, 匹配将根据输出图像的现有内容进行绘制
  cv::DrawMatchesFlags::DRAW_OVER_OUTIMG = 1,
// 不会绘制未匹配的单个关键点
  cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS = 2,
// 会清晰的标注点, 绘制围绕关键点的圆以及关键点的大小和方向
  cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS = 4
}
```


四个函数, 其中一个用于 Keypoints, 四个用于 Matches
```cpp
void cv::drawKeypoints 	( 	InputArray  	image,
		const std::vector< KeyPoint > &  	keypoints,
		InputOutputArray  	outImage,
		const Scalar &  	color = Scalar::all(-1),
		DrawMatchesFlags  	flags = DrawMatchesFlags::DEFAULT 
	) 		
Python:
	cv.drawKeypoints(	image, keypoints, outImage[, color[, flags]]	) -> 	outImage
// 可能需要了解一下  KeyPoint 类和 Scalar 类
// flags 也可以在此处生效



void cv::drawMatches 	( 	InputArray  	img1,
		const std::vector< KeyPoint > &  	keypoints1,
		InputArray  	img2,
		const std::vector< KeyPoint > &  	keypoints2,
		const std::vector< DMatch > &  	matches1to2,
		InputOutputArray  	outImg,
		const Scalar &  	matchColor = Scalar::all(-1),
		const Scalar &  	singlePointColor = Scalar::all(-1),
		const std::vector< char > &  	matchesMask = std::vector< char >(),
		DrawMatchesFlags  	flags = DrawMatchesFlags::DEFAULT 
	) 	
// 参数还是比较好理解的
// 两张原始图像以及各自的 keypoints vector
// 比配的结果 DMatch
// 用于指定颜色的两个 Scalar
// std::vector< char > Mask determining which matches are drawn. If the mask is empty, all matches are drawn. 


void cv::drawMatches 	( 	InputArray  	img1,
		const std::vector< KeyPoint > &  	keypoints1,
		InputArray  	img2,
		const std::vector< KeyPoint > &  	keypoints2,
		const std::vector< DMatch > &  	matches1to2,
		InputOutputArray  	outImg,
		const int  	matchesThickness,
		const Scalar &  	matchColor = Scalar::all(-1),
		const Scalar &  	singlePointColor = Scalar::all(-1),
		const std::vector< char > &  	matchesMask = std::vector< char >(),
		DrawMatchesFlags  	flags = DrawMatchesFlags::DEFAULT 
	) 	
// 可以指定匹配项的线宽

void cv::drawMatches 	( 	InputArray  	img1,
		const std::vector< KeyPoint > &  	keypoints1,
		InputArray  	img2,
		const std::vector< KeyPoint > &  	keypoints2,
		const std::vector< std::vector< DMatch > > &  	matches1to2,
		InputOutputArray  	outImg,
		const Scalar &  	matchColor = Scalar::all(-1),
		const Scalar &  	singlePointColor = Scalar::all(-1),
		const std::vector< std::vector< char > > &  	matchesMask = std::vector< std::vector< char > >(),
		DrawMatchesFlags  	flags = DrawMatchesFlags::DEFAULT 
	) 	
// matches1to2 和 mask 多加了一维


Python: 
  cv.drawMatches(	img1, keypoints1, img2, keypoints2, matches1to2, outImg[, matchColor[, singlePointColor[, matchesMask[, flags]]]]	) -> 	outImg
  cv.drawMatches(	img1, keypoints1, img2, keypoints2, matches1to2, outImg, matchesThickness[, matchColor[, singlePointColor[, matchesMask[, flags]]]]	) -> 	outImg
  cv.drawMatchesKnn(	img1, keypoints1, img2, keypoints2, matches1to2, outImg[, matchColor[, singlePointColor[, matchesMask[, flags]]]]	) -> 	outImg


```


# 10. objdetect Object Detection 最常用的物体检测模型

## 10.1. Cascade 模型 Cascade Classifier for Object Detection

一个经典的级联传统机器学习模型

`class  	cv::CascadeClassifier`  

官方示例: https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html

预训练模型的下载地址 : https://github.com/opencv/opencv/blob/4.x/samples/python/tutorial_code/objectDetection/cascade_classifier/objectDetection.py



# 11. photo Computational Photography 计算图像处理

包括了几个基于计算的图像处理领域的算法实现, 主要应用于影像


该模组下实现的算法列表, 方便后续单独查找学习:
* inpainting
  * Navier-Stokes based method
  * Alexandru Telea
    * Alexandru Telea. An image inpainting technique based on the fast marching method. Journal of graphics tools, 9(1):23–34, 2004.

* HDR imaging
  * 高动态 成像


## 11.1. inpainting

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

## 11.2. HDR imaging

该分类下实现了多个 class, 用 class 来进行图像的管理  

实现了一个枚举类型, 但只有一个值  `enum  	{ cv::LDR_SIZE = 256 }`  


### Merge

将多个曝光的图片合成到 single image

OpenCV 中实现了一个基类和 3 个派生方法  





### Tone Mapping

High Dynamic Range (HDR), Low Dynamic Range (LDR), 通过相机设备采集到较高位宽的内容, 映射到 LDR 的显示设备的时候, 使用一些算法来提高显示效果, 称为 Tone mapping

从2002 年 Reinhard 的论文开始, 基于计算的 Tone Mapping 有了一系列的发展
* 2002 : Reinhard tone mapping
* 2003 : Adaptive logarithmic mapping (ALM), Frédéric Drago, Karol Myszkowski, Thomas Annen, and Norishige Chiba.
  * 根据局部对比度和亮度分布自适应的调整映射函数的参数  
  * 在 OpenCV 里实现为 TonemapDrago 
* 2006 : 带感知的 HDR 框架, Rafal Mantiuk, Karol Myszkowski, and Hans-Peter Seidel.
  * 通过计算 高斯金字塔, 将各级的梯度转化为对比度, 再转化为 HVS 并缩放, 最后根据新的对比度重建图像.  
* 2007 : CE tone mapping
* 2010 : Uncharted, Flimic tone mapping
  * 通过专家人肉进行 Tone mapping, 再将结果进行拟合
* 美国电影艺术与科学学会  : Academy Color Encoding System (ACES) : 一套颜色编码系统
  * 也是拟合后的曲线, 但是效果更好

```cpp
// Reinhard tone mapping
/* 
	adapted_lum 是根据整个图像计算出来的图像亮度
	MIDDLE_GREY 表示把什么颜色定义成 "灰", 是一个 Magic Number
	整体图像灰暗
 */
float3 ReinhardToneMapping(float3 color, float adapted_lum) 
{
    const float MIDDLE_GREY = 1;
    color *= MIDDLE_GREY / adapted_lum;
    return color / (1.0f + color);
}

// CryEngine : CEToneMapping
/* 
	直接粗暴的搞一个 S 曲线, 没有 Magic Number
	对比度更大, 颜色更鲜艳, 但仍然有些灰
 */
float3 CEToneMapping(float3 color, float adapted_lum) 
{
    return 1 - exp(-adapted_lum * color);
}


// Uncharted2ToneMapping
/* 
	通过应用 专家的拟合曲线实现
 */
float3 F(float3 x)
{
	const float A = 0.22f;
	const float B = 0.30f;
	const float C = 0.10f;
	const float D = 0.20f;
	const float E = 0.01f;
	const float F = 0.30f;
 
	return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}
float3 Uncharted2ToneMapping(float3 color, float adapted_lum)
{
	const float WHITE = 11.2f;
	return F(1.6f * adapted_lum * color) / F(WHITE);
}

// ACES 编码的 Tone Mapping
float3 ACESToneMapping(float3 color, float adapted_lum)
{
	const float A = 2.51f;
	const float B = 0.03f;
	const float C = 2.43f;
	const float D = 0.59f;
	const float E = 0.14f;

	color *= adapted_lum;
	return (color * (A * color + B)) / (color * (C * color + D) + E);
}
```

OpenCV 中实现的 ToneMapping 里则有
* Tonemap 基类
  * TonemapDrago
  * TonemapMantiuk
  * TonemapReinhard
* 通过函数创建 Tonemap 的类 相关参数通过函数参数给出
  * createTonemap_*


```cpp
// gamma : 最基本的 gamma 矫正, 默认值是 1.0 即不启用矫正. 2.2f 是最常用的矫正数值, 是所有 tone mapping 的通用数值 
Ptr<Tonemap> cv::createTonemap 	( 	float  	gamma = 1.0f	) 	

// 
Ptr<TonemapDrago> cv::createTonemapDrago 	( 	float  	gamma = 1.0f,
		float  	saturation = 1.0f,
		float  	bias = 0.85f 
	) 		
Ptr<TonemapMantiuk> cv::createTonemapMantiuk 	( 	float  	gamma = 1.0f,
		float  	scale = 0.7f,
		float  	saturation = 1.0f 
	) 		
Ptr<TonemapReinhard> cv::createTonemapReinhard 	( 	float  	gamma = 1.0f,
		float  	intensity = 0.0f,
		float  	light_adapt = 1.0f,
		float  	color_adapt = 0.0f 
	) 		
// 类的实现较为简单, 除了根据类的属性不同而分别设置的 get set 函数以外, 就是基类中的 set get gamma, 以及处理 method process()
virtual void cv::Tonemap::process 	( 	InputArray  	src,
		OutputArray  	dst 
	) 		

Python:
	cv.createTonemap(	[, gamma]	)  return retval ;
	cv.createTonemapDrago(	[, gamma[, saturation[, bias]]]	)  return retval;
	cv.createTonemapMantiuk(	[, gamma[, scale[, saturation]]]	)  return retval;
	cv.createTonemapReinhard(	[, gamma[, intensity[, light_adapt[, color_adapt]]]]	)  return retval;

	cv.Tonemap.process(	src[, dst]	)  return dst;
```

# 12. contrib : ximgoproc  Extended Image Processing

contrib 包, 拓展的图像处理  


## 12.1. Filter


### 12.1.1. Guided Filter

```cpp
void cv::ximgproc::guidedFilter 	( 	
		InputArray  	guide,		// guided image
		InputArray  	src,		// filtering img
		OutputArray  	dst,
		int  	radius,				// filter 半径
		double  	eps,			// guided filter 的超参数
		int  	dDepth = -1 
	) 		
Python:
	cv.ximgproc.guidedFilter(	guide, src, radius, eps[, dst[, dDepth]]	) -> 	dst

Ptr<GuidedFilter> cv::ximgproc::createGuidedFilter 	( 	
		InputArray  	guide,		// guided image
		int  	radius,				// 半径
		double  	eps 			// epsilon
	) 		
Python:
	cv.ximgproc.createGuidedFilter(	guide, radius, eps	) -> 	retval

virtual void cv::ximgproc::GuidedFilter::filter 	( 	
		InputArray  	src,
		OutputArray  	dst,
		int  	dDepth = -1 
	) 		
	pure virtual
Python:
	cv.ximgproc.GuidedFilter.filter(	src[, dst[, dDepth]]	) -> 	dst
```

对 Guided Filter 的实现, 包括了一个函数实现和类实现  
* `void cv::ximgproc::guidedFilter` : 快速单行应用 Filter, 如果有多张图片要用到同样的 guided img, 使用类的形式更快
* `Ptr<GuidedFilter> cv::ximgproc::createGuidedFilter` : Factory method, 类的工厂构造函数  
* `cv::ximgproc::GuidedFilter::filter` : 类方法

