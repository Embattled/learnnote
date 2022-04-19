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

### API Concepts 

* 尽管从文档上, OpenCV 分成了数个模组
* 事实上整个 OpenCV 库都放在了 `cv` 的命名空间中
* 有些函数会和 STL 冲突, 因此有时候需要特殊指定

```cpp
#include "opencv2/core.hpp"


```

### 1.2.1. Install on linux

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

### 1.2.2. configuration options

管理 OpenCV



### 1.2.3. Installation layout



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



# core

定义了 OpenCV 中最核心的类 (图像), 以及一些数学上的操作 

## mat  core/mat.hpp

同 numpy 一样, opencv并没有特地的图片类, 而是用矩阵来表示 -> `cv::Mat`  




# imgcodecs  Image file reading and writing 

同 Image Process 不同, 该模组中的函数都是图像 IO 相关的  

`#include <opencv2/imgcodecs.hpp>` 

## Flags used for image file reading and writing

用作图片 IO 的参数, 一般都是枚举类型, 用于指定图片的规格等



## 基础图像读写

`imread()` 用于从一个文件路径中读取图像
* filename  :Name of file to be loaded.
* flags     :Flag that can take values of `cv::ImreadModes`

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
```

```python
cv.imread(filename[, flags]	) -> 	retval

cv.imwrite(	filename, img[, params]	) -> 	retval
```

