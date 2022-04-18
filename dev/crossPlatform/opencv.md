# 1. OpenCV

OpenCV (Open Source Computer Vision Library) :  
1. An open-source library that includes several hundreds of computer vision algorithms.
2. C-based OpenCV 1.x API has been deprecated.
3. OpenCV 2.x API is essentially a C++ API.




## 1.1. Modular Structure

1. Core functionality (core)        : Basic data structures.
2. Image Processing (imgproc)       : Image processing module.
3. Video Analysis (video)           : Motion estimation, tracking.
4. Video I/O (videoio)
5. Camera Calibration and 3D Reconstruction (calib3d)
6. 2D Features Framework (features2d)
7. Object Detection (objdetect)
8. High-level GUI (highgui) 


## 1.2. opencv C++

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

* OpenCV now supports a multitude of algorithms related to Computer Vision and Machine Learning and is expanding day by day.  

* OpenCV-Python is the Python API for OpenCV, combining the best qualities of the OpenCV C++ API and the Python language.
* OpenCV-Python is a Python wrapper for the original OpenCV C++ implementation.

**Numpy**
* OpenCV-Python makes use of Numpy. And OpenCV-Python requires **only Numpy**.
* All the OpenCV array structures are converted to and from Numpy arrays. 
* This also makes it easier to integrate with other libraries that use Numpy.
  * such as **SciPy** and **Matplotlib**.

