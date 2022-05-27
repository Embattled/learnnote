# 1. Overview

This project is established for storing learn note written by Yongzhen Long, all contents in English or Chinese.   
Most of the content is copied or translated from Internet resources, without identifying the source.  
Most of the content comes from the official doc and http://c.biancheng.net.  

This readme is rarely updated, links in this file may expired.

该项目用于保存我的个人笔记, 全部为英文或者中文书写.  
绝大部分内容复制或者翻译自互联网资源, 没有在内容中标明来源, 不做商业用途.  
大部分内容来源于项目官方文档和[C语言中文网](http://c.biancheng.net).  

Readme的更新频率极低, 其中的链接常有失效.

Basic knowledge of this node:

1. [Markdown](markdown.md)  : Markdown's syntax, for this node.
2. [Git](gitbasic.md)       : Basic usage of git.

# 2. Development

* 各种开发技术的知识, 包括各种开发语言和相应的库
* Knowledge about development, including various languages and corresponding libraries.

## 2.1. C/CPP

* C/CPP的语言知识, 特性, 官方STL和第三方库.  
* C/CPP's offical knowledge, new feature, stl library, third party library.  

### 2.1.1. C++ knowledge

C 和 C++ 的基础知识以及编译器知识.  
C and C++'s knowledge.  

1. [C Grammar and feature](dev/cpp/note/c.md)   : The C grammar and feature, with some system basic knowledge( Memory allocate, Linux multi-thread).
2. [C++ Grammar and feature](dev/cpp/note/cpp.md)   : The C++ grammar and feature, different place with C.
3. [C++ 11](dev/cpp/note/cpp11.md) : C++11's new feature.
4. [C++ 14](dev/cpp/note/cpp14.md) : C++14's new feature.
5. [C++ 17](dev/cpp/note/cpp17.md) : C++17's new feature.
6. [GCC ](/dev/cpp/note/gcc.md)   : The usage of gcc compiler.
7. [GDB](dev/cpp/note/gdb.md) : The usage of GDB.
8. [CLang](dev/cpp/note/clang.md) : Basic usage of clang complier.



### 2.1.2. C++ STL

C++STL的学习笔记, 包括了C++ STL库和 C 兼容库.  
C/C++ standard library.  

1. [STL Overview](dev/cpp/cppstl/cppstl.md) : The overview of c/cpp std.
2. [Containers](dev/cpp/cppstl/containers.md)  : All cpp containers, like vector, map, list.
3. [String](dev/cpp/cppstl/string.md)  : C++ string class and original c null-terminal string.
5. [Time](dev/cpp/cppstl/time.md)  : Time class defined by C and C++.
6. [IO](dev/cpp/cppstl/io.md)  : Input and output header of C/CPP, like stream or *FILE.
7. [Algorithm](dev/cpp/cppstl/algorithm.md) : Cpp's algorithm header, like sort or find function.
8. [Thread](dev/cpp/cppstl/thread.md) : Thread support libraries, include `<thread>`
10. [Numeric](dev/cpp/cppstl/numeric.md) : All numeric header, cmath, float, numberic, etc.

### 2.1.3. Third-Party Libary

1. [OpenCV C++](dev/cpp/library/opencv.md) : The C++ part of opencv.
2. [HPC](dev/cpp/library/hpc.md)   : The basic knowledge of openmp and mpi.

## 2.2. Python

* Python以及机器学习框架笔记.  
* Python and ML framework library and other library.  

### 2.2.1. Python and Python's tools

1. [Python](dev/python/python/python.md)  : Python syntax and feature note.
2. [PythonExEm](dev/python/python/pythonexem.md) : Python Extending and Embedding the Python Interpreter.
3. [anaconda](dev/python/python/anaconda.md)  : The usage of anaconda suit, includes conda, jupyter note.

### 2.2.2. Python STL

* Python's build-in module.  [File-link](dev/python/pystl/)  

### 2.2.3. Python module

Third-party python module.

* DataOP module
1. [Pandas](dev/python/pymodule/data/pandas.md)    : The usage of pandas.
2. [Numpy](dev/python/pymodule/data/numpy.md)  : Numpy is a really basic modual.
3. [matplotlib](dev/python/pymodule/data/matplotlib.md) : Usage of matplotlib.

* Image Library
1. [PIL](dev/python/pymodule/pil/pillow.md) : Usage of pillow.  

### 2.2.4. ML Framework

* Pytorch
1. [torch](dev/python/mlframework/pytorch/pytorch.md)  : Usage of torch namespace.
2. [torchvision](dev/python/mlframework/pytorch/torchvision.md) : Usage of torchvision.
3. [Net Structure](dev/python/mlframework/pytorch/torchvisionmodel.py) : Source code of network in torchivison.

* sci-kit
1. [scipy](dev/python/mlframework/scikit/scipy.md) : Brief introduce about scipy series.  
2. [scikit-learn](dev/python/mlframework/scikit/scikit-learn.md) : Usage of sklearn.
3. [scikit-image](dev/python/mlframework/scikit/scikit-image.md) : Usage of skimage.

* Todo

1. [TensorFlow](dev/python/mlframework/tensorflow.md)
2. [Chainer](dev/python/mlframework/chainer.md)


## 2.3. Container 

1. [docker](dev/container/docker.md) : Basic usage of docker and syntax of dockerfile.
2. [kubernetes](dev/container/kubernetes.md) : Todo

## 2.4. vscode

Document about various vscode pluge-in.  

1. [cpp](dev/vscode/cpp.md) : Create cpp environment in vscode.
2. [python](dev/vscode/python.md) : Create python environment in vscode.

## 2.5. Cuda

Developement use CUDA.

1. [cuda basic](dev/cuda/cuda1basic.md) : basic CUDA.
2. [cuda linux](dev/cuda/cudalinux.md) : CUDA environment in linux.

# 3. CS Knowledge

* [algorithm](knowledge/algorithm.md) : Algorithm note.

## 3.1. ML Note

1. [machine learning](knowledge/mlnote/machinelearning.md) : My learning note about machine learning.
2. [deep learning](knowledge/mlnote/deeplearning.md) : My learning note about deep learning.

## 3.2. Network 

1. [http protocol](knowledge/network/http.md) : Structure of http.

# 4. Linux

Linux笔记以及相关软件使用笔记.  
Knowledge of linux.  

1. [Linux Basic](linux/linuxbasic.md)   : The basic knowledge of linux.  
2. [ssh](linux/ssh.md)  : The usage of ssh and syntax of config file.  
3. [vivim](linux/vivim.md)  : The usage of editor "vi".
4. [software](linux/software.md)    : The software management method on linux.
5. [terminal](linux/terminal.md) : Usage of tmux.  
6. [Makefile](linux/makefile.md) : The basic syntax of makefile.

