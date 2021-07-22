# 1. Overview

This project is established for storing learn note written by Yongzhen Long, all contents in English or Chinese.   
Most of the content is copied or translated from Internet resources, without identifying the source.  
Most of the content comes from the official doc and http://c.biancheng.net.  

该项目用于保存我的个人笔记, 全部为英文或者中文书写.  
绝大部分内容复制或者翻译自互联网资源, 没有在内容中标明来源, 不做商业用途.  
大部分内容来源于项目官方文档和[C语言中文网](http://c.biancheng.net).  



Basic knowledge of this node:

1. [Markdown](markdown.md)  : Markdown's syntax, for this node.
2. [Git](gitbasic.md)       : Basic usage of git.


# 2. C/CPP

C/CPP的语言知识, 官方STL和第三方库.  
C/CPP's offical knowledge, new feature, stl library, third party library.  

## 2.1. C++ knowledge

C 和 C++ 的基础知识以及编译器知识.  
C and C++'s knowledge.  

1. [C Grammar and feature](cpp/note/c.md)   : The C grammar and feature, with some system basic knowledge( Memory allocate, Linux multi-thread).
2. [C++ Grammar and feature](cpp/note/cpp.md)   : The C++ grammar and feature, different place with C.
3. [C++ 11](cpp/note/cpp11.md) : C++11's new feature.
4. [C++ 14](cpp/note/cpp14.md) : C++14's new feature.
5. [Makefile](cpp/note/makefile.md) : The basic syntax of makefile.
6. [GCC ](/cpp/note/gcc.md)   : The usage of gcc compiler.
7. [GDB](cpp/note/gdb.md) : The usage of GDB.


## 2.2. STL

C++STL的学习笔记, 包括了C++ STL库和 C 兼容库.  
C/C++ standard library.  

1. [STL Overview](cpp/cppstl/cppstl.md) : The overview of cpp std.
2. [Containers](/cpp/cppstl/containers.md)  : All cpp containers, like vector, map, list.
3. [String](/cpp/cppstl/string.md)  : C++ string class and original c null-terminal string.
5. [Time](/cpp/cppstl/time.md)  : Time class defined by C and C++.
6. [IO](/cpp/cppstl/io.md)  : Input and output header of C/CPP, like stream or *FILE.
7. [Algorithm](cpp/cppstl/algorithm.md) : Cpp's algorithm header, like sort or find function.
8. [Thread](cpp/cppstl/thread.md) : Thread support libraries, include `<thread>`
10. [Numeric](cpp/cppstl/numeric.md) : All numeric header, cmath, float, numberic, etc.



## 2.3. Third-Party Libary

1. [OpenCV C++](/cpp/library/opencv.md) : The C++ part of opencv.
2. [HPC](cpp/note/hpc.md)   : The basic knowledge of openmp and mpi.

# 3. CVML

Python以及机器学习框架笔记.  
Machine Learning knowledge node, Python and ML framework library and other library.  

## 3.1. Python and Python's tools

1. [Python](cvml/python/python.md)  : Python syntax and feature note.
2. [PythonExEm](cvml/python/pythonexem.md) : Python Extending and Embedding the Python Interpreter.
3. [anaconda](cvml/python/anaconda.md)  : The usage of anaconda suit, includes conda, jupyter note.

## 3.2. Python STL

Python's build-in module.  [File-link](cvml/pystl/)  

## 3.3. Python module

Third-party python module.

### 3.3.1. DataOP module

1. [Pandas](/cvml/data/pandas.md)    : The usage of pandas.
2. [Numpy](/cvml/data/numpy.md)  : Numpy is a really basic modual.
3. [matplotlib](cvml/data/matplotlib.md) : Usage of matplotlib.


### 3.3.2. Image Library

1. [PIL](cvml/pymodule/pil/pillow.md) : Usage of pillow.  

## 3.4. ML Note

1. [machine learning](cvml/mlnote/machinelearning.md) : My learning note about machine learning.
2. [deep learning](cvml/mlnote/deeplearning.md) : My learning note about deep learning.

## 3.5. ML Framework

### 3.5.1. Pytorch

1. [torch](cvml/mlframework/pytorch/pytorch.md)  : Usage of torch namespace.
2. [torchvision](cvml/mlframework/pytorch/torchvision.md) : Usage of torchvision.
3. [Net Structure](cvml/mlframework/pytorch/torchvisionmodel.py) : Source code of network in torchivison.

### 3.5.2. sci-kit

1. [scipy](cvml/mlframework/scikit/scipy.md) : Brief introduce about scipy series.  
2. [scikit-learn](cvml/mlframework/scikit/scikit-learn.md) : Usage of sklearn.
3. [scikit-image](cvml/mlframework/scikit/scikit-image.md) : Usage of skimage.

### 3.5.3. Todo

1. [TensorFlow](cvml/mlframework/tensorflow.md)
2. [Chainer](cvml/mlframework/chainer.md)

# 4. Linux

Linux笔记以及相关软件使用笔记.  
Knowledge of linux.  

1. [Linux Basic](linux/linuxbasic.md)   : The basic knowledge of linux.  
2. [ssh](linux/ssh.md)  : The usage of ssh and syntax of config file.  
3. [vivim](linux/vivim.md)  : The usage of editor "vi".
4. [software](linux/software.md)    : The software management method on linux.
5. [terminal](linux/terminal.md) : Usage of tmux.  

# 5. Advance Development 

## 5.1. Container 

1. [docker](dev/container/kubernetes.md) : Basic usage of docker and syntax of dockerfile.
2. [kubernetes](dev/container/kubernetes.md) : Todo

## 5.2. Vscode

Document about various vscode pluge-in.  

1. [cpp](dev/vscode/cpp.md) : Create cpp environment in vscode.
2. [python](dev/vscode/python.md) : Create python environment in vscode.

## 5.3. Cuda

Developement use CUDA.

1. [cuda basic](dev/cuda/cuda1basic.md) : basic CUDA.
2. [cuda linux](dev/cuda/cudalinux.md) : CUDA environment in linux.
