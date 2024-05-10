# C++/Eigen3

什么是 Eigen3 , 为什么要用它
* 可以理解为 C++ 的 Numpy
* Header only implementation: no libraries to compile and install (easy). 完全定义在头文件里的纯模板库
* Support for dense and sparse matrices, vectors and arrays.


Requirement:
* 只需要 C++ 标准库
* 如果需要编译文档, 则需要CMake

官方文档  
https://eigen.tuxfamily.org/dox/


## short guide 

简短篇幅快速入门  

如何安装?
* 下载源代码, 并根据需要导入  `Eigen` 目录下的各种头文件即可

```cpp
#include <iostream>
#include <Eigen/Dense>
 
// 对于类型规则, 这里  X 代表矩阵是 arbitrary size的
// d 代表 double
using Eigen::MatrixXd;
 
int main()
{
  MatrixXd m(2,2);
  m(0,0) = 3;
  m(1,0) = 2.5;
  m(0,1) = -1;
  m(1,1) = m(1,0) + m(0,1);
  std::cout << m << std::endl;
}
```

如何编译?
* 由于是纯头文件的库, 因此只需要保证编译器能够访问到 Eigen 头文件即可, 且无需担心各种平台的兼容性问题  
* `g++ -I /path/to/eigen/ my_program.cpp -o my_program`

## Quick Reference Guide

库的 API 文档没有做很好的可读化和分类, 因此要查找类接口的话需要找到对应类的 doxygen 页面  

提供了一个快速介绍用于方便定位要查找的内容  


## Modules and Header files - 模组和头文件

源代码下提供了数个头文件用于定位 Eigen 核心内容和几个不同的模组  

| Header file `#include <Eigen/*>` | 内容                                                                                                                      |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| Core                             | 核心的矩阵和数列类, 对应的操纵接口, 基础的线性代数 (including triangular and selfadjoint products)                        |
| Geometry                         | 转换, 平移, Scaling 缩放, 旋转,  3D rotations (Quaternion, AngleAxis)                                                     |
| LU                               | 逆矩阵, 秩, LU分解 LU decompositions with solver (FullPivLU, PartialPivLU)                                                |
| Cholesky                         | LLT and LDLT Cholesky factorization with solver                                                                           |
| Householder                      | 可能是一些底层实现? Householder transformations; this module is used by several linear algebra modules.                   |
| SVD                              | SVD 分解 with least-squares solver (JacobiSVD, BDCSVD)                                                                    |
| QR                               | QR 分解 with solver (HouseholderQR, ColPivHouseholderQR, FullPivHouseholderQR)                                            |
| Eigenvalues                      | 特征值, 特征向量分解 Eigenvalue, eigenvector decompositions (EigenSolver, SelfAdjointEigenSolver, ComplexEigenSolver)     |
| Sparse                           | 稀疏矩阵相关的线性代数是实现                                                                                              |
| Dense                            | 用于方便的导入所有矩阵相关的头文件 `Core, Geometry, LU, Cholesky, SVD, QR, and Eigenvalues `, 即除了 Sparse 和 HoseHolder |
| Eigen                            | 导入全部头文件                                                                                                            |


