# Linux下配置CUDA

## 安装结束后 

```shell
# 更新用户PATH
# 编辑profile  即 ~/.profile 或者全局的 /etc/profile  末尾加入代码
export PATH=/usr/local/cuda/bin:$PATH


# 更新 UNIX shell 启动文件
# 编辑 ~/.bashrc

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBARY_PATH

```

## 删除 CUDA

To uninstall the CUDA Toolkit, run cuda-uninstaller in /usr/local/cuda-11.0/bin


To uninstall the NVIDIA Driver, run nvidia-uninstall


##　CUDA on WSL User Guide

NVIDIA driver support for WSL 2 includes not only `CUDA` but also `DirectX` and `Direct ML` support


WSL 上的 CUDA 系统层级栈
* TensorFlow, Pytorch, AI Frameworks & APPs
* NVIDIA CUDA
* WSL 2 Environment (Linux Kernel) WSL
* GPU paravirtualization
* NVIDIA Windows driver
* Windows kernel
* Windows with GPU Hardware Machines

https://docs.nvidia.com/cuda/wsl-user-guide/index.html#installing-insider-preview-builds

安装流程
* 注意: 不需要在 WSL 中安装 NVIDIA 驱动, 不要安装
* 在 windows 中安装NVIDIA 驱动
* 安装 WSL
* 

# NCCL - NVIDIA Collective Communications Library

NVIDIA Collective Communications Library (NCCL, pronounced Nickel)

实现集体通信和点对点 接/发 原语, 不是一个并行编程框架, 更加专注于 inter-GPU 通信加速.  

NCCL provides the following collective communication primitives :

    AllReduce
    Broadcast
    Reduce
    AllGather
    ReduceScatter


