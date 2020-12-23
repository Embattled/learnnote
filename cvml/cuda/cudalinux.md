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