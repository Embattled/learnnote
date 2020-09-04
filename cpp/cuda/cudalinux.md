# Linux下配置CUDA

## 安装结束后 

```shell
# 更新用户PATH
# 编辑profile  即 ~/.profile  末尾加入代码
export PATH=/usr/local/cuda/bin:$PATH


# 更新 UNIX shell 启动文件
# 编辑 ~/.bashrc

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBARY_PATH

```