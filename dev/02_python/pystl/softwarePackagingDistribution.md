# 1. Software Packaging and Distribution

该分组下的模组是官方提供的用于 publishing 和 installing  python software 的模组.  

默认情况下是与 pypi 官方包索引一起使用的, 但是也可以完全工作在本地 index 服务器下, 或者没有任何索引服务器.  

# 2. distutils - Building and installing Python modules

提供了一种完全本地化的 打包和 解包安装的 模组


# 3. venv - Creation of virtual environments

python 3.3 新功能, 完全 python 内部的虚拟环境管理工具, 用于最轻量化的创建一个虚拟 python 环境, 可以管理包, 但是不能管理 python 自己的版本  

在 venv 下, 使用 pip 会自动将包安装在当前激活的 venv 下  


## 3.1. venv CLI

使用 `python -m venv ...` 命令调用 venv 包的功能, 进行虚拟环境操作  

```shell
# vene ENV_DIR  指定虚拟环境的位置  
python3 -m venv tutorial-env
python3 -m venv /path/to/new/virtual/environment

# 建立的虚拟环境会有其子目录, 通过 source 该目录下的  `bin/activate` 来激活虚拟环境
source /path/to/new/virtual/environment/bin/activate
```

## 3.2. API

venv 内建的一些类提供了 API 用于其他第三方基于 venv 建立定制化的虚拟环境:  the `EnvBuilder` class.


# 4. ensurepip — Bootstrapping the pip installer