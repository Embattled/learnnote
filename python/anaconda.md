# Anaconda 的安装和使用

# conda 包管理和环境管理

conda 是 anaconda 下包管理和环境管理的工具  
类似于 pip 和 vitualenv 的组合体

##  conda 的环境管理 

```shell
#创建虚拟环境
#基于python3.8创建一个名字为python36的环境
conda create --name python36 python=3.8

#激活虚拟环境
activate python36   # windows 平台
source activate python36 # linux/mac 平台

#退出当前虚拟环境
deactivate python36 

#删除虚拟环境
conda remove -n python36 --all
# 或者 
conda env remove  -n python36

#查看所有已安装的虚拟环境
conda info -e
python36              *  D:\Programs\Anaconda3\envs\python36
root                     D:\Programs\Anaconda3
```

## conda 的包管理

conda 的包管理功能可 pip 是一样的，当然你选择 pip 来安装包也是没问题的。  

```shell
# 安装 matplotlib 
conda install matplotlib
# 查看已安装的包
conda list 
# 包更新
conda update matplotlib
# 删除包
conda remove matplotlib

# 在 conda 中 anything is a package。conda 本身可以看作是一个包，python 环境可以看作是一个包
# anaconda 也可以看作是一个包，因此除了普通的第三方包支持更新之外，这3个包也支持。比如：

# 更新conda本身
conda update conda
# 更新anaconda 应用
conda update anaconda
# 更新python，假设当前python环境是3.6.1，而最新版本是3.6.2，那么就会升级到3.6.2
conda update python
```

## Jupyter Notebook

Files: 显示当前 Notebook工作文件夹 中所有文件和文件夹
Running : 列出所有正在运行的 notebook

