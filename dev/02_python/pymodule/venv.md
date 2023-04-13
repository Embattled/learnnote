# 1. Python 的环境管理

[官方推荐文档]<https://packaging.python.org/guides/tool-recommendations/>  

在开发Python应用程序的时候, 系统安装的Python3只有一个版本：3.4.  所有第三方的包都会被pip安装到Python3的site-packages目录下。

如果我们要同时开发多个应用程序, 那这些应用程序都会共用一个Python, 就是安装在系统的Python 3  
如果应用A需要jinja 2.7, 而应用B需要jinja 2.6怎么办  

python 环境的各种包或者工具
* pyenv
* conda

## pyenv 解释器版本管理器

https://github.com/pyenv/pyenv/

专门用于管理 python 版本的软件, 不属于 python 的 module, 作为独立的软件存在, 没有虚拟环境的功能  

```sh
pyenv install 3.10
pyenv global 3.10
```

## 2.1. virtualenv

virtualenv 是目前最流行的 python 虚拟环境配置工具

- 同时支持 python2 和 python3
- 可以为每个虚拟环境指定 python 解释器 并选择不继承基础版本的包。

使用pip3安装  
`pip3 install virtualenv`  

```shell
# 测试安装 查看版本
virtualenv --version

# 创建虚拟环境
cd my_project
virtualenv my_project_env

# 指定python 执行器
-p /usr/bin/python2.7

# 激活虚拟环境
source my_project_env/bin/activate
# 停用 回到系统默认的Python解释器
deactivate
```

## 2.2. virtualenvwrapper

`pip install virtualenv virtualenvwrapper`  

virtualenvwrapper 是对 virtualenv 的一个封装, 目的是使后者更好用
使用 shell 脚本开发, 不支持 Windows  

它使得和虚拟环境工作变得愉快许多

- 将您的所有虚拟环境在一个地方。
- 包装用于管理虚拟环境（创建, 删除, 复制）。
- 使用一个命令来环境之间进行切换。

```shell


#设置环境变量 这样所有的虚拟环境都默认保存到这个目录
export WORKON_HOME=~/Envs  
#创建虚拟环境管理目录
mkdir -p $WORKON_HOME


# 每次要想使用virtualenvwrapper 工具时, 都必须先激活virtualenvwrapper.sh
find / -name virtualenvwrapper.sh #找到virtualenvwrapper.sh的路径
source 路径 #激活virtualenvwrapper.sh

# 创建虚拟环境  
# 该工具是统一在当前用户的envs文件夹下创建, 并且会自动进入到该虚拟环境下  
mkvirtualenv ENV
mkvirtualenv ENV  --python=python2.7

# 进入虚拟环境目录  
cdvirtualenv

Create an environment with `mkvirtualenv`

Activate an environment (or switch to a different one) with `workon`

Deactivate an environment with` deactivate`

Remove an environment with`rmvirtualenv`

# 在当前文件夹创建独立运行环境-命名
# 得到独立第三方包的环境, 并且指定解释器是python3
$ mkvirtualenv cv -p python3

# 进入虚拟环境  
source venv/bin/activate  

#接下来就可以在该虚拟环境下pip安装包或者做各种事了, 比如要安装django2.0版本就可以：
pip install django==2.0

```

**其他命令**:

- workon `ENV`          : 启用虚拟环境
- deactivate            : 停止虚拟环境
- rmvirtualenv `ENV`    : 删除一个虚拟环境
- lsvirtualenv          : 列举所有环境
- cdvirtualenv          : 导航到当前激活的虚拟环境的目录中, 比如说这样您就能够浏览它的 site-packages
- cdsitepackages        : 和上面的类似, 但是是直接进入到 site-packages 目录中
- lssitepackages        : 显示 site-packages 目录中的内容
