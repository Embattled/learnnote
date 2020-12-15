# 1. Anaconda 的安装和使用

* 开源
* 安装过程简单
* 高性能使用Python和R语言
* 免费的社区支持

Anaconda是一个包含180+的科学包及其依赖项的发行版本。其包含的科学包包括：conda, numpy, scipy, ipython notebook等  

## 1.1. 在linux 下安装

1. 在官方下载页面下载好 用 bash 运行  除非被要求使用root权限，否则均选择  Install Anaconda as a user  
2. `bash ~/Anaconda3-5.0.1-Linux-x86_64.sh`  
3. 跟随终端文字提示下一步即可  
4. "你希望安装器添加Anaconda安装路径在 `/home/<user>/.bashrc` 文件中吗"  建议输入“yes  
5. 输入 `source ~/.bashrc`  更新用户环境变量
6. 验证安装
   1. `conda --version` 输出 `conda` 版本  
   2. 终端中输入命令 `condal list` ，如果Anaconda被成功安装，则会显示已经安装的包名和版本号。
   3. 在终端中输入 `python` 如果Anaconda被成功安装并且可以运行，则将会在Python版本号的右边显示 "Anaconda custom (64-bit)"
   4. 在终端中输入 `anaconda-navigator` 。如果Anaconda被成功安装，则Anaconda Navigator将会被启动

管理 base 环境:  
* 可以设定是否对所有 shell 连接默认进入 base 环境
* `conda config --set auto_activate_base False` or True  


## 1.2. 升级

```shell

conda update conda
conda update anaconda=VersionNumber

# 升级所有 , 包括包以及依赖, 让所有东西都尽可能的新
# 只会升级你当前所选择的环境
conda update --all
# 如果要指定别的环境升级, 用 -n 指定名称
conda update -n myenv --all


# 在 conda 中 anything is a package。conda 本身可以看作是一个包，python 环境可以看作是一个包
# anaconda 也可以看作是一个包，因此除了普通的第三方包支持更新之外，这3个包也支持。比如：

# 更新conda本身
conda update conda
# 更新anaconda 应用
conda update anaconda
# 更新python，假设当前python环境是3.6.1，而最新版本是3.6.2，那么就会升级到3.6.2
conda update python
```
## 1.3. 卸载


**简单卸载**
直接 `rm -rf ~/anaconda3`  即可  可以在 bashrc 中删除 conda 的相关环境变量  

**程序卸载** 必须在简单卸载之前执行   
```shell
# 安装卸载工具
conda install anaconda-clean
# 带确认窗口的删除
anaconda-clean
# 不带确认窗口的删除
anaconda-clean --yes

```


# 2. conda 包管理和环境管理

conda 是 anaconda 下包管理和环境管理的工具  
类似于 pip 和 vitualenv 的组合体


适用语言：Python, R, Ruby, Lua, Scala, Java, JavaScript, C/C++, FORTRAN。   
conda为Python项目而创造，但可适用于上述的多种语言。  
对比于python自建的环境管理包 `Pipenv` 或者 `Poetry` 包   
conda的环境管理更底层, 因为 python 本身就是 conda 的一个包  

对比差距:  
pip：
    不一定会展示所需其他依赖包。
    安装包时或许会直接忽略依赖项而安装，仅在结果中提示错误。
    在系统自带Python中包的**更新/回退版本/卸载将影响其他程序。
conda：
    列出所需其他依赖包。
    安装包时自动安装其依赖项。
    可以便捷地在包的不同版本中自由切换。
    不会影响系统自带Python。


conda 的操作主要通过 `conda` 命令来实现  
1. 快速的查找所有 anaconda 包以及当前已安装的包
2. 创建环境
3. 修改环境中的包 (安装和升级)

```sh
# conda 支持命令简写 -- 双横线的命令可以使用单横线加首字母的形式运行
conda --name
conda -n

conda --envs
conda -e
```

## 2.1. conda 格式

### 2.1.1. conda package

conda package 是一个压缩包 (.tar.bz2) 或者 .conda 文件  
.conda 文件是 conda 4.7 新加入的, 相比压缩包更轻量化更快  

包中包括:  
* 系统及的库 system-level libraries
* Python 或者其他模组 
* 可执行程序以及其他组件
* info/ 目录下的 metadata
* 一组直接安装的 install 文件
      .
      ├── bin
      │   └── pyflakes
      ├── info
      │   ├── LICENSE.txt
      │   ├── files
      │   ├── index.json
      │   ├── paths.json
      │   └── recipe
      └── lib
         └── python3.5
* bin : 包相关的二进制文件
* lib : 包相关的库文件
* info: 包的 metadata

### 2.1.2. conda 的目录结构

* /pkgs  : 保存了已被解压的包, 可以直接被链接到一个 conda 环境  
* /bin /include /lib/ share 都是 base 环境的内容
* /envs  : 用于保存额外的环境 即(base)环境之外的环境, 子目录下的内容和上条相同

## 2.2. conda channel

channel 是一个包目录的 url , 用于检索conda 包  

默认的channel 是 `https://repo.anaconda.com/pkgs/`  

除此之外有名为 `Conda-forge` 的社区型 conda 索引库  


```sh
# 查看当前环境的 conda 所有配置信息
conda config --show
# 查看当前 channel
conda config --show channels

# 通过指定 channel 安装 scipy
conda install scipy --channel conda-forge

# 同时指定多个channel 优先级从左到右
conda install scipy --channel conda-forge --channel bioconda

# 使用 --override-channels 来指定只使用指定的 channel 安装  
conda search scipy --channel conda-forge --override-channels
```


### 2.2.1. channel priority

为了解决不同 channel 之间的冲突, 高优先级的 channel 会覆盖低优先级的  
不管低优先级的 channel 中包的版本是否更加新  
* 当默认通道中没有想要的包时, 可以安全的将新的通道放在最下层优先级  
* 如果想要 conda 只安装最新的版本 通过命令修改 config 
  * `conda config --set channel_priority false`
  * 这样总会安装版本号更新的python


### channel 管理

增加新的 channel 应该明确优先级

```sh
# 放在顶部, 拥有最高优先级
conda config --add channels new_channel
conda config --prepend channels new_channel

# 追加在 list 底部 最低优先级
conda config --append channels new_channel

# 删除一个旧的channel
conda config --remove channels old_channel
```

##  2.3. conda 的环境管理 
conda environment 是一个目录, 保存了对该环境安装了的 conda packages.  

通过 `environment.yaml` 可以轻松的分享运行环境  

`conda info --envs`  查看当前系统所拥有的环境  

### 2.3.1. 创建
```shell

#创建虚拟环境 conda create --name <env_name> <package_names>
#基于python3.8创建一个名字为python36的环境
conda create --name python38 python=3.8

# 加入多个包
conda create -n python3 python=3.5 numpy pandas
# 不指定python版本的话则会安装与 anaconda　版本相同的　python 版本


# 复制一个环境
conda create --name <new_env_name> --clone <copied_env_name>

```

### 2.3.2. 使用
```shell
#激活虚拟环境
activate python36   # windows 平台
source activate python36 # linux/mac 平台

#退出当前虚拟环境
deactivate python36 

```

### 2.3.3. 管理

```sh
#查看所有已安装的虚拟环境
conda info -e
python36              *  D:\Programs\Anaconda3\envs\python36
root                     D:\Programs\Anaconda3


#删除虚拟环境
conda remove -n python36 --all
# 或者 
conda env remove  -n python36

```

## 2.4. conda 的包管理

conda 的包管理功能可 pip 是一样的，当然你选择 pip 来安装包也是没问题的。  

```shell

# 查看已安装的包
# Check to see if the newly installed program is in this environment:
conda list 

# 搜索一个包 获取所有的版本的信息
conda search scipy

# 安装 matplotlib 到当前环境
conda install matplotlib
# 包更新
conda update matplotlib

# conda updata 不会跨大版本升级 same major version number
# conda install 则可以指定任意安装版本 默认安装或者更新到最新版
conda install python=3 

# 删除包
conda remove matplotlib

# 在安装了 conda-build 后 还可以直接通过源文件安装一个包
conda build my_fun_package
```

# 3. Jupyter Notebook

Files: 显示当前 Notebook工作文件夹 中所有文件和文件夹
Running : 列出所有正在运行的 notebook

## 3.1. Google Colab

Colab 笔记本是由 Google 托管的 Jupyter 笔记,   
可以将可执行代码、富文本以及图像、HTML、LaTeX 等内容合入 1 个文档中   

Colab的代码区默认是python代码  

### 3.1.1. 插入表单

在代码区右键可以插入表格   

插入后的样式如下:
```
#@title String fields

text = 'value' #@param {type:"string"}
dropdown = '1st option' #@param ["1st option", "2nd option", "3rd option"]
text_and_dropdown = 'value' #@param ["1st option", "2nd option", "3rd option"] {allow-input: true}
```

使用 #@param 来对表格的值的类型 默认值 选项进行设置 (必须)  



### 3.1.2. 安装非默认python库 指定TF版本

You can use `!pip install` or `!apt-get install`.  
来安装默认库环境里没有的包  

Colab 里默认安装好了TensorFlow 1.x 和 2.x  
默认是2.x版本的, 可以手动指定成版本1  使用`%`加关键字即可  
`%tensorflow_version 1.x`  
 
一旦运行了 `import tensorflow`  若想更改版本必须重启运行Colab系统  



```
!pip install matplotlib-venn
!apt-get -qq install -y libfluidsynth1

可以指定TensorFlow的版本, 不建议这么做    
!pip show tensorflow
!pip install --upgrade tensorflow
!pip install tensorflow==1.2
!pip install tf-nightly

!apt-get -qq install -y libarchive-dev && pip install -U libarchive
import libarchive

!apt-get -qq install -y graphviz && pip install pydot
import pydot

!apt-get -qq install python-cartopy python3-cartopy
import cartopy

```


### 3.1.3. 执行系统命令

在代码区使用 "!" 可以执行shell命令

结果甚至可以存储到变量里  
```python
!ls /bin

message = 'Colaboratory is great!'
foo = !echo -e '$message\n$message'

```

### 3.1.4. 执行html文本

使用 %%html可以将代码区变成 html文本  

```html
%%html
<marquee style='width: 10%; color: blue;'><b>Whee!</b></marquee>
```

### 3.1.5. 连接Github

Colab 可以整合的连接Github  包括从Github读取笔记本和把笔记本存到Github

对于一个Github库:  
`https://github.com/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb.`

那么其对应的Colab笔记本链接为:  
`https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb.`


这里分层次:
1. 访问Github
   `http://colab.research.google.com/github`
2. 访问用户 googlecolab 的空间
   `http://colab.research.google.com/github/googlecolab/ `
3. 访问 googlecolab 的 colabtools 库, 这里将会默认浏览 master 分支
   `http://colab.research.google.com/github/googlecolab/colabtools/`
4. 手动指定访问该库的 master 分支, 注意 `blob` 是关键字
   `http://colab.research.google.com/github/googlecolab/colabtools/blob/master `


