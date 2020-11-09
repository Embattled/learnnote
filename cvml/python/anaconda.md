# 1. Anaconda 的安装和使用

# 2. conda 包管理和环境管理

conda 是 anaconda 下包管理和环境管理的工具  
类似于 pip 和 vitualenv 的组合体

##  2.1. conda 的环境管理 

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

## 2.2. conda 的包管理

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

# 3. Jupyter Notebook

Files: 显示当前 Notebook工作文件夹 中所有文件和文件夹
Running : 列出所有正在运行的 notebook

## Google Colab

Colab 笔记本是由 Google 托管的 Jupyter 笔记,   
可以将可执行代码、富文本以及图像、HTML、LaTeX 等内容合入 1 个文档中   

Colab的代码区默认是python代码  

### 插入表单

在代码区右键可以插入表格   

插入后的样式如下:
```
#@title String fields

text = 'value' #@param {type:"string"}
dropdown = '1st option' #@param ["1st option", "2nd option", "3rd option"]
text_and_dropdown = 'value' #@param ["1st option", "2nd option", "3rd option"] {allow-input: true}
```

使用 #@param 来对表格的值的类型 默认值 选项进行设置 (必须)  



### 安装非默认python库 指定TF版本

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


### 执行系统命令

在代码区使用 "!" 可以执行shell命令

结果甚至可以存储到变量里  
```python
!ls /bin

message = 'Colaboratory is great!'
foo = !echo -e '$message\n$message'

```

### 执行html文本

使用 %%html可以将代码区变成 html文本  

```html
%%html
<marquee style='width: 10%; color: blue;'><b>Whee!</b></marquee>
```

### 连接Github

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


