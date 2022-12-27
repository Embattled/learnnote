# 1. Runtime Services

该分类的模组用于提供 Python 解释器和其运行环境, 也就是python 解释器和操作系统之间的交互  

# 2. sys python解释器服务

与 python 解释器密切相关的一些函数与变量  

可以用于配置 python 整个程序的运行环境  
* 经常与 os 包搞混
* 该包的内容更贴近于 python system

## 2.1. 标准流对象

* sys.stdin     stdin 用于所有交互式输入 （包括`input()`）
* sys.stdout    stdout用于所有输出( 包括`print()`, `Input()`的提醒信息, 以及)
* sys.stderr    error messages 输出
这些标准流对象都和 `open()` 返回的对象类似, 都可以进行作为参数传入相关函数  

可以将输出流或者错误流赋值 `None` 或者 linux 标准垃圾箱 `/dev/null` 来减少输出  

以下对象保存了程序在一开始运行(也就是系统赋予的)的 std* 的初始值  
可以确保在 std* 被重定向后仍然能重定向回来  
* `sys.__stdin__`
* `sys.__stdout__`
* `sys.__stderr__`

## sys.path 库查找路径

`sys.path` 是一个 string 的 list 对象  
* 会参照系统的 `PYTHONPATH` 的内容来初始化
* 执行 python 解释器的时候会自动加入各种安装好的包的路径以及 STL 路径


`sys.path[0]` 是特殊的, 用于存储调用 python 解释器的脚本的路径  
* 如果是以交互式方式启动的 python 解释器或者从标准输入流得到的 python 代码, 则该项目为空字符串
* 几个默认路径, 通过 `site` 包被自动导入  
  * `/lib/python310.zip`
  * `/lib/python3.10`                   python STL
  * `/lib/python3.10/lib-dynload`
  * `/lib/python3.10/site-packages`
* 通过往该路径加入新的元素可以动态扩充包的搜索路径  

## sys.platform 平台识别



# 3. sysconfig python解释器配置

用于指定 python 自己的相关控制参数  


# 4. inspect

审查模组
* 获取活动对象的信息, (模组, 类, 方法, 函数等)
    1. examine the contents of a class
    2. retrieve the source code of a method
    3. extract and format the argument list for a function
    4. get all the information you need to display a detailed traceback.


## 4.1. 类型以及成员



### 4.1.1. 对象自带属性
在该模组的文档中有 python 所有对象的自带属性的[说明](https://docs.python.org/3/library/inspect.html)

## 4.2. 获取源代码

* getsource(obj)

## 4.3. 


# 5. traceback 用于异常溯源


# site Site-specific configuration hook¶

这是一个会被 python 解释器自动导入的包  
* This module is automatically imported during initialization.
* The automatic import can be suppressed using the interpreter’s -S option.
* 用于自动导入几个 python 解释器默认的路径  