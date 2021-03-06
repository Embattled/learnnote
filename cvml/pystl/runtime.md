# 1. Runtime Services

该分类的模组用于提供 Python 解释器和其运行环境, 也就是操作系统之间的交互  

# 2. sys

与 python 解释器密切相关的一些函数与变量  

## 2.1. 标准流对象

* sys.stdin     stdin 用于所有交互式输入 （包括`input()`）
* sys.stdout    stdout用于所有输出( 包括`print()`, `Input()`的提醒信息, 以及)
* sys.stderr    error messages 输出
这些标准流对象都和 `open()` 返回的对象类似, 都可以进行作为参数传入相关函数  


以下对象保存了程序在一开始运行(也就是系统赋予的)的 std* 的初始值  
可以确保在 std* 被重定向后仍然能重定向回来  
* `sys.__stdin__`
* `sys.__stdout__`
* `sys.__stderr__`

# 3. sysconfig

用于指定 python 自己的相关控制参数  


# inspect

审查模组
* 获取活动对象的信息, (模组, 类, 方法, 函数等)
    1. examine the contents of a class
    2. retrieve the source code of a method
    3. extract and format the argument list for a function
    4. get all the information you need to display a detailed traceback.


## 类型以及成员



### 对象自带属性
在该模组的文档中有 python 所有对象的自带属性的[说明](https://docs.python.org/3/library/inspect.html)

## 获取源代码

* getsource(obj)

## 