# Python Language Services

用于与 python 语言进行工作, 似乎与编译原理中的语法有关    



# 超小型包

完全没什么大的内容的包

## keyword — Testing for Python keywords

与 python 的关键字相关的包, 可以获取 python 的关键字列表

* `keyword.iskeyword(s)`        :用于判断一个字符串是否是 python 关键字  
* `keyword.kwlist`      : 存储了 python 解释器的所有定义的 keywords, 包括了只有在 `__future__` 环境下才会启用的关键字


Python 3.9 新加入了 softkeyword 概念 , 模组内并未阐述 softkeyword 相关的概念

* `keyword.issoftkeyword(s)`    : 判断一个字符串是否是 softkeyword
* `keyword.softkwlist`          : 同 kwlist 类似