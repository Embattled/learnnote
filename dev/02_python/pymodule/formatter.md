# python formatter

没有找到 C 和 python 通用的格式化程序, 记录一下 vscode 中提及的几个 formatter
* black
* isort
* autopep8
* YAPF


# black

* Black is used by some very popular open-source projects, such as pytest, tox, Pyramid, Django Channels, Poetry, and so on.
* which has propelled its usage amongst developers who don't want to think about style, yet want to follow a consistent style guide.

风格固定, 在不想仔细考虑自己的风格, 但又想有固定风格的开发者中比较受欢迎


# autopep8

PEP 8 就是 python 官方推荐的 style, 尽管 autopep8 不是官方发布的套件, 却仍然好用且受欢迎


# YAPF

`Yet Another Python Formatter`, 相对来说比较智能的格式化程序
* 算法认为格式上满足 PEP 8 的程序仍然不一定易读
* 因此会对代码进行格式化使之满足YAPF自己的格式style



