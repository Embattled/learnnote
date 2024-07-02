# 1. Text Processing Services

字符串操作处理的模组
a wide range of string manipulation operations and other text processing services.

* `Binary Data Services` 下面的 `codecs` 与该模组集合高度关联
* built-in 类型 str 与 string 包高度关联

# 2. string — Common string operations

定义了一些用于辅助 内置类 str 的方法的函数, 还有一些用于快速书写正则表达式的常量 

# 3. re — Regular expression operations

标准 python STL 正则表达式库, 一般足够使用. 在此基础之上还有外部库 `regex`, 提供了与 `re` 的完整兼容以及对 Unicode 的更好的支持

要在python中使用正则表达式, 需要导入`re`包  
`import  re`  

官方文档[https://docs.python.org/3/library/re.html]

## 3.1. 使用正则表达式的基础函数

```python
# 基础search函数,用来查找表达式第一次出现的位置
m = re.search(regex, text)
print(m)   # 查找的结果是这样的一个体 <_sre.SRE_Match object; span=(6, 8), match='in'>
print(m.span()) # (6, 8) 返回一个元组, 包含了两个值, 分别是匹配的开始位置索引和结束位置索引
print(m.group()) # 'abc'  直接返回匹配的字符串内容

# 与search不同, findall返回一个列表, 表项就是匹配的字符串内容, 若无匹配的内容则返回空列表
re.findall(regex, text)

# re.split使用正则表达式的分割字符串, 返回分割后的字符串列表  
x = re.split(' ', text)

# re.sub用来进行字符串替换  
x = re.sub('被替换的字符串', '填充的字符串', text)

```

## 3.2. 正则表达式-单字符

```python
# 使用[]来代表一个单字符的可选列表
# 匹配sta/stb/stc
re.search('st[abc]', 'understand')

# 使用破折号代表范围
re.search('st[0-9a-fA-F]', 'best_first99')

# 在[]中使用^来表示不包含的符号
# 匹配一个非数字的字符
re.search('[^0-9]', 'str1ke_one')

# 使用'\' 来进行关键字转义
# 查找 e-o
re.search('e\-o', 'str1ke-one')

# 万能字符 . , '.'表示任意一个单字符
re.search('.e', 'str1ke^one')

# 字符集合
# \w is equal to [a-zA-Z0-9_]
# \W is equivalent to [^a-zA-Z0-9_].
# \d is equal to [0-9].
# \D is equal to [^0-9]
# \s matches any whitespace character (including tab and newline). 
# \S is the opposite of  \s  It matches any character that isn't whitespace.


# 位置字符
# 使用^或\A来代表字符串起始位置
re.search('^fi', 'first')

# 使用$或\Z来代表字符串末尾
re.search('st$', 'first')

# \b 来同时表示以上两种, 代表边界位置
# \B does the opposite of, 表示处于字符串中间
re.search(r'\bst', 'first strike')
re.search(r'\Bst\B', 'strike first estimate')
```

## 3.3. 正则表达式-多次匹配

```python
#  * 代表任意次  + 代表至少1次 ,? 代表0或者1次
# * 和 + 和 ?都是默认贪心, 查找出现最多的字符串段 
# *? 和 +? ??来指定多字符匹配会选择重复次数最少的片段


# 查找纯数字片段
re.search('[0-9]*', '1st_strike')

# 使用{} 来直接指定前一个字符的匹配重复次数
re.search('x\d{3}x', 'x789x')

# 使用{m,n} 来指定重复次数区间
# 同样默认是贪心匹配, 使用{m,n}? 来代表最小匹配
# {,} 省略m或者n的数字来代表最小0次和最大无限次

re.search('x-{2,4}x', 'x--xx----x')
re.search('x-{2,4}?x', 'x--xx----x')

# 使用()来直接指定匹配的字符串片段,多字符匹配
re.search('(bar)+', 'foo barbarbar baz')

```

# difflib — Helpers for computing deltas
