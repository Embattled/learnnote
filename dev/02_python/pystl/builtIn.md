# 1. Built-in 

整合了 STL 中不需要额外导入库的, 解释器自带的内容.  

把原本的 Python 笔记中的 built-in 拷贝过来 (2023.12.14)

# 2. Built-in Functions


可以直接使用, 不需要导入某个模块, 解释器自带的函数叫做内置函数  

    在 Python 2.x 中, print 是一个关键字；到了 Python 3.x 中, print 变成了内置函数。

不要使用内置函数的名字作为标识符使用, 虽然这样做 Python 解释器不会报错, 但这会导致同名的内置函数被覆盖, 从而无法使用  

其他未学习的内置函数
```
 hash()    
 dir()    next()  slicea()
ascii()  divmod()   object()  
      staticmethod()
breakpoint() 
  filter()  issubclass()   super()
    iter()  
callable()  format()  len()  property() 
frozenset()     
classmethod()  
compile()    map()   
complex()  round()
```
## 2.1. 基本函数

过于语言基本的函数, 无法分类

- type() : 输出变量或者表达式的类型
- id()  : 获取变量或者对象的内存地址
- help(): 输出用户定义参数对象的说明文档

执行一个字符串形式的 Python 代码:  

- exec(source, globals=None, locals=None, /) : 执行完不返回结果
- eval(source, globals=None, locals=None, /) : 执行完要返回结果

导入包的内置函数, 用于处理名称里带空格或者首字母是数字的特殊名称模块
- `__import__()`


## 2.2. 内置类函数

这些内置函数实际上是内置类的定义函数, 具体用法查看下文的内置类  

- dict()
- list()
- tuple()
- set()
- 二进制序列
  - bytes()
  - bytearray()
  - memoryview()

## 2.3. 类型转换函数

虽然 Python 是弱类型编程语言, 在一些特定场景中, 仍然需要考虑到类型  

- 确认一个变量的类型
  - isinstance(object, classinfo)   如果 object 的类型属于 classinfo 中的一种, 返回 True

1. 因为python默认读入的内容都被识别为字符串, 因此需要类型转换
   - int(x,n)  将 x 转换成整数类型
   - float(x)  将 x 转换成浮点数类型
   - bool()

2. 使用print()输出时, 通过一行语句输出字符串和变量内容但不使用格式化字符串时, 需要转换
   - str(x)  将 x 转换为字符串
   - repr(x)  将 x 转换为表达式字符串

3. ASCII码函数
   - ord() : 字符转化成 Unicode 码
   - chr(integer ) : 数字的 Unicode 码 转化成字符

4. 进制转换
   - 除了转换成10进制的数, 其他的进制转换结果都带有前缀, 可以使用切片去除前缀
   - hex(x,n)  : 将一个整数 x 转换为一个十六进制字符串, n代表原本的进制
   - oct(x,n)  : 将一个整数 x 转换为一个八进制的字符串, n代表原本的进制
   - int(x,n)  : 将一个整数或 (字符串,bytes,bytearray) x 转换为整数, n代表原本的进制
   - bin(x,n)  : 将一个整数 x 转换为一个二进制的字符串, n代表原本的进制

## 2.4. 作用域变量获取函数

- vars(object) : 返回object 的 `__dict__` 属性, 如果object没有该属性则报异常
- locals()    : 将局部空间的所有变量以字典形式返回, 等同于 `vars(空)`
- globals()   : 将全局空间的所有变量以字典形式返回, 注意以本 module 为基准, 不包括调用的module

## 2.5. 类/包相关的函数  

attr系列
* `getattr(object, name, default)`  : 从一个 object 中提取名为 name 的成员, name 必须是  string, 如果设置了 default 则 object 中不存在该成员的时候会返回该 default, 否则报 `AttributeError`
* `setattr(object, name, value)`    : 为一个 object 设置名为 name 的成员, 如果对象允许的话, 会把值赋予对应的 name 成员
* `hasattr(object, name)`           : 简单的逻辑判断 object 中是否有名为 name 的成员
* `delattr(object, name)`           : 可以理解为 setattr 的反函数, 删除名为 name 的成员, 相当于 `del object.name`


## 2.6. 数学逻辑函数


数学函数
* `pow(base, exp, mod=None)`    : 返回 base 的 exp 次方, 如果数值过大的话可以传入 mod, 会返回结果的取模值.  传入2个参数的时候与 `base**exp` 的功能相同
  * 注意, 如果 base 是整数 , exp 是负数, 则返回值是小数
  * 如果 base 是负数或 float, 而 exp 是非整数, 则返回值是 复数
  * 如果 mod 存在, 而 exp 是负数, 则 base 必须与 mod 互质, 此时会计算 模的逆运算?? 没看懂
* `abs(x)`      : 简单的计算绝对值, x 必须是 integer or float. 或者一个实现了 `__abs__()` 接口的类
  *  If the argument is a complex number, its magnitude is returned.
* 


值筛选函数
* min max 支持两种传值, 单个 iterable 传值或者多个位置参数直接传值
* defalut 用于 iterable 为空的时候的返回值, 如果不设置的话 iterable 为空的时候会返回 ValueError
* key 用于指定排序的方法 
* max
  * `max(iterable, *, default, key=None)`
  * `max(arg1, arg2, *args, key=None)`
* min
  * `min(iterable, *, default, key=None)`
  * `min(arg1, arg2, *args, key=None)`



逻辑函数
* `all(iterable)`   : 如果所有的成员都是 True, 或者 参数本身是一个空的 iterable, 则返回 True
* `any(iterable)`   : 与 all 相对, 如果所有的成员都是 False 或者是空的 iterable 则返回 False


## 2.7. print()

`print (value,...,sep=' ',end='\n',file=sys.stdout,flush=False)`  

- value 参数可以接受任意多个变量或值, 因此 print() 函数可以输出多个值
- `end` 参数的默认值是 `\n`
- `sep` 参数的默认值是 `` 空格
- `file` 参数的默认值为 sys.stdout, 代表标准输出, 即屏幕输出  
- `flush` 参数用于控制输出缓存, 该参数一般保持为 `False` 即可, 这样可以获得较好的性能
  - 根据系统的不同, 输出缓冲区的大小也不同, 存在某些时候要输出程序进度的时候, 如果不能立即输出的话会影响到功能, 因此如果需要立即输出的时候该参数置为 True

```py
print("读者名：",user_name,"年龄：",user_age)

#打开文件
f = open("demo.txt","w")
print('123456',file=f)
```

### 2.7.1. 格式化输出

类似于C语言的格式化输出, print() 函数提供了类似的功能

- 在 print() 函数中, 由引号包围的是格式化字符串, 它相当于一个字符串模板
- 格式化字符串中的`%` 是转换说明符（Conversion Specifier）只是一个占位符
- 转换说明符会被后面表达式替代
- `%` 也被用作分隔符, 格式化字符串后接一个`%` 再接输出的表达式
- 格式化字符串中可以包含多个转换说明符, 这个时候也得提供多个表达式, 此时多个表达式必须使用小括号( )包围起来

```py
name = "C语言中文网"
age = 8
url = "http://c.biancheng.net/"
print("%s已经%d岁了, 它的网址是%s。" % (name, age, url))
```

转换说明符的各种格式和C语言基本一致
| 转换符 | 解释方法                               |
| ------ | -------------------------------------- |
| %d、%i | 转换为带符号的十进制整数               |
| %o     | 转换为带符号的八进制整数               |
| %x、%X | 转换为带符号的十六进制整数             |
| %e     | 转化为科学计数法表示的浮点数（e 小写） |
| %E     | 转化为科学计数法表示的浮点数（E 大写） |
| %f、%F | 转化为十进制浮点数                     |
| %g     | 智能选择使用 %f 或 %e 格式             |
| %G     | 智能选择使用 %F 或 %E 格式             |
| %c     | 格式化字符及其 ASCII 码                |
| %r     | 使用 repr() 函数将表达式转换为字符串   |
| %s     | 使用 str() 函数将表达式转换为字符串    |

### 2.7.2. 控制输出方式

转换说明符的`%`和类型字符中间可以加入控制内容

- 数字指定最小输出宽度  `%10d`
  - 当数据的实际宽度小于指定宽度时, 会在左侧以空格补齐
  - 当数据的实际宽度大于指定宽度时, 会按照数据的实际宽度输出, 不会裁剪
- 对齐标志指定对齐方法, 加在`数字之前`, 默认是左侧补空格, 数据靠右边
  - `-` 指定左对齐,
  - `+` 指定数字数据带正负号 `+-`, 只对数字数据生效
  - `0` 指定数据用 0 代替空格补足, 只对数字数据生效
    - 左对齐时因为右侧补零会改变整数的数值, 因此左对齐时不对整数生效
- 对小数指定精度, 放在`最小宽度数据之后, 用点号隔开`  
  - `%m.nf` `m` 表示最小宽度, `n` 表示输出精度, `.`是必须存在的, f表示小数输出

```py
n = 1234567
print("%10d." % n)
print("%5d." % n)

f = 3.141592653
# 最小宽度为8, 小数点后保留3位
print("%8.3f" % f)
# 最小宽度为8, 小数点后保留3位, 左边补0
print("%08.3f" % f)
# 最小宽度为8, 小数点后保留3位, 左边补0, 带符号
print("%+08.3f" % f)


```

## 2.8. input()

字符串形式接受用户输入  

`str = input(tipmsg)`  

- str   : 表示输入存入的变量
- tipmsg: 表示在控制台中输出的提示信息, 提示输入什么内容

## 2.9. open() 基础文件操作

open() 函数用于创建或打开指定文件, 返回一个 `file object`  
`open(file, mode='r', buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None)`  

- file    : 一个路径, 最好是 path-like oject
- mode    : 文件模式
- buffering: 设置 buffering policy, 一个整数.
- encoding: 只能在文本模式下使用, 用字符串代表编码格式
  
文件模式

- r读取 w写入 +读写 a谁家
- b二进制, t 文本
- x 如果文件已存在则失败

file object 是python的一个内置类, 包含了一些基础函数用于读写文件
### 2.9.1. read

* f.read(size)
* f.readline()


如果是按行来读取文件, 可以使用 for 循环
This is memory efficient, fast, and leads to simple code:
```py
for line in f:
  print(line, end='')
```
### 2.9.2. write

* f.write(str)

### 2.9.3. position

* f.tell()
* f.seek(offset,whence)

## 2.10. 序列类型数据的操作函数
### 2.10.1. range()

- range() 函数能够轻松地生成一系列的数字  
- 函数的返回值并不直接是列表类型 list  而是 range
- 如果想要得到 range() 函数创建的数字列表, 还需要借助 list() 函数

```py
# 参数分别是 
# 初始值
# 结束值 , 数字列表中不会包含该值, 是开区间
# 步长
even_numbers = list(range(2,11,2))

```
### 2.10.2. zip()

- 可以将多个序列（列表、元组、字典、集合、字符串以及 range() 区间构成的列表）“压缩”成一个 zip 对象
- 所谓“压缩”, 其实就是将这些序列中对应位置的元素重新组合, 生成一个个新的元组
- 函数“压缩”多个序列时, 它会分别取各序列中第 1 个元素、第 2 个元素、... 第 n 个元素, 各自组成新的元组
- 需要注意的是, 当多个序列中元素个数不一致时, 会以最短的序列为准进行压缩

```py
# iterable,... 表示多个列表、元组、字典、集合、字符串, 甚至还可以为 range() 区间
zip(iterable, ...)


my_list = [11,12,13]
my_tuple = (21,22,23)
print([x for x in zip(my_list,my_tuple)])
# [(11, 21), (12, 22), (13, 23)]

my_pychar = "python"
my_shechar = "shell"
print([x for x in zip(my_pychar,my_shechar)])
# [('p', 's'), ('y', 'h'), ('t', 'e'), ('h', 'l'), ('o', 'l')]

```

### 2.10.3. reserved()

- `reversed(seq)` 并不会修改原来序列中元素的顺序
- 对于给定的序列（包括列表、元组、字符串以及 range(n) 区间）, 该函数可以返回一个逆序列表的`迭代器`

```py
print([x for x in reversed([1,2,3,4,5])])
# [5, 4, 3, 2, 1]


print([x for x in reversed((1,2,3,4,5))])
# [5, 4, 3, 2, 1]
# 逆序元组, 返回的还是一个列表
```

### 2.10.4. sorted()

- `list = sorted(iterable, key=None, reverse=False)`
- iterable 表示指定的序列
- key 参数可以自定义排序规则
- reverse 参数指定以升序（False, 默认）还是降序（True）进行排序
- sorted() 函数会返回一个排好序的**列表**
- 字典默认按照key进行排序

```py
a = "51423"
print(sorted(a))
# ['1', '2', '3', '4', '5']


chars=['http://c.biancheng.net',\
       'http://c.biancheng.net/python/',\
       'http://c.biancheng.net/shell/',\
       'http://c.biancheng.net/java/',\
       'http://c.biancheng.net/golang/']
#自定义按照字符串长度排序
# 传入 key 等于一个 lambda 函数
print(sorted(chars,key=lambda x:len(x)))
```

### 2.10.5. enumerate() 遍历对象函数

- 一般用在 for 循环中, 将一个可遍历的数据对象组合成一个索引序列  
- 可以同时列出数据和数据下标

```py
# start 代表起始位置
enumerate(sequence, [start=0])

# 返回枚举对象
seq = ['one', 'two', 'three']
for i, element in enumerate(seq):
   print i, element
```
### 2.10.6. sum() 提取并相加迭代器的所有元素

`sum(iterable, /, start=0)`
Changed in version 3.8: The start parameter can be specified as a keyword argument.
* start 函数用于对所有元素的这个初始值进行指定
* 对于数字的 iterable, start 默认值0即可
* 对于字符串相连, `start=''`
* 对于list相连, `start=[]`
* 不正确的指定 start的话, 会导致 + 运算符的失效


python官方提供了三个更推荐使用别的函数的使用场景, 可以获得更快的速度
* 把一组字符串链接起来, 使用 `''.join(sequence)`
* 把小数数据提高精度, 使用 `math.fsum()`
* 把数个 iterables 链接起来, 使用 `itertools.chain()`

# 3. Built-in Constants

# 4. Built-in Types


Built-in Types 代表那些包括在了 python 解释器里的变量, 不需要任何包(包括STL), 即可正确处理

主要的 built-in type 可以按照以下分类:
* numerics
* sequences
* mappings
* classes
* instances
* exceptions

这些数据类型包括集合类型, 也有成员可变集合类型

所有的内置成员变量都实现了以下功能
* 可以互相之间比较 equality
* Truth Value Testing
* 转换成 string 类型, 包括
  * str()  , 作为 print() 的参数被输出时会隐式调用
  * repr()
* 一些集合类 collection classes 是可变的, 即 对实例进行 增减重排时采用 in place 方法, 返回None

## 4.1. Numeric Types — int, float, complex 数值类型

总体上数值可以分成三大类, 同时 Booleans 属于一个 subtype of integers
* Python 中的 integers 拥有 unlimited precision
* Python 中的 float 通常情况下是基于 C 的 double 实现的
  * 具体情况可以查看 sys 里的特定标识 `sys.float_info`
* complex 中实部虚部分别是一个 float
  * 通过在数字后追加 `j` `J` 来定义一个只有实部的复数
* python stl 中还定义了其他的数字类型

三个构造函数 `int() float() complex()` 可以用来定义指定的 type

### 4.1.1. Bitwise Operations on Integer Types - 整数上的 bit 操作


## 4.2. Sequence Types — list, tuple, range   序列类型

序列是Python所有核心序列的统称, 包括字符串、列表、元组、集合和字典  
全部支持: 存在检查, 遍历  

- 标准序列: 字符串 str 、列表 list 、元组 tuple , 支持索引、切片、相加和相乘
- 二进制序列: bytes
- 广义序列: 集合和字典,        不支持索引、切片、相加和相乘

### 4.2.1. 序列通用
#### 4.2.1.1. 序列内置函数

因为是序列的内置函数, 因此字符串列表元组字典集合都有这些函数  

| 函数        | 功能                                          |
| ----------- | --------------------------------------------- |
| len()       | 计算序列的长度, 即返回序列中包含多少个元素。  |
| max()       | 找出序列中的最大元素。                        |
| min()       | 找出序列中的最小元素。                        |
| list()      | 将序列转换为列表。                            |
| str()       | 将序列转换为字符串。                          |
| sum()       | 计算元素和。                                  |
| sorted()    | 对元素进行排序。                              |
| reversed()  | 反向序列中的元素。                            |
| enumerate() | 将序列组合为一个索引序列, 多用在 for 循环中。 |

- 注意, 对序列使用 sum() 函数时, 做加和操作的必须都是数字, 不能是字符或字符串, 否则该函数将抛出异常
- 因为解释器无法判定是要做连接操作（+ 运算符可以连接两个序列）, 还是做加和操作。

#### 4.2.1.2. 序列的引用

对于变量保存字符串和整数值, 变量保存的是值本身  
对于列表等序列对象, 变量保存的是列表引用, 类似于指针  

```py
spam=[1,2,3]
cheese=spam
```

1. 在函数调用时, 对于列表参数是传递引用, 因此需要注意可能的缺陷
2. copy模块,有时候为了不影响原来的列表或字典,使用copy模块,确保复制了列表
   - `spam=['A','B','C']`
   - `cheese=copy.copy(spam)`

#### 4.2.1.3. 存在检查 in

`value in sequence`  
可以和 not 结合 `not in`

```py
str="c.biancheng.net"
print('c'in str) # True
```

#### 4.2.1.4. 索引

- 最基础的操作, 方括号`[]`配合数字访问元素  
- python是 `0索引`
- python支持负数索引,  `[-n]` 等同于 `[len()-n]`

#### 4.2.1.5. 切片

访问一定范围内的元素, 通过切片操作生成一个新的对应序列  

`sname[start : end : step]`  

- sname : 序列
- start : 开始索引位置, 包括该位置 ,默认0
- end   : 结束索引位置, **不包括该位置**, 默认为序列的长度
- step  : 切片间隔, 默认为1, 默认值的时候可以省略第二个冒号`:`

- 切片下标的省略`[:2] [1:] [:}` 分别代表从头开始,到末尾,全部

#### 4.2.1.6. 序列相加

两种**类型相同**的序列使用 `+` 运算符做相加操作  
它会将两个序列进行**连接**, 但不会去除重复的元素  

#### 4.2.1.7. 序列相乘

使用数字 `n` 乘以一个序列会生成新的序列, 其内容为原来序列被重复 `n` 次的结果

### 4.2.2. 列表类型 list

- 列表类型是最基础的序列类型, 其他的序列类可以用 `.list()` 转化为 list  
- 所有序列都有 `.list()` 成员就可想而知列表的基础性
- Python的列表取代了其他语言的数组  

- 列表值: 指的是列表本身,可以作为值保存在变量中,或者传递给函数
- 表项:   指的是列表内的值,用逗号分隔
- 使用下标访问列表中的每个值 `spam[0]` 若超出列表中值的个数,将返回 `IndexError`
- 列表中包含其他的列表值,构成多重列表,使用多重下标访问 `spam[0][1]`
- 负数下标: 使用负数来代表倒数的索引, `[-1]` 代表列表中倒数第一下标

#### 4.2.2.1. 列表的创建使用

python的列表总是动态的

```python
# 使用 [ ] 直接创建列表
catName=[]
while True:
    catName=catName+[name]
for name in catName:
    print(name)


# 使用内置函数 list() 函数创建列表
list1 = list("hello")
print(list1)

#将字典转换成列表会丢失vlaue
dict1 = {'a':100, 'b':42, 'c':9}
list3 = list(dict1)
# ['a', 'b', 'c']
print(list3)
```

#### 4.2.2.2. 元素添加

1. 序列中提到的, `+` 可以连接多个列表, 运算返回新列表不会改变原有列表, 但是效率低
2. .append(obj) 成员函数在末尾添加元素, obj 可以是元组或列表但都会被当做一整个元素
3. .extend(obj) 成员函数在末尾添加元素, 会读取 obj 内部的单个元素一一添加
4. .insert(index,obj) 在指定位置插入元素, obj 是一整个元素

```py
l = ['Python', 'C++', 'Java']
t = ('JavaScript', 'C#', 'Go')

l.append(t)
# ['Python', 'C++', 'Java', 'PHP', ('JavaScript', 'C#', 'Go')]

l.extend(t)
# ['Python', 'C++', 'Java', 'C', 'JavaScript', 'C#', 'Go']

l.insert(1, 'C')
# ['Python', 'C', 'C++', 'Java']
```

#### 4.2.2.3. 元素删除

1. del 关键字, 专门执行删除操作
   - 可以删除整个列表, 但是python有垃圾自动整理, 不常用
   - 可以删除列表单个元素 `del listname[index]`
   - 可以删除切片 `del listname[start : end]`
2. .pop(index) 成员函数, 删除指定索引处的元素并返回
   - 注: python 没有提供 push 函数, pop 也属于广义上的 pop
   - 不提供参数的时候, 和普通的 pop() 一样返回的是末尾的函数
3. .remove(value) 成员函数 , 按照值来删除元素
   - 只会删除第一个和指定值相同的元素
   - 必须保证元素是存在的, 否则会引发报错 `ValueError`
4. .clear() 清空列表

总结:

- 若知道删除的下标,就用 `del spam[2]`
- 在删除的同时返回删除的值,就用`a=spam.pop(2)`
- 若知道要删除的值,就用`spam.remove('a')`

#### 4.2.2.4. 元素修改

1. 用索引直接修改
2. 用切片直接修改多个元素, 但是必须保证输入的元素个数和切片的元素个数相同

```py
nums = [40, 36, 89, 2, 36, 100, 7]
nums[2] = -26  #使用正数索引
nums[-3] = -66.2  #使用负数索引

nums[1: 4] = [45.25, -77, -52.5]

# 带步长的切片也可以使用, 确保个数相同比较麻烦
nums[1: 6: 2] = [0.025, -99, 20.5]
```

#### 4.2.2.5. 列表的查找

1. 除了序列提供的 in 关键字操作, 列表查找元素还有其他方法
2. .count(obj) 统计元素obj在列表中的出现个数
3. .index(obj, start,end) 在范围[start,end) 中查找元素出现的位置
   - 该方法非常脆弱 , 元素不存在的话会报错 `ValueError`
   - 使用前最好用 count() 统计一下

#### 4.2.2.6. 列表的赋值

使用列表可以同时为多个变量赋值  
`size,color,dispo=cat`  
变量的数目和列表的长度必须相等,否则报错  
`cat=['fat','black','loud']`

#### 4.2.2.7. 增强的赋值操作

针对`+、-、*、/、%`操作符  
有`+=、-=、*=、/=、%=`意义同C语言

```python
spam +=1 
spam=spam+1
```

`+=`可以完成字符串或列表的连接  

```python
spam='hello'  
spam+='world'
```

`*=`可以完成字符串或列表的复制  

```python
bacon=['zon']
bacon*=3
```

#### 4.2.2.8. 列表类型的方法

- `remove()`: `spam.remove('bat')` 传入一个值, 将该值从列表中删除, 若该值不在列表中, 返回`ValueError`错误,只会删除第一次出现的位置
  *
- `sort()`:
  - `spam.sort()`数值的列表或者字符串的列表可以使用该方法
  - `spam.sort(reverse=True)` 指定反向排序
  - `sort()` 是当场排序,不是返回值,不能对混合值列表进行排序
  - `sort()` 是按照ASCII表排序,因此在大写是排在小写字母之前
    - 可以使用`spam.sort(key=str.lower)`使方法把所有的表项当作小写
- 函数`sorted()`
  - 可以返回一个排序的值,但是不会影响原本的值 `ab=sorted(spam)`

### 4.2.3. 元组

元组(tuple)是另一个重要的序列, 和列表非常类似

- 列表的元素是可以更改的
- 元组一旦创建就不可更改, 不可变序列

- 转换成元组的函数是 `tuple(data)`  
- data 表示可以转化为元组的数据, 包括字符串、元组、range 对象等。
- data 为字典时转化后会丢失 value

#### 4.2.3.1. 元组创建

元组与列表几乎一样,除了两点  

1. 元组输入时使用`()`而不是`[]` 例:`eggs=('hello',42,0.5)`
2. 元组属于不可变类型, 同字符串一致,即值不能增改,字符串属于元组的一种
3. 为了区分只有一个值的元组与与普通的值,在值的后面添加一个 `,`
4. 元组通常都是使用一对小括号将所有元素包围起来的, 但小括号不是必须的, 只要将各元素用逗号隔开, Python 就会将其视为元组

```py
type ( ('hello',) )
# <class 'tuple'>

type( ('hello') )
# <class 'str'>

course = "Python教程", "http://c.biancheng.net/python/"
# ('Python教程', 'http://c.biancheng.net/python/')
print(course)
```

#### 4.2.3.2. 元组的访问和修改

1. 作为标准序列之一, 支持索引和切片
2. 元组作为不可变序列, 修改时只能用新的元组代替旧的元组
3. 删除时也只能用 `del` 直接删除整个元组

### 4.2.4. 二进制序列 bytes

专门用于处理二进制数据的类型  
包括 bytes, bytearray, memoryview 三种细微不同的子类
* bytes       : 二进制对象的存储类, 具有不可改变性, 类似于元组
* bytearray   : 相比于 bytes, 赋予了可更改的特性
* momoryview  : 用于直接映射到 内存中的二进制对象, 从而不需要拷贝就可以访问

#### 4.2.4.1. bytes


- 非可变类型, 存储单一 byte 的序列
- 拥有语法定义, 可以通过类似于字符串的方式被创建, 但是引号前要加一个字母 b `b"abc"`, 对三种字符串语法都生效
- 只有 ASCII 的字符可以直接转换成 bytes

`class bytes([source[, encoding[, errors]]])`

* 因为一个 byte 可以被表示成 2个16进制数, 所以有很多情况下 二进制数据被16进制数来保存或存储
* 类方法 `bytes.fromhex(str)`
  * 从一个被字符串化的 16进制序列还原出一个 bytes 对象
* 对象方法 `hex([sep[, bytes_per_sep]])`
  * 把一个 bytes 对象转换成 16进制数表示的字符串
  * sep 是每个 byte 之间的间隔符, 一般是空格类
  * bytes_per_sep 表示间隔符的出现频次, 默认是1, 正负号代表从右左开始统计

#### 4.2.4.2. bytearray

- 可变类型
- 无语法定义, 必须用构造函数创建

`class bytearray([source[, encoding[, errors]]])` 
- `bytearray()` 创建空 bytearray
- `bytearray(10)` 创建 bytearray 并给每个 byte 填入 0
- `bytearray([1,255])` 创建对应内容的 bytearray 需要给每个字节填入数据 1~255, `bytearray(b'\x01\xff')`

<!-- ### bytes类 对象方法 -->



## 4.3. Text Sequence Type — str  字符串

最基本的将object转为string的函数：

- str() 保留了字符串最原始的样子
- repr() 用引号将字符串包围起来, Python 字符串的表达式形式

字符串是一个 python 内建类:
* python的字符串使用 `''` 单引号输入, 也可以使用 `""`双引号输入, 区别在于使用双引号时字符串中可以包括单引号
* 所有的对应语句都会直接转换成字符串类, 并且可以使用对应的类方法


### 4.3.1. 字符串的输入

- 转义字符,使用`\` 反斜杠来输入特殊字符,  `\t` 制表位, `\n` 换行符, `\\` 反斜杠
- 原始字符串,在字符串前面加一个`r`,例如 `r'abc'` ,可以完全忽略字符串中的所有反斜杠
- 多行字符串,对于有换行的字符串可以更为方便的输入
- raw字符串,在字符串前加一个 'r' 代表rawString
   1. 取消对反斜杠\ 的转义
   2. 如果一个字符串包含很多需要转义的字符, 对每一个字符都进行转义会很麻烦。为了避免这种情况,我们可以在字符串前面加个前缀r, 表示这是一个 raw 字符串, 里面的字符就不需要转义了。
- 多行字符串`'''abc'''`,字符串可以跨过多行
   1. 三引号还会对其内的单双引号起作用
   2. 可以将raw字符串和多行字符串结合起来使用
- 字符串的编码类型
   1. python中字符串默认采用的ASCII编码,
   2. 如果要显示声明为unicode类型的话, 需要在字符串前面加上'u'或者'U' `print(u'字符串')`
   3. 如果中文字符串在Python环境下遇到 UnicodeDecodeError,在源码第一行添加注释`# -*- coding: utf-8 -*-`

```
>>> print(r'\n')
\n
```

```python
abc='''this is a long
string with multiline.
'''
```

字符串可以使用切片以及 `in`  `not in` 操作符,用来比较前一个字符串是否在后一个字符串中间
  
### 4.3.2. 字符串编码

字符串有一个方法 `.encode()` 可以将字符串使用对应格式进行编码, 返回一个对应的 bytes 类型

#### 4.3.2.1. 大小写及内容检测方法

- `upper()` 和 `lower()` 返回一个**新字符串**,将原本字符串中所有字母转变为大写/小写
- `name.title()`  标题方法, 将字符串的单词第一个字母大写  
- `isupper()` 和 `islower()` 返回布尔值,如果这个字符串不为空且全部字符为大写/小写则返回`True`
- 其他的 isX 方法,返回True的情况
  - isalpha()  非空且只包含字母
  - isalnum()  非空且只包含字母和数字
  - isdecimal() 非空且只包含数字字符
  - isspace()  非空且只包含空格,制表符,换行
  - istitle()  非空且只包含以大写字母开头,后面都是小写字母的 `单词` 及可以包含空格及数字

#### 4.3.2.2. 开头结尾检测方法  

   startswith() 和 endswith()
   以传入的字符串为开始/结尾,则返回True

#### 4.3.2.3. 组合与切割方法  

   join() 和 split()
    join()是将一个字符串列表连接起来的方法,并在字符串中间插入调用`join`的字符串  
    `','.join(['a','b','c'])`   返回 ` 'a,b,c' `  
    split()则相反,在被拆分的字符串上调用,默认以各种空白字符分隔  
    `'My name is'.split()`   返回 `['My','name','is']`  
    常被用来分割多行字符串  `spam.split('\n')`

#### 4.3.2.4. 对齐方法  

   rjust()  ljust()  center()
   在一个字符串上调用,传入希望得到的字符串的长度,将返回一个以空格填充的字符串  
   分别代表左对齐,右对齐  
   `'a'.rjust(5)`  返回  `'    a'`
   `'a'.ljust(5)`  返回  `'a    '`
   可以输入第二个参数改变填充字符  
   `'a'.ljust(5,'*')`  返回  `'a****'`

#### 4.3.2.5. 清除对齐方法  

   `strip()  rstrip() lstrip()`  
   在左右删除空白字符  
   传入参数指定需要删除的字符  注:这里第二个参数无视字符出现顺序  
   `'abcccba'.strip('ab')` 与
   `'abcccba'.strip('ba')` 作用相同

#### 4.3.2.6. 识别转换方法 **str.extract()**

和pandas组合使用,拆解字符串

```py

# Extracting ID into ['faculty','major','num']
id_ext = df['id'].str.extract('(\w{3})-(\w{2})(\d{2})',expand=True)
id_ext.columns = ['faculty','major','num']


# Extracting 'stats' into ['sex', 'age']
stats_ext = df['stats'].str.extract('(\w{1})_(\d{2})years',expand=True)
stats_ext.columns = ['sex', 'age']
```

#### 4.3.2.7. str.get_dummies()

为了对数据进行统计,例如多个学生选择了多种科目,科目可能会重复,这种统计  

```py
'''
name: course
Ali: Maths, English, Algebra, Geometry
Bea: Science, Biology, Physics, Chemistry, Maths, Music

'''
stud_course = pd.read_csv('student_course.csv', sep=":", skipinitialspace=True)


# Converting course_name into dummy variables
course_dummy_var = stud_course['course'].str.get_dummies(sep=', ')
print(course_dummy_var)

'''
  Algebra  Art  Biology   ...     Phys. Ed.  Physics  Science
0         1    0        0   ...             0        0        0
1         0    0        1   ...             0        1        1
'''

```
## 4.4. Mapping Types — dict 字典

- 类似于 C++ 的 map  但是字典是无序的, 没有**第一个**的概念
- 字典的键可以使用许多不同的数据类型
  - 不只是整数, 可以是字符串或者元组, 但是不能是列表
  - 键定义后即不能改变
- 可以任意深度的嵌套

- dict 的 in 和 not in 运算都是基于  key 来判断的
  
### 4.4.1. 字典创建删除

基本定义方法:

- 所有元素放在大括号`{}` 中
- 键和值之间使用冒号 `:` 分隔
- 相邻元素之间使用逗号 `,` 分隔

函数定义方法:

- dict.fromkeys(list,value=None) 带有默认值的创建
- dict() 映射函数, 默认创建空字典
  - a = dict(str1=value1, str2=value2) str 表示字符串的键, 必须是变量输入, 不能是引号输入
  - `demo = [('two',2), ['one',1]] a = dict(demo)` 传入一系列二元列表或元组
  - keys = 序列 values =序列  `a = dict( zip(keys, values) )`
  - dict() 和 zip(组合构成字典)

```py
# 标准创建
dictname = {'key':'value1', 'key2':'value2', ..., 'keyn':'valuen'}

#创建空字典
dict2 = {}

# 使用元组和数字作为key 键可以是不同类型
# 但是列表不能作为key, 因为列表天生是可变的
dict1 = {(20, 30): 'great', 30: [1,2,3]}

# 加入一个新的键值对:
ice_cream_1['flavor'] = 'vanilla'
# 删除一个键值对
del ice_cream_1['flavor'] 

# 带默认值的字典
knowledge = ['语文', '数学', '英语']
scores = dict.fromkeys(knowledge, 60)
# {'语文': 60, '英语': 60, '数学': 60}
print(scores)

```

### 4.4.2. 访问使用

字典有三个打印方法

- 这些方法返回的不是真正的列表(是 view) 不能被修改, 不能使用 append() 方法
- `keys()`  :返回键值
- `values()`:具体值
- `items()` :键-值对

- 但可以用于 `for` 循环  
- 可以使用 `list(spam.keys())` 将返回的值转换为列表  
- 类型分别为 dict_keys、dict_values 和 dict_items

安全访问方法: `get(key[, default])`

- 可以安全的通过索引访问一个值,当这个索引在字典中不存在时返回备用值  
- 若不使用`get`方法直接用索引会报错
- `print( a.get('one') )`

### 4.4.3. 成员函数操作

1. `.copy()`
   - 返回一个表层深拷贝的 dict
   - 即如果 value 是一个列表, 那么根据列表的引用原则, 列表的元素是新旧dict 共享内存的
2. `setdefault(key,defaultvalue)`
   - 可以安全的初始化一个键
   - 当该键存在则返回键值
   - 键不存在则设定为第二个参数, 并返回第二个参数
3. `.update(dict)`
   - 传入参数是一个 dict
   - 已有的key会被更新value
   - 不存在的key会被创建并赋值
4. `.pop(key)`
   - 弹出对应的键值对
5. `.popitem()`
   - 弹出dict内部的最后一个键值对
   - 表面上看是无序的

## 4.5. set Types — set, frozenset

1. 集合（set）是一个无序的**不重复元素**序列, 相当于没有value的字典
2. 无法存储列表、字典、集合这些可变的数据类型

### 4.5.1. 创建

1. 通过花括号直接创建
2. 注意: 创建一个空集合必须用`set()` 而不是 `{ }`, 因为` { } `是用来创建一个空字典  
3. `set()` 函数实现
   - 将字符串、列表、元组、range 对象等可迭代对象转换成集合

```py
setname = {element1,element2,...,elementn}

a = {1,'c',1,(1,2,3),'c'}
```

### 4.5.2. 访问

- set 也是无序的, 因此不能使用下标

```py
a = {1,'c',1,(1,2,3),'c'}
for ele in a:
    print(ele,end=' ')
# 1 c (1, 2, 3)
```

### 4.5.3. 元素操作

1. 添加元素 `s.add( x )`
   - 如果 x 已经存在于集合, 则无改变,
   - `thisset.add("Facebook")`
2. 更高级的添加元素 `s.update( x )`
   - 这里 x 可以可以是列表, 元组, 字典等, x 还可以有多个用逗号分开
   - `thisset.update([1,4],[5,6])`
3. 移除元素 `s.remove( x )` 如果元素不存在就会错误`KeyError`
   - 另一个不会发生错误的移除元素方法 `s.discard( x )`
4. 清空集合 `s.clear()`
5. `pop()` 随机移除元素 并返回值

### 4.5.4. 集合运算

python的set作为一个集合, 可以运行数学上的集合运算  

- `&`  交集, 取公共元素
- `|`  并集, 取全部元素
- `^`  差集, A|B 取 A-A&B
- `-`  对称差集, 取 A和B 中不属于 A&B 的元素 A&B-A|B


## 4.6. Iterator 

python 的迭代器概念: 
* 用户可以自己创建能够被迭代的自定义类
* 如下两个内部函数构成了 `iterator protocol`

`iterator.__iter__()`
* 返回该迭代器对象本身, 可以说用于证明该对象是迭代器, 可以用于初始化迭代的 count
* This method corresponds to the `tp_iter` slot of the type structure for Python objects in the Python/C API.

`iterator.__next__()`
* 返回下一个元素, 如果已经遍历完成, 唤起 `StopIteration` exception
  * 如果再次被调用 `__next__`, 必须持续的唤醒该 exception
* This method corresponds to the `tp_iternext` slot of the type structure for Python objects in the Python/C API.

`container.__iter__()`
* Return an iterator object.
* 每太看懂, 可能是所谓的 container 类的内部实现是通过定义别的 iterator 对象, 所以该方法就是返回该类内部具体的 迭代对象attribute

### 4.6.1. Generator Types



# 5. Built-in Exceptions

