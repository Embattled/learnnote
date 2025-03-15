# 1. Structured Markup Processing Tools

用于处理各种 structured data markup, 包括:
* work with the Standard Generalized Markup Language (SGML)
* and the Hypertext Markup Language (HTML)
* several interfaces for working with the Extensible Markup Language (XML).


# 2. XML Processing Modules

警告信息: XML 的脆弱性
* https://docs.python.org/3/library/xml.html#xml-vulnerabilities
* https://docs.python.org/3/library/xml.html#defusedxml-package

标准XML只允许有1个 root node, 但是有很多地方会使用 pseudo-XML 来存储各种配置文件

Python’s interfaces for processing XML are grouped in the `xml` package.

XML数据的解包需要至少一种可用的 `SAX-compliant XML parse`(?), python 内部自带了一种 (Expat parser).  
因此 `xml.parsers.expat` 作为 Python 自带的 总是可用, 所以也应该是最常用的.  

The XML handling submodules are: 包括了各种不同的XML读取处理方法, 但是本质上的目的应该都一样  
* `xml.etree.ElementTree`: 最简单最轻量级的, the ElementTree API, a simple and lightweight XML processor
* `xml.parsers.expat`: Pyhon 自带的, 在任意情况下都可用, the Expat parser binding
* `xml.dom`: the DOM API definition
* `xml.dom.minidom`: a minimal DOM implementation
* `xml.dom.pulldom`: support for building partial DOM trees
* `xml.sax`: SAX2 base classes and convenience functions SAX2基类和便捷函数

## 2.1. XML 漏洞

<!-- 完 -->
XML 处理模块没有鉴别而已构造的数据, 攻击者可能通过滥用 XML 来执行 Dos 攻击, 访问本地文件等

python STL 中5种 xml 处理模块对于各种攻击的耐受性

| 种类                      | sax | etree | minidom | pulldom | xmlrpc |
| ------------------------- | --- | ----- | ------- | ------- | ------ |
| billion laughs            | x   | x     | x       | x       | x      |
| quadratic blowup          | x   | x     | x       | x       | x      |
| external entity expansion | o   | o     | o       | o       | o      |
| DTD retrieval             | o   | o     | o       | o       | o      |
| decompression bomb        | o   | o     | o       | o       | x      |
| 解析大量词元              | x   | x     | x       | x       | x      |



* billion laughs / exponential entity expansion  (狂笑/递归实体扩展)
  * Billion Laughs 攻击, 递归实体拓展, 使用多级嵌套 entry
  * 每一个实体多次引用另一个实体, 最终实体定义仅包含一个小字符串, 但最终会指数级拓展导致几千 GB 的文本, 消耗大量 内存和 CPU时间
  * 所有模块都不具有对该攻击的安全性
* quadratic blowup entity expansion (二次爆炸实体扩展)
  * 类似于 Billion Laughs, 同样滥用了实体拓展, 但不是递归嵌套的, 仅仅是多次重复一个具有几千字符的大型实体
  * 这种攻击力不如 billion laughs, 但是可以绕过对于深度嵌套实体有检测能力的解析器
* external entity expansion
  * 外部资源解析
  * XML 指向外部或者本地文件并嵌入导 XML 种
* DTD retrieval
  * 从远程或者本地检索文档类型定义
  * 该功能与 external entity expansion 具有相似的含义, 指的是同一种攻击
* decompression bomb
  * 解压炸弹
  * 适用于可以解析压缩后的 XML 流的解析器
  * 压缩后传输的数据量将减少 3个量级或更多, 导致解压后的数据爆炸
* 解析大量次元
  * Expat 需要重新解析未完成的词元
  * 在 Expat 2.6.0 之前, 会导致被用来在解析 XML 的应用程序中制造 Dos 攻击

* PyPI 中的 `defusedxml` 
  * 修改了所有 stdlib XML 解析器的子类
  * 可以防止任意类型的恶意操作
  * 对于要解析不信任的 XML 数据的服务器, 推荐使用该软件包
  * 文档包含了已知的 XML 攻击种类和信息, 以及利用示例

补充信息
* 对于 Expat
  * 在 2.4.1 更新后不容易再受到 billion laughs 和 quadratic blowup 攻击的影响, 但由于需要依赖系统提供的库所以仍然有风险
  * 在 2.6.0 更新后不容易收到 解析词元攻击, 但仍然依赖系统提供的库, 所以仍然有风险
* 对于 external entity expansion
  * extree 不会拓展外部实体, 且会引发 ParseError
  * minidom 不会拓展外部实体, 仅简单的返回未拓展状态的实体
  * xmlrpc 不会拓展外部实体, 并且会忽略
  * Python 3.7.1 开始, 默认情况下都不会处理外部通用实体
  


## 2.2. xml.etree.ElementTree

The `xml.etree.ElementTree` module implements a simple and efficient API for parsing and creating XML data.

`Changed in version 3.3: This module will use a fast implementation whenever available.`

一般会把该模组导入为 `as ET`

用于以树的形式对 xml 文件进行读取处理  

### 2.2.1. Tutorial 教程



#### 2.2.1.1. 基本读取方法
```py
import xml.etree.ElementTree as ET

# 拿到 XML 树
tree = ET.parse('country_data.xml')

# 拿到根节点
root = tree.getroot()

# 直接 从字符串解析
root = ET.fromstring(country_data_as_string)

# root 本身是一个节点 Element, 拥有标签和属性, 以及子节点
root.tag
root.attrib
for child in root:
    print(child.tag, child.attrib)

```

#### 2.2.1.2. 非阻塞方式解析 XML

在解析前不需要完整的读取整个文档, 而是以增量的方式读取, 使用 XMLParser 类

#### 2.2.1.3. 查找感兴趣的元素

Element 的拥有高级迭代方法, 可以直接遍历其下的所有子树, 包括 子级的子级等
* Element.iter('name')  查找并迭代所有子树
* Element.findall('name') 查找并迭代直接子树
* Element.find('name') 查找并访问第一个直接子树

```py
for country in root.findall('country'):
    rank = country.find('rank').text
    name = country.get('name')
    print(name, rank)

```


#### 2.2.1.4. 修改XML文件

* `ElementTree.write()` 将 XML 文件写入硬盘
* Element.text 直接修改节点的文本
* Element.set() 添加和修改属性
* Element.append() 添加新的子元素
* Element.remove() 删除子元素

```py
for rank in root.iter('rank'):

    # 修改文本
    new_rank = int(rank.text) + 1
    rank.text = str(new_rank)

    # 更新属性
    rank.set('updated', 'yes')

tree.write('output.xml')

```

### 2.2.2. XPath支持

提供了  XPath 的有限支持用于 元素在树中的定位


```py
import xml.etree.ElementTree as ET

root = ET.fromstring(countrydata)

# 最高层级的元素
root.findall(".")

# 最高层级下的 'country' 子元素的所有 'neighbor' 孙子元素
root.findall("./country/neighbor")

# 有一个 'year' 子元素的包含 name='Singapore' 的节点
root.findall(".//year/..[@name='Singapore']")

# 是包含 name='Singapore' 的节点的子元素的 'year' 节点
root.findall(".//*[@name='Singapore']/year")

# 是其父元素的第二个子元素的所有 'neighbor' 节点
root.findall(".//neighbor[2]")
```

模块所支持的 XPath 语法包括
* `tag` 选择具有给定标记的所有子元素
* `*` 选择所有子元素,  `*/egg` 指代所有 tag 为 `egg` 的孙元素
* `.` 当前节点, 在路径开头用于指示相对路径
* `..` 父元素, 如果试图访问起始元素的上级, 则返回 None 
* `//` 选择所有的下级元素
* `[@]` attrib 匹配
  * `@attrib` : 匹配具有给定属性的元素
  * `@attrib='value'` : 匹配具有给定属性以及对应值的元素
  * `@attrib!='value'` : python 3.10 匹配具有给定属性以及, 值不是对应参数值的元素
  * 注意在 python 中使用的时候要注意和路径字符串本身的标识符区分


### 2.2.3. XInclude 支持

Xinclude 指令支持

通过模块 `xml.etree.ElementInclude` 辅助模块提供了对 XInclude 指令的优先支持, 用于根据元素树的信息在树中插入子树或者文本





### 2.2.4. API 函数 Functions

不知道为啥这个包里的函数在文档里分了好几个地方写了  

基本函数
* `xml.etree.ElementTree.parse(source, parser=None)`
  * 将 XML 解析为 element tree
  * `source` 是 filename or file object containing XML data
  * `parser` 用于手动指定 parser, 如果不指定的话会使用默认的 `XMLParser`
  * 返回树实例 `ElementTree`



* `xml.etree.ElementInclude.default_loader(href, parse, encoding=None)`
* `xml.etree.ElementInclude.include(elem, loader=None, base_url=None, max_depth=6)`

### 2.2.5. ElementTree

一般对于整个文档进行 改查 是在 Tree 层面进行的

`class xml.etree.ElementTree.ElementTree(element=None, file=None)`  



### 2.2.6. Element
<!-- 完 -->

对于单个元素的操作是在 Element 进行的
` class xml.etree.ElementTree.Element(tag, attrib={}, **extra)`

一个节点拥有的成分
* tag : str, 标识这个数据是什么数据, 数据类型
* text, tail  : 描述起来比较复杂, 存放元素相关联的额外数据
  * text 会存放元素的 `开始标记到 第一个子元素或结束标记` 之间的文本
  * tail 会存放 元素的 `结束标记到 下一个标记` 之间的的文本
  * 或者为 None
* attrib: 包含元素属性的字典
  * attrib 的值总是一个可变的字典 `dict`
  * ElementTree 的实现可以让 attrib 以内部的实现方式来访问, 可能有性能上的优势?
  * 如果需要以 ElementTree 的内部方式访问 attrib, 则需要直接调用元素对象的 字典方法
* 字典方法
  * clear()   : 移除所有子元素, 清空属性, text 和 tail 设置为 None
  * get(key, default=None)
  * items()   : 返回键值对, 顺序任意
  * keys()
  * set(key, value) : 设置 key 属性的值为 value
  * append(element)  :  添加子元素
    * 会进行 Element 类型检查
  * extend(elements) : python3.2, 从一个包含多个元素的可迭代对象添加多个元素
    * 会进行 Element 类型检查
  * insert(index, element) : 将子元素插入到给位置中
    * 会进行 Element 类型检查
* 元素查找方法
  * find(match, namespaces=None)   : 查找第一个匹配的子元素
    * 返回 Element 或者 None
  * findall( match, namespace=None) : 查找所有匹配的子元素
    * 返回 Element 的列表
  * findtext(match, default=None, namespace) : 查找第一个匹配的子元素的文本
    * 返回第一个匹配的元素的文本内容
    * 如果匹配到的元素没有文本内容, 则返回 `""` 空字符串
    * 如果没有匹配到则返回参数 default 的值
  * iter(tag=None) python 3.2
    * 以当前节点为根, 遍历的访问所有下级元素 (深度优先)
    * tag 为 `None` 或者 `*` 的时候会访问所有, 否则会只返回标记为 tag 的元素
    * 该方法不保证 XML 树在访问期间被修改时候的行为
  * iterfind(match, namespaces=None) python 3.2
    * 查找所有匹配的子元素
    * 返回迭代器
  * itertext()  python3.2
    * 没有参数
    * 文本迭代器, 将按照文档顺序遍历此元素以及所有子元素, 返回所有内部文本
  * remove(subelement)
    * 从该节点中查找匹配参数 元素tag 的元素并删除

注意: 创建子元素的时候使用 `SubElement()` 工厂函数
* 存在 `makeelement(tag, attrib)` 接口但是不被建议使用


### 2.2.7. XMLParser

`class xml.etree.ElementTree.XMLParser(*, target=None, encoding=None)`