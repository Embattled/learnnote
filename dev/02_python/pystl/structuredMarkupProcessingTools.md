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
因此 `xml.parsers.expat` 因为总是可用, 所以也应该是最常用的.  

The XML handling submodules are: 包括了各种不同的XML读取处理方法, 但是本质上的目的应该都一样  
* `xml.parsers.expat`: the Expat parser binding
* `xml.etree.ElementTree`: the ElementTree API, a simple and lightweight XML processor
* `xml.dom`: the DOM API definition
* `xml.dom.minidom`: a minimal DOM implementation
* `xml.dom.pulldom`: support for building partial DOM trees
* `xml.sax`: SAX2 base classes and convenience functions

## 2.1. xml.etree.ElementTree

The `xml.etree.ElementTree` module implements a simple and efficient API for parsing and creating XML data.

`Changed in version 3.3: This module will use a fast implementation whenever available.`

一般会把该模组导入为 `as ET`

用于以树的形式对 xml 文件进行读取处理  

### 函数 Functions

不知道为啥这个包里的函数在文档里分了好几个地方写了  

#### 基本函数

* `xml.etree.ElementTree.parse(source, parser=None)`
  * 将 XML 解析为 element tree
  * `source` 是 filename or file object containing XML data
  * `parser` 用于手动指定 parser, 如果不指定的话会使用默认的 `XMLParser`
  * 返回树实例 `ElementTree`



* `xml.etree.ElementInclude.default_loader(href, parse, encoding=None)`
* `xml.etree.ElementInclude.include(elem, loader=None, base_url=None, max_depth=6)`

### ElementTree

一般对于整个文档进行 改查 是在 Tree 层面进行的
`class xml.etree.ElementTree.ElementTree(element=None, file=None)`  

### Element

对于单个元素的操作是在 Element 进行的
` class xml.etree.ElementTree.Element(tag, attrib={}, **extra)`

### XMLParser

`class xml.etree.ElementTree.XMLParser(*, target=None, encoding=None)`