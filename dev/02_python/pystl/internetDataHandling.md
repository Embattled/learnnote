- [1. Internet Data Handling and Structured Markup Processing Tools](#1-internet-data-handling-and-structured-markup-processing-tools)
- [2. json JSON encoder and decoder](#2-json-json-encoder-and-decoder)
  - [2.1. Basic Usage](#21-basic-usage)
    - [2.1.1. load](#211-load)
    - [2.1.2. dump](#212-dump)
  - [2.2. Endocer Decoder](#22-endocer-decoder)
  - [2.3. Exception](#23-exception)
  - [2.4. python 命令行直接使用](#24-python-命令行直接使用)
- [base64 — Base16, Base32, Base64, Base85 Data Encodings](#base64--base16-base32-base64-base85-data-encodings)

# 1. Internet Data Handling and Structured Markup Processing Tools

用于处理在互联网上常用的数据类型
This chapter describes modules which support handling data formats commonly used on the internet.

Internet Data 分类为:  
用于处理一些网络上的通用数据格式的模组定义在该分类下, 具体包括
1. email
2. json
3. mailcap
4. mailbox
5. mimetypes
6. base64
7. binhex
8. binascii
9. quopri

除此之外, 还有一个小分组, 用于处理网页的基石, html 和 xml  
Structured Markup Processing Tools:  
* xml
* html


# 2. json JSON encoder and decoder

JSON 原名 JavaScript Object Notation, 从 JS 的语法衍生而来, Python 提供了 JSON处理的 STL 包  

要注意 JSON 有可能会包含极大数据, 占用过多的 CPU 资源, 因此在使用时要注意数据来源, 设置好数据大小上限

`JSON` 目前是 YAML 1.2 的一个子集, 因此该包也可以直接用来处理 YAML 数据 (as a YAML serializer).

该包定义的特殊的 JSON 类是有序的, 算是一个改进后的 dict (python 内建 dict 是key无序的), 因此如果不手动转换成 python 的 dict, 则能够保证数据的顺序不变


快速使用实例
```python
import json

# 把序列和字典的符合数据打包
json.dumps(['foo', {'bar': ('baz', None, 1.0, 2)}])
# '["foo", {"bar": ["baz", null, 1.0, 2]}]'

print(json.dumps("\"foo\bar"))
# "\"foo\bar"

print(json.dumps('\u1234'))
# "\u1234"

print(json.dumps('\\'))
# "\\"

print(json.dumps({"c": 0, "b": 0, "a": 0}, sort_keys=True))
# {"a": 0, "b": 0, "c": 0}

from io import StringIO
io = StringIO()
json.dump(['streaming API'], io)
io.getvalue()
'["streaming API"]'

# dump 出来的 json 通过 Print 漂亮的打印, 
print(json.dumps({'4': 5, '6': 7}, sort_keys=True, indent=4))
#{
    # "4": 5,
    # "6": 7
# }
```

## 2.1. Basic Usage

### 2.1.1. load

`json.load(fp, *, cls=None, object_hook=None, parse_float=None, parse_int=None, parse_constant=None, object_pairs_hook=None, **kw)`
* 从一个 file like 中读取 JSON
* `object_hook` : 自定义解码器, 传入的话返回的就不是 dict 了, 需要保证与 dict 的兼容性
* `object_pairs_hook` : 更加优先的自定义解码器, 跟顺序有关, 不太懂
* `parse_float` `parse_int` `parse_constant`

`json.loads(s, *, cls=None, object_hook=None, parse_float=None, parse_int=None, parse_constant=None, object_pairs_hook=None, **kw)`
* 从字符串中读取 JSON


### 2.1.2. dump

注意 
* 在 dump 的时候, 字典的所有 key/value 都会被转化成 str, 因此如果 kv 中有non-str 的话 `loads(dumps(x)) != x`
* JSON 不是 `framed protocol` (不太懂), 所以不要尝试多次调用 dump 来把多个 obj 写入同一个 file, 会导致整个文件失效

`json.dump(obj, fp, *, skipkeys=False, ensure_ascii=True, check_circular=True, allow_nan=True, cls=None, indent=None, separators=None, default=None, sort_keys=False, **kw)`  
* 把一个 object 保存为 json
* `fp` : file-like object 支持写的类文件
* `skipkeys`: bool, 验证所有的 key 是否是基础类型 `str, int, float, bool, None`, 如果是特殊类型则会报错
* `ensure_ascii` : 如果为真, 则会保证所有传入的非 ASCII 字符都通过转义(不太懂), 如果 False, 字符则会原样输出
* `check_circular` : 检查嵌套的 dict 防止无穷递归
* `allow_nan` : 允许超过JS float 范围的数据, 例如 python 中的 inf, -inf 会被转化成 JS 语言中的等效项目 Infinity -Infinity, False 的话会直接报错
* `indent` : 自定义缩进符号, 传入正整数或者只包含空格的字符串, 如果是 None 的话则会使用最紧凑的方法 (可能会不方便阅读)
* `sort_keys` : 是否重排列 字典的顺序 by key
* `separators` : 自定义 key 和 value 的分隔符, 输入需要保证为 `(item_separator, key_separator)` 的格式, 默认是 `(', ', ': ') `
* `cls` : 自定义的 JSONEncoder


`json.dumps(obj, *, skipkeys=False, ensure_ascii=True, check_circular=True, allow_nan=True, cls=None, indent=None, separators=None, default=None, sort_keys=False, **kw)`
* 把 obj 转化成 JSON format 的 字符串
  



## 2.2. Endocer Decoder



## 2.3. Exception


## 2.4. python 命令行直接使用


# base64 — Base16, Base32, Base64, Base85 Data Encodings

将 2进制数据编码为可打印的 ASCII 字符串

提供了 RFC 4648 中定义的 Base16, Base32, Base64 算法, 以及 de-facto 标准的 Ascii85 和 Base85 的编码算法

RFC 4648 的编码适用于将二进制数据编码成能够方便的通过邮件发送, 或者作为 URL 以及 http post 的字符格式  
编码算法与 uuencode 程序不同


模块提供两个接口
* 将 bytes-like 对象转为 ASCII bytes
* 将 bytes-like 对象 或者 ASCII字符串 转回 bytes
* base-64 alphabets 定义在 RFC 4648 文件中

模组中有 legacy 接口:
* 仅支持 Base64 标准字母
* 不支持 字符串接口, 但是支持从 文件中解码
* 根据 RFC 2045, 对每 76 个字符添加一个 newlines
* 应该是专用于电子邮件的解码格式, 然而专用于 RFC 2045 的解码支持定义在了 `email` 中, 因此传统接口理论上没有应用场景



接口
* `b64encode(s, altchars=None)`
  * 编码 bytes-like, `altchars` 用于替代 Base64 编码的 `+/` flag, 因为 `+/` 字符在文件名和 url 中可能不安全
  * `altchars` 必须长度为 2
* `standard_b64encode(s)`
  * 标准 Base64 编码
* `urlsafe_b64encode(s)`
  * 直接生成文件名和URL安全的 Base64 编码
  * 用 `-`替代 `+`, 用 `_` 替代 `/`, 感觉应该相当于 `b64encode(s, altchars='-_')`
* `b32encode(s)`
  * Base32 编码
* `b32hexencode(s)` : 3.10 新接口
  * 使用 RFC 4648 定义的拓展 16进制字符 Hel Alphabet
* `b16encode(s)`
  * Base16 编码


The legacy interface: 不推荐使用的
* decode
* decodebytes
* encode
* encodebytes