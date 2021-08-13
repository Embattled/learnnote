# 1. YAML

几乎是专门用来写配置文件的语言, 并且容易阅读
* 后缀名 `.yml`
* 于2001年发布, 现在已经支持多种语言

# 2. 运行库

* LibYAML 官方制作的 C 的运行库
* pyyaml  python的运行库

# 3. 基本知识
## 3.1. 语法

1. 大小写敏感
2. 使用缩进来表示层级关系, 但是缩进只能使用空格而不能用 tab
3. 缩进的空格数不做要求, 只需要同一层级对齐即可
4. `#` 井号表示注释

## 分块

* 一个 yml 文档中可以同时包括几个不同的配置文件, 使用 `---` 进行分割
* 一个文件块可以 optionally 的使用 `...` 来进行结尾

```yml
---
name: qiyu
age: 20岁
...
---
name: qingqing
age: 19岁
...
```

## 3.2. 数据类型

YAML支持的数据类型:
- 纯量 scalars : 单个值, 不可再分的值
- 数组 sequence: 一组值, 又称序列或者列表
- 对象         : 键值对的集合, 因此又称为 映射 mapping, 哈希 hashes, 字典 dictionary


### 3.2.1. 纯量

最基本的数据类型
* 基础的四种: 字符串, 布尔值, 整数, 浮点数
* 空值: Null
* 时间和日期
  * 日期的表示必须使用特殊规定, ISO 8601格式, yyyy-MM-dd
  * 时间也同理 `15:02:31`
  * 统一的时间日期为: 日期后接`T`连接时间, 再接`+`连接时区

```yml
boolean: 
    - TRUE  #true,True都可以
    - FALSE  #false，False都可以
float:
    - 3.14
    - 6.8523015e+5  #可以使用科学计数法
int:
    - 123
    - 0b1010_0111_0100_1010_1110    #二进制表示
null:
    nodeName: 'node'
    parent: ~  #使用~表示null
string:
    - 哈哈
    - 'Hello world'  #可以使用双引号或者单引号包裹特殊字符
    - newline
      newline2    #字符串可以拆成多行，每一行会被转化成一个空格
date:
    - 2018-02-17    #日期必须使用ISO 8601格式，即yyyy-MM-dd
datetime: 
    -  2018-02-17T15:02:31+08:00    #时间使用ISO 8601格式，时间和日期之间使用T连接，最后使用+代表时区
```
### 3.2.2. 数组

* 以 `-` 开头的行代表一个数组
* 用 `[]` 方括号括起来的数据也是一个数组

```yml
- A
- B
-
  - suba
  - subb

key: [v1,v2,v3]
```

### 3.2.3. 对象类型

* 用冒号来分割键值 `key: value` , 注意冒号后面有一个空格
* 可以递归的定义, 用缩进或者大括号来表示层级关系
* 如果一个key非常复杂, 不能用一行表示, 例如 key 是一个数组
  * 使用 `? ` 一个问号跟着一个空格代表一个key
  * 结束后使用 `: ` 同样一个冒号跟着一个空格代表value
* 使用对象和数组可以组成复杂的复合结构

```yml
key: {key1: value1, key2: value2}

key:
     child-key: value
     child-key2: value

# 复杂对象
? 
  - long key
  - long key2
:
  - value  

# 复合结构
languages:
  - Ruby
  - Perl
  - Python 
websites:
  YAML: yaml.org 
  Ruby: ruby-lang.org 
  Python: python.org 
  Perl: use.perl.org
```

### 引用

* 在 yaml 中也有引用规则
* `&` 锚点用来建立一个引用
* `*` 调取一个引用
* `<<` 用于合并引用中的数据, 相当于会拆包一层


```yml
# 使用引用
- &showell Steve 
- Clark 
- *showell

# 转换成的实际代码是
[ 'Steve', 'Clark', 'Steve' ]

# 建立引用
defaults: &defaults
  adapter:  postgres
# 使用引用合并数据
development:
  database: myapp_development
  <<: *defaults
# 最终的实际数据
development:
  database: myapp_development
  adapter:  postgres
```
