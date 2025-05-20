# 1. Development Tools

专门用于辅助开发的模组种类 

# 2. typing

https://docs.python.org/3/library/typing.html#module-typing


* python 是弱类型语言, 语法上不强制任何类型匹配
* 可以通过注释来添加类型标识, 解释器层面不会进行检查, 但是可以用于 IDE 来帮助检查
* 简单来说就是符合语法的各种注释写法

不经过 typing 也可以为函数添加简单的类型注释, 例如
```py
def surface_area_of_cube(edge_length: float) -> str:
    return f"The surface area of the cube is {6 * edge_length ** 2}."
```
通过 typing, 可以实现更加高级的注释方法

typing 包是一个高频率更新的包, 并且新版本的包会将功能添加到旧版本的  `typing_extensions`, 因此理论上可以避免因为 typing 导致的代码与旧版本不兼容

The most fundamental support consists of the types `Any, Union, Tuple, Callable, TypeVar, and Generic`

## 2.1. Specification for the Python Type System

python 语言的 type 系统规范可以在别的页面找到

[“Specification for the Python type system”.](https://typing.readthedocs.io/en/latest/spec/index.html)

## 2.2. Type aliases - 类型别名

可以查看 `type statement`, 虽然是 typing 模组里的功能但是使用不需要导入包 (python3.12 版本以后)
```py
type Vector = list[float]

def scale(scalar: float, vector: Vector) -> Vector:
    return [scalar * num for num in vector]

# passes type checking; a list of floats qualifies as a Vector.
new_vector = scale(2.0, [1.0, -4.2, 5.4])
```

type alias 对于将复杂类型简单化非常有用

```py
from collections.abc import Sequence

type ConnectionOptions = dict[str, str]
type Address = tuple[str, int]
type Server = tuple[Address, ConnectionOptions]

def broadcast_message(message: str, servers: Sequence[Server]) -> None:
    ...

# The static type checker will treat the previous type signature as
# being exactly equivalent to this one.
def broadcast_message(
    message: str,
    servers: Sequence[tuple[tuple[str, int], dict[str, str]]]
) -> None:
    ...
```

type 语法是 3.12 新功能, 但旧版本的 typing 包里面已经有 type alias, 对于旧版本 python 的支持
* 可以直接定义 `Vector = list[float]`, created through simple assignment
* 也可以导入 Typing 包然后声明好 这是一个 类型别名 `Vector: TypeAlias = list[float]`



## 2.3. Module contents

模组的正式 API 文档
typing 模组定义的一系列用于类型提示的 类, 函数, 描述符

### 2.3.1. Special typing primitives

用于 annotation 的各种特殊类型 

#### 2.3.1.1. Special types 特殊类型

These can be used as types in annotations. They do not support subscription using [].
不支持放在方括号中 `[]`

* `typing.Any`
  * Any 作为 annotation 的时候可以匹配任何类型
* `typing.NoReturn`
  * NoReturn 用作函数返回值的 annotation, 表示这个函数永远不会返回值

```py
from typing import NoReturn, Any

def stop(
    para: Any
) -> NoReturn:
    raise RuntimeError('no way')
```

#### 2.3.1.2. Special forms

* 这几个类型可以用于描述一个使用方括号 `[]`  的类型, 每一个类型**都有特殊的语法**
* 最早用于方便 annotation 的类型有一些已经被 新版本的抽象类实现了

整理
* Union     : 用于定义 `或`
* Optional  : 相当于 `Union[X|None]`
* Literal   : 用于定义参数为  一个字符串列表中的其中之一 


用于 TypedDict 的特殊 forms
* 3.11 typing.Required       : 定义一个必须的 key
* 3.11 typing.NotRequired    : 定义一个非必须的 key
* 3.13 typing.ReadOnly       : 定义一个只读的 key



* `typing.Union`
  * 用于指定该类型是可选列表中的其中一种, 且不能为空
  * `Union[int, str]` 这类型同样只用于 annotation, 不能继承或实例化
  * 该类型描述不支持嵌套, 无顺序, 且可以自动删除重复
    * `Union[int] == int`
    * `Union[Union[int, str], float] == Union[int, str, float]`
    * `Union[int, str, int] == Union[int, str]`
    * `Union[int, str] == Union[str, int]`
  * 可以在方括号中加入 None, 但是对应的功能可以用 `Optional` 来实现
* `typing.Optional`
  * 很简单的描述 `Optional[X] is equivalent to Union[X, None]`
  * 有意思的是这个和带有默认值的参数不同, 因为 Optional 本来就是带有 None 的Union 的简写
  * `def foo(arg: int = 0)` 直接带有非None 默认值的参数不能用 Optional
  * `def foo(arg: Optional[int] = None)` 合理





以下的类型已经不推荐使用, 已经从原本章节中被移动到弃用章节
* `typing.Tuple (Deprecated) `
  * 用于指定一个元组类型, 元组中的每个元素类型都可以单独指定
  * 3.9 版本后不需要加 Tuple, 直接用 `[int,...]` 就可以
  * `Tuple[int, float, str]`  空元组可以写作 `Tuple[()]`
  * 可以用 `...` 来代表相同元素的变长元组
  * `Tuple[int, ...]`  一个通用的元组可以用 `Tuple[Any, ...]` 来匹配
* `typing.Callable (Deprecated)`  使用 `collections.abc.Callable` 作为替代
  * 专门用来描述函数的类型 `Callable[[int], str] is a function of (int) -> str`
  * `Callable[[Arg1Type, Arg2Type], ReturnType]`
  * 一个方括号, 里面有两个参数, 第一个是参数列表, 第二个是返回值类型

#### 2.3.1.3. Building generic types and type aliases

该模组的内容主要是用于定义 `泛型的数据类型` 以及 类型别名  
该模组的内容不应该直接用作 annotations

该模组的内容对早期 python 有兼容性, 因此有多种定义语法  
* type parameter lists
* type statement
* 模组中的定义方法为更早期 3.11 以前的版本提供了兼容性


`class typing.Generic`
* Abstract base class for generic types.  虚基类, 用于通用的类型定义  
* 通过定义一个类的方式来定义一个 数据类型 的类

```py
# 定义一个类的方式来定义一种泛型数据类型, 语法是使用方括号
# 这个语法会自动的继承 typing.Generic
class Mapping[KT, VT]:
    def __getitem__(self, key: KT) -> VT:
      ...

# 使用该类来为函数参数添加注解
# 这里 Mapping 后使用方括号 为 泛型数据类型的使用方法
def lookup_name[X, Y](mapping: Mapping[X, Y], key: X, default: Y) -> Y:
    try:
        return mapping[key]
    except KeyError:
        return default

# 方括号的语法可能不兼容旧版本 python
# 直接定义相关的 TypeVar 可以获得旧版本兼容性
KT = TypeVar('KT')
VT = TypeVar('VT')

class Mapping(Generic[KT, VT]):
    def __getitem__(self, key: KT) -> VT:
        ...
        # Etc.

```

#### 2.3.1.4. Other special directives - 其他特殊的类型指令

这里的对象都是用于 定义类型的 工厂类, 因此不能直接被用于 annotation



`class typing.TypedDict(dict)`
* 一个用于 type hint 的 创建字典的特殊类, 在运行的时候该类仍然是普通的 dict
  * 如果违反规定也只会被 typechecker 报错
* 会声明该类型是一个特殊的字典, 拥有指定的 key 和其对应的 值类型

```py
class Point2D(TypedDict):
    x: int
    y: int
    label: NotRequired[str]
```


## 2.4. 基础用法

类型别称:
* 用于辅助类型标识, 可以理解为 `typedef`
* 用法 `Vector = list[float]`
* 这样定义的 Vector 类型仍然只能用于 annotation 

函数的带注释写法为:

```py
def greeting(
    name: str
    ) -> str:
    return 'Hello ' + name

Vector = list[float]

def scale(
    scalar: float, 
    vector: Vector
    )-> Vector:
    return [scalar * num for num in vector]
```

## 2.5. NewType

* NewType 是 typing 包中的一个函数 `from typing import NewType`
* NewType 同样类似于 `typedef` 不过该类型定义是有程序含义的, 类似于 subclass
* 这样定义的子类在运算上和原本的类型没有区别, 返回值也总是父类, 这个功能主要用于类型检查


```py
from typing import NewType

# UserId 是 int 的子类, 但是不是相同的类型, an int is not a UserId
UserId = NewType('UserId', int)
some_id = UserId(524313)

# 'output' is of type 'int', not 'UserId'
output = UserId(23413) + UserId(54341)
```

## 2.6. Callable

Callable 是一个 注释类型, 用于说明这是一个函数

`Callable[[Arg1Type, Arg2Type], ReturnType]`

```py
from collections.abc import Callable

def feeder(
    get_next_item: Callable[[], str]
    ) -> None:

def async_query(
    on_success: Callable[[int], None],
    on_error: Callable[[int, Exception], None]
    ) -> None:
```

## 2.7. Generics

```py
from collections.abc import Mapping, Sequence

def notify_by_email(
    employees: Sequence[Employee],
    overrides: Mapping[str, str]
    ) -> None: ...

from collections.abc import Sequence
from typing import TypeVar

T = TypeVar('T')      # Declare type variable

def first(
    l: Sequence[T]
    ) -> T:   # Generic function
    return l[0]
```