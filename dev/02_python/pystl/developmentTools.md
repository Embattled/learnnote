# 1. Development Tools

专门用于辅助开发的模组种类 

# 2. typing

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

## Specification for the Python Type System

python 语言的 type 系统规范可以在别的页面找到

[“Specification for the Python type system”.](https://typing.readthedocs.io/en/latest/spec/index.html)

## 2.1. Special typing primitives

用于 annotation 的各种特殊类型 

## Type aliases - 类型别名

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

### 2.1.1. Special types 特殊类型

两个非常常用的类型, 不支持放在方括号中 `[]`

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

### 2.1.2. Special forms

* 这几个类型可以用于描述一个使用方括号的类型, 每一个类型都有特殊的语法
* 最早用于方便 annotation 的类型有一些已经被 新版本的抽象类实现了

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
* `typing.Callable (Deprecated)`
  * 专门用来描述函数的类型 `Callable[[int], str] is a function of (int) -> str`
  * `Callable[[Arg1Type, Arg2Type], ReturnType]`
  * 一个方括号, 里面有两个参数, 第一个是参数列表, 第二个是返回值类型


## 2.2. 基础用法

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

## 2.3. NewType

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

## 2.4. Callable

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

## 2.5. Generics

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