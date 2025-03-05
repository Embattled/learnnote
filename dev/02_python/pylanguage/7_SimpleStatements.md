
# 13. Simple statements - 简单语句

## 13.1. Expression statements - 表达式语句

## 13.2. Assignment statements - 赋值语句

## 13.3. The assert statement - 断言语句
<!-- 完 -->

用于向程序插入 调试断言的 便捷的方法
类似于 C 语言的 assert

`assert_stmt ::=  "assert" expression ["," expression]`

```py

assert expression

# 等同于
if 表达式==True:
    程序继续执行
else:
    程序报 AssertionError 错误

# equivalent 
if __debug__:
    if not expression: raise AssertionError

assert expression1, expression2
if __debug__:
    if not expression1: raise AssertionError(expression2)
```

assert 语句的行为会参照 内置变量 `__debug__`
* `__debug__` 的值在解释器启动的时候确定, 为 `__debug__` 进行赋值是非法的
* 对解释器添加 `-O` 选项即可启动最优化, 即 在代码运行的时候不会针对 assert 语句编译任何内容, 此时 `__debug__` 为 False
* 错误信息不需要包括源代码, 会作为 stack trace 的一部分被输出

## 13.4. The pass statement
<!-- 完 -->
`pass_stmt ::=  "pass"` 

空运算符, 作为 占位符.  
当某个语法需要 statement 的时候使用


## 13.5. The type statement - type 语句
<!-- 完 -->
python 3.12 新语法

`type_stmt ::=  'type' identifier [type_params] "=" expression`  

当前版本 两类 soft keyword 的其中之一 `type` 所对应的语句

type 被作为 soft keyword 从 typing 模组加入到了语言 built-in, type statement 会创建一个 `type alias`, 其为 `typing.TypeAliasType` 的实例
可以查看 typing 模组的说明

type 将某个符合类型声明为一个新的 类型别名 对象
```py
type Point = tuple[float, float]

# This code is roughly equivalent to:
annotation-def VALUE_OF_Point():
    return tuple[float, float]
Point = typing.TypeAliasType("Point", VALUE_OF_Point())
```

type alias 对象使用了 延迟求值(lazy evaluation) 方案, 即创建的时候不会对齐求值, 直接访问类型的 `__value__` 属性的时候才求值
这允许了 type alias 的定义可以引用尚未定义的名称
