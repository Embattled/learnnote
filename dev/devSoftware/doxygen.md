# doxygen 文档生成

* doxygen 是解析源代码, 自动生成文档的工具
* doxywizard 是用于配置 doxygen 的 GUI 工具

doxygen 默认支持的语言有:
* C, C++, C#, Objective-C
* Python
* IDL
* Java
* VHDL
* PHP
* Fortran
* D

`doxygen --help` 查看简易帮助文档

## 配置文件

对于一个 project 首先需要指定 doxygen 的配置, 配置文件类似于 makefile, 对于每一个项目指定一个 doxygen 配置文件

* `doxygen -g <config-file>` 基于模板生成配置文件
   * `INPUT` : 指定源文件目录, 置空时在当前目录查找
   * 设置的时候可以使用通配符
   * 如果需要递归遍历, 需要 `RECRSIVE=yes` 


## 文档生成

* `doxygen <config-file>` 生成文档



# doxygen 规范注释

以C语言为代表, 本章说明 doxygen 文档注释的写法

以下所有内容都省略了语言自己的注释符号

* 对于代码中的每个实体, 有多种类型的描述, 共同构成该实体的文档:
  * 简短描述    : 在内容前面加 `\brief `
  * 详细描述    : 什么都不加, 默认的
  * in-body 描述 : 对于方法和函数 的特殊的描述类型
* 对于每个注释块, 最多只有一个详细描述和一个简短描述, 因为如果有多个, 则多个之间不能确定顺序
* `\` 反斜杠在注释块里作为特殊命令的前缀符, 例如 `\brief `



## 实体前标准注释
实体前doxygen注释的几种写法:
* `///` 三个反斜杠, 比普通注释多一个, 一个注释块的内部要连续不能空行
* `/** */` or `/*! */` 多行注释, 在开头部分多加一个星号或叹号 (内部每行也带星号便于美观识别)
* 对于 inbody 描述, 三种描述方法
  * `///`
  * `//!`
  * `/*! */`

```cpp
///
/// \brief brief description is start with %\brief
/// 
/// This is a detail description
///
void test0() {}

///////////////////////////////
/// \brief brief description is start with %\brief
/// 
/// This is a detail description
///////////////////////////////
void test1() {}

/**
 * \brief brief description
 *
 * This is a detail description
 */
void test2() {}
​
/*!
 * \brief brief description
 *
 * This is a detail description
 */
void test3() {}
​
/*!
\brief brief description
​
 This is a detail description
*/
void test4() {}
​
​
class Test 
{
public:
    int value0; //!< member description
    int value1; /*!< member description */
    int value2; ///< member description
    int vluae3; // This is not be parsed
};
```

## 结构化注释

在需要特殊情况的时候, 可以将注释写在任何地方, 称为结构化命令
* 结构化命令以 `\` 或者 `@`
* `\class name_of_class` 代表该注释段是针对 这个类添加的
* 同理的针对特殊类的 结构化命令有 (省略命令首部的反斜杠):
  * struct
  * union
  * enum
  * fn     : 对函数进行文档化注释
  * var
  * def
  * typedef
  * file    : 对整个文件的注释
  * namespace
  * package : 主要出现在 java 里
  * interface: 针对 IDL 接口
* 如果要文档一些全局的对象, 需要先给文件添加文档注释

## 美化注释块内容

注释块中, doxygen 支持 markdown 格式, 甚至还支持一些 markdown extra

doxygen 也有自己的支持的语法, 在 markdown 的列表的基础上, doxygen 的写法:  
* 同 markdown 一样  `-` `1.` 的数字
* doxygen 独有的 `-#` 会在编译后自动生成数字
```cpp
 /*! 
  *  A list of events:
  *    - mouse events
  *         -# mouse move event
  *         -# mouse click event\n
  *            More info about the click event.
  *         -# mouse double click event
  *
  *      End of first list
  *
  *    - keyboard events
  *         1. key down event
  *         2. key up event
  *
  *  More text here.
  */
```

## 分组 modules

用于内容分级聚合:
* 被分到一个 group 的内容会生成在同一个页面里
* group 的成员可以是任何单位: 文件, 命名空间, 类, 函数, 变量

`\defgroup <name_of_group> [alias_of_name]` 用于定义一个组
* 第二个参数决定的是 组在文档中显示的命令, 可以是方便理解的一个小标题
* `\addtogroup` 可以冗余的创建一个组, 在组名已经存在的时候不会报错
* 非连续的将内容加入组
  * 通过`\ingroup` 将函数类等添加到已有的group中
  * 通过 `@{` 和 `@}` 来将多行内容添加到组中

```cpp
// 创建组 A, 使用 @ 来添加组的内容
/// \addtogroup A
/// @{
int bool InA;
/// @}

// 使用结构化命令来将 VarInA 变量添加到组中
/**
 * \ingroup A
 */
extern int VarInA;

// 定义一个组, 并使用 @{ 开始定义内容
/**
 * \defgroup IntVariables Global integer variables
 * @{
 */

// 组中内容
/** an integer variable */
extern int IntegerVariable;
​
// 组结束
/**@}*/



​

```