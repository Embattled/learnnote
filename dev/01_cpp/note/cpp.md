- [1. C++ 特性](#1-c-特性)
  - [1.1. 程序参数配置](#11-程序参数配置)
    - [1.1.1. C的输出与C++流的同步](#111-c的输出与c流的同步)
    - [1.1.2. cin 和 cout 的绑定](#112-cin-和-cout-的绑定)
- [2. Declarations 声明](#2-declarations-声明)
  - [2.1. Specifiers](#21-specifiers)
    - [2.1.1. storage class specifiers 存储级别](#211-storage-class-specifiers-存储级别)
    - [2.1.2. Type specifiers](#212-type-specifiers)
      - [2.1.2.1. decltype (C++11)](#2121-decltype-c11)
      - [2.1.2.2. cv (const and volatile) type qualifiers](#2122-cv-const-and-volatile-type-qualifiers)
    - [2.1.3. Attribute specifier sequence (since C++11)](#213-attribute-specifier-sequence-since-c11)
      - [2.1.3.1. C++标准语法](#2131-c标准语法)
      - [2.1.3.2. c++ standard attribute](#2132-c-standard-attribute)
      - [2.1.3.3. attribute in GCC MSVC CLANG](#2133-attribute-in-gcc-msvc-clang)
- [3. Expressions 表达式](#3-expressions-表达式)
  - [3.1. Value Categories　值类别](#31-value-categories值类别)
    - [3.1.1. lvalue 左值](#311-lvalue-左值)
    - [3.1.2. prvalue 纯右值](#312-prvalue-纯右值)
    - [3.1.3. xvalue](#313-xvalue)
  - [3.2. conversions (cast)](#32-conversions-cast)
- [4. Initialization 初始化](#4-initialization-初始化)
  - [4.1. Reference initialization](#41-reference-initialization)
- [5. Classes 类](#5-classes-类)
  - [5.1. 初始化列表](#51-初始化列表)
  - [5.2. 类和struct的区别 C++中增强的struct](#52-类和struct的区别-c中增强的struct)
  - [5.3. 构造函数 转换构造函数](#53-构造函数-转换构造函数)
  - [5.4. 函数对象](#54-函数对象)
    - [5.4.1. 使用定义](#541-使用定义)
  - [5.5. STL 函数对象模板类](#55-stl-函数对象模板类)
  - [5.6. Operator Overloading 运算符重载](#56-operator-overloading-运算符重载)
    - [5.6.1. 为什么要以全局函数重载运算符 加号 +](#561-为什么要以全局函数重载运算符-加号-)
  - [5.7. const](#57-const)
    - [5.7.1. C语言的常量](#571-c语言的常量)
    - [5.7.2. C++的 const](#572-c的-const)
    - [5.7.3. 类的 const 函数](#573-类的-const-函数)
    - [5.7.4. const 对象](#574-const-对象)
  - [5.8. Inheritance 类的继承](#58-inheritance-类的继承)
- [6. Templates 模板](#6-templates-模板)
  - [6.1. full (explicit) specializations](#61-full-explicit-specializations)
  - [6.2. partial specializations](#62-partial-specializations)
- [7. Exceptions C++ 异常](#7-exceptions-c-异常)
  - [7.1. throw expression](#71-throw-expression)
  - [7.2. exceptionType exception类](#72-exceptiontype-exception类)

# 1. C++ 特性

该笔记用于记录 C++ 特有的, 与 C 不同的语法, 规定等.  

若与C相同但是版本不同的特性则单独标出

C++11
* C++11 标准对 C++ 语言增添了约 140 个新特性
* 是C++第二个主要版本 (上一个是 C++98, 后一个是C++17)
* 从C++03到C++11花了8年, 而从那之后C++开始维持3年一更新的频率

该文档着重记录 C++ reference 中 Language 的部分 

https://en.cppreference.com/w/cpp/language


## 1.1. 程序参数配置

### 1.1.1. C的输出与C++流的同步

```cpp
ios::sync_with_stdio(false);
```

* 在C++中的输入和输出有两种方式，一种是scanf和printf，另一种是cin和cout
* 在`#include<bits/stdc++.h>`这个万能头文件下，这两种方式是可以互换的


* iostream默认是与stdio关联在一起的，以使两者同步，保证在代码中同时出现`std :: cin`和scanf或`std :: cout`和`printf`时输出不发生混乱,因此消耗了`iostream`不少性能。
  * cin和cout的输入和输出效率比第一种低  
  * 因为cin，cout 先把要输出的东西存入缓冲区 再输出 
* 系统默认 `standard stream` 应该都是同步的
  * 设置sync_with_stdio(false)，是让C风格的stream和C++风格的stream变成async且分用不同buffer。
  * 解除同步之后，注意不要与scanf和printf混用以免出现问题

### 1.1.2. cin 和 cout 的绑定

`std :: cin`默认是与`std :: cout`绑定的，所以每次操作的时候都要调用`fflush`  
这样增加了IO的负担，通过`tie(nullptr)`来解除`cin`和`cout`之间的绑定，进一步加快执行效率。

```cpp
// std::tie是将两个stream绑定的函数，空参数的话返回当前的输出流指针。

std::cin.tie(nullptr);
std::cout.tie(nullptr);
```


# 2. Declarations 声明

* Declarations 声明: 将 名称 引入 C++程序   
  * Declarations introduce (or re-introduce) names into the C++ program.
  * Each kind of entity is declared differently. 
* Definitions  定义: 让一个 由名称标识的实体 足以使用 的声明
  * Definitions are declarations that are sufficient to use the entity identified by the name. 




## 2.1. Specifiers

声明说明符

Declaration specifiers (decl-specifier-seq) is a sequence (`decl-specifier-seq`) of the following whitespace-separated specifiers, in any order: 
* typedef
* inline
* friend
* constexpr
* consteval
* constinit
* 存储级别说明符 register, static, thread_local, extern, mutable
* 类型说明符 Type specifiers
* Attributes 

### 2.1.1. storage class specifiers 存储级别

相关内容在 C 语言中也有


### 2.1.2. Type specifiers

A sequence of specifiers that names a type: `type-specifier-seq`
* The type of every entity introduced by the declaration is this type
* This sequence of specifiers is also used by type-id.

以下类型可以作为 type specifier 出现, 但是只能出现一个
* class 类
* enum 枚举
* 简单 type specifiers
  * char, int, long, void, double 等基本类型
  * auto (C++11)
  * decltype specifier (C++11)
* class, union, struct 定义的自定义类型名称
* cv qualifier
* typename specifier

多关键字 type specifiers 的特殊情况:
* `const` `volatile` can be combined with any type specifier except itself. 
* signed or unsigned can be combined with char, long, short, or int. 
* short or long can be combined with int. 
* long can be combined with `double`. 
* long can be combined with long. (since C++11)


#### 2.1.2.1. decltype (C++11)

* decltype ( entity )
* decltype ( expression )

获取一个表达式的类型, 依据表达式的内容有多种结果 `template <T a>`
* 传入值是 右值, 返回类型 T
* 传入值是 左值, 返回类型 T&
* 

使用方法:
```cpp
int i=1;
// 通过类型推导来定义新的变量
decltype(i) j = i *2;


auto f = [](int a, int b) -> int
{
    return a * b;
};
// 对 lambda 函数使用 decltype 
decltype(f) g = f;


// 定义一个结构体
struct A { double x; };
// 定义一个指针
const A* a;
 
//  
decltype(a->x) y;       // type of y is double (declared type)
// 
decltype((a->x)) z = y; // type of z is const double& (lvalue expression)
```

#### 2.1.2.2. cv (const and volatile) type qualifiers


### 2.1.3. Attribute specifier sequence (since C++11)

Introduces implementation-defined attributes for types, objects, code, etc.   
定义一些修饰实体的实现方式的相关属性 (字节对齐等)  

原本是各个C++编译器的自己的特色功能, 在 GCC 和 MSVC 中都有自己的语法.  
在C++11被引入了标准, 并定义了标准语法.  
在 C23 中也作为标准被引入了C语言.  

#### 2.1.3.1. C++标准语法

```cpp
[[attr]] [[attr1, attr2, attr3(args)]] [[namespace::attr(args)]] alignas_specifier 

// Formally, the syntax is
[[ attribute-list ]] 		                          // (since C++11)
[[ using attribute-namespace : attribute-list ]]  //(since C++17)

// attribute 的书写方式, 对应于不同种类的 attribute
// (1) identifier    简单 attribute
[[ noreturn ]] void f() 
[[deprecated]]
void TriassicPeriod(
// (2) identifier ( argument-list )   带参数的简单 attribute
[[deprecated("Use NeogenePeriod() instead.")]] 
void JurassicPeriod() {}

attribute-namespace :: identifier   //(3)
attribute-namespace :: identifier ( argument-list ) 	//(4)
```

#### 2.1.3.2. c++ standard attribute

一下仅仅是被加入 C++ 标准的 attribute, 可以看到各个 attribute 的加入时间都不同  
被加入标准里的仅仅是各个编译器支持的很小一部分, 需要具体查看各个编译器的手册  

1. `[[noreturn]]`  (C++11)
   * Indicates that the function does not return. 
   * applies to the name of the function being declared in function declarations only.

2. `[[deprecated( string-literal )]]`  (C++14)
   * 用于标注非建议的实体
   * 提示信息是可选的
   * Compilers typically issue warnings on such uses.

3. `[[fallthrough]]` (C++17)

#### 2.1.3.3. attribute in GCC MSVC CLANG

语法:
* GNU C : `__attribute__ ((attribute-list))`  和标准语法的方括号不同, 这里是二重圆括号
  * 放于声明的尾部分号 `;` 之前 
  * 运行 gcc 时需要加上 `-Wall` 来激活 attribute 功能
* MSVC  : `__declspec()` 

同理, 各个编译器也有名称相同的通用 attribute, 也有特有的


# 3. Expressions 表达式

An expression is a sequence of operators and their operands, that specifies a computation.   


## 3.1. Value Categories　值类别 

每个C++`表达式`的种类都可以按照两种独立的特性进行区分, C语言也是同理, 但C++的类别更加细分化  
1. 类型, 每个值都有的 eg. int, char
2. 值类别

值类别可以进行多段细分, 且与C++的版本更替进行着变更:  
目前按照值的属性分出的类别有两个
* glvalue : (generalized lvalue) 是一个表达式, 表达式的值代表一个对象或函数的身份
  * glvalue = lvalue + xvalue
  * glvalue 可以被隐式转换成 prvalue, 通过 
    * lvalue-to-rvalue
    * array-to-pointer
    * function-to-pointer
  * glvalue 可以是多态的
    * glvalue 表达式代表的 static type 和其标识的 object 的 dynamic type 不一定一致
  * glvalue 可以是不完整的, 如果表达式允许的话
* rvalue : 右值, 因为只能出现在赋值运算符的右边
  * rvalue = prvalue + xvalue , 所以右值其实是一个综合的概念
  * 不能通过 `&` 取址运算符拿到右值在内存中的地址, 例 `&42, &int()`
  * 右值不能通过 `=` 赋值运算符更改值, 例 `3=1`
  * 右值可以用于初始化左值(基础用法), 也可以初始化 `const 左值引用`
    * 此时右值标识的对象的生命周期将会被延长到引用的作用域结束
  * 右值也可以用于初始化 `右值引用`
    * 同理, 该右值的对象生命周期交由引用的作用域
  * 当右值用于函数参数, 函数有两个重载可以使用, 一个是 `rvalue reference`, 另一个是 `lvalue reference`
    * 此时会使用 右值引用的那个重载


### 3.1.1. lvalue 左值

* lvalue : 左值, 因为可以放在赋值运算符的左边

左值是最基本的类型, 特征:
1. 可以进行取址运算 `&a`  `&std::endl`, 在内存中有空间
2. `可以修改的左值` 可以用作内建赋值和内建复合赋值运算符的`左操作数` (说人话就是可以被赋值)
3. 左值可以初始化一个`左值引用`, 即引用(将一个新名字关联给表达式所标识的对象)
4. 与泛左值相同的特性

具体描述:
* 所有变量, 函数, 模板形参对象, 数据成员. - 即使变量的类型是右值引用, 这个变量的名字构成的表达式依然是左值表达式
* `a=b` 等内建赋值表达式
* `++a` 等内建`前置`自增表达式
* `*p`  内建的间接寻址表达式
* `a[i]`内建的下表表达式
* `a.b  a->b` 内建的成员表达式
* `"abc"`字符串的字面量 (特殊)
* 返回类型为左值引用的函数调用或重载运算符表达式


### 3.1.2. prvalue 纯右值

* prvalue :  (pure rvalue)  也是表达式 
  * 计算内置运算符的运算值  (无 result object 的 pvalue)
  * 初始化一个对象          (有 result object 的 pvalue)

纯右值一般代表一次性的值, 特性
* `12 true nullptr` 等字面量 (除了字符串)
* `a--` 等内建`后置`自增表达式

### 3.1.3. xvalue

* xvalue :  (eXpiring value) 是一个 资源可以被重复使用的 glvalue

eXpiring value 的表达式:
* 返回值是 右值引用的 函数调用 or 重载运算符表达式(其实也是函数调用)
* 下标表达式 `a[n]`
* 成员表达式 `a.m` 
* 条件表达式 `a ? b : c`
* cast表达式 将右值引用转换成目标类型`static_cast<char&&>(x)`

## 3.2. conversions (cast)

同 C 语言的类型转换相比, C++提供了更加精细化的操作

类型转换包括基础的标准转换和条件限定的转换:
* 隐式转换 (标准转换) 
* const
  * 两个指针之间, 指针指向的类型可以相互转换, 则转换成立
  * 左值转换成 左值或者右值引用, 
  * 空值指针之间
* static
* dynamic
* reinterpret
* explicit cast conversion using C-style cast notation and functional notation  
* user-defined conversion makes it possible to specify conversion from user-defined classes 

```cpp
// const_cast < new-type > ( expression ) 		



// static_cast < new-type > ( expression ) 		

```

# 4. Initialization 初始化


## 4.1. Reference initialization


# 5. Classes 类

https://en.cppreference.com/w/cpp/language/classes 

A class is a user-defined type. 
A class type is defined by class-specifier, which appears in `decl-specifier-seq` of the declaration syntax.



## 5.1. 初始化列表

在函数首部与函数体之间添加了一个冒号:  后面紧跟`成员名(参数名),成员名(参数名)`  
1. 初始化列表可以将构造函数变得更简洁
2. 初始化 const 成员变量。初始化 const 成员变量的唯一方法就是使用初始化列表。  
```cpp
class Student{
private:
    char *m_name;
    int m_age;
    float m_score;
public:
    Student(char *name, int age, float score);
};

// 使用初始化列表对成员赋值
Student::Student(char *name, int age, float score): m_name(name), m_age(age), m_score(score){
    //TODO:
}

```

注意: 成员变量的初始化顺序只与成员变量在类中声明的顺序有关, 与初始化列表中列出的变量的顺序无关

```cpp
class Demo{
private:
    // a 在 b 的前面
    int m_a;
    int m_b;
public:
    Demo(int b);
};

// 虽然初始化列表里 先是 m_b(b) , 但实际上先赋值的仍然是 m_a(m_b) 此时 m_a 变成了随机数
Demo::Demo(int b): m_b(b), m_a(m_b){ }

// 相当于
Demo::Demo(int b){
  m_a = m_b;
  m_b = b;
}

```


## 5.2. 类和struct的区别 C++中增强的struct

* C++ 中保留了C语言的 struct 关键字，并且加以扩充。  
* 在C语言中，struct 只能包含成员变量，不能包含成员函数。  
* 而在C++中，struct 类似于 class，既可以包含成员变量，又可以包含成员函数。  


C++中的 struct 和 class 基本是通用的，唯有几个细节不同：
* 使用 class 时，类中的成员默认都是 `private` 属性的；而使用 `struct` 时，结构体中的成员默认都是 `public` 属性的。
* class 继承默认是 `private` 继承，而 `struct` 继承默认是 `public` 继承（《C++继承与派生》一章会讲解继承）。
* class 可以使用模板，而 `struct` 不能（《模板、字符串和异常》一章会讲解模板）。

对于一段没有标明访问权限的成员
```cpp
struct Student{
    Student(char *name):m_name(name);
    void show();
    char *m_name;
};
Student *pstu = new Student("李华", 16, 96);
pstu -> show();

```

struct 默认的成员都是 public 属性的，所以可以通过对象访问成员函数。如果将 struct 关键字替换为 class，那么就会编译报错。

## 5.3. 构造函数 转换构造函数

C++的构造函数除了定义对象, 还可以作为转换构造函数  
转换构造函数用来将其它类型 (可以是 bool、int、double 等基本类型，也可以是数组、指针、结构体、类等构造类型) 转换为当前类类型  

```cpp

// 对于一个普通的复数类

// 普通的使用浮点数的构造函数
Complex(double real): m_real(real), m_imag(0.0){ }


// 能够实现 对象和基础类型的直接相加
Complex c1(25, 35);
Complex c2 = c1 + 15.6;
// 隐式转换函数 实际上执行的是这个
Complex c2 = c1 + Complex(15.6)
```

## 5.4. 函数对象

### 5.4.1. 使用定义
如果一个类将`()` 运算符重载为成员函数  这个类的对象就是函数对象  

1. 函数对象是一个对象, 但是使用的形式看起来 像函数调用
2. 函数对象的 operator() 成员函数可以根据对象内部的不同状态执行不同操作，而普通函数就无法做到这一点。
   因此函数对象的功能比普通函数更强大


```cpp
class CAverage
{
public:
    // ()是目数不限的运算符，因此重载为成员函数时，有多少个参数都可以。
    double operator()(int a1, int a2, int a3) //参数
    {  //重载()运算符
        return (double)(a1 + a2 + a3) / 3;
    }
};
cout << average(3, 2, 3); 
```

函数对象的应用
1. accumulate 算法
2. sort　算法
3. 


**在 accumulate 中使用**
```cpp
// accumulate 的定义
template <class InIt, class T, class Pred>
T accumulate(InIt first, Init last, T init, Pred op)
{
  for (; first != last; ++first)
      init = op(init, *first);
  return init;
};
//  op 只能是函数指针或者函数对象

// 建立模板函数类
template<class T>
class SumPowers
{
private:
    int power;
public:
    SumPowers(int p) :power(p) { }
    const T operator() (const T & total, const T & value)
    { //计算 value的power次方，加到total上
        T v = value;
        for (int i = 0; i < power - 1; ++i)
            v = v * value;
        return total + v;
    }
};

vector<int> v{1,2,3,4,5,6,7,8,9};
result = accumulate(v.begin(), v.end(), 0, SumPowers<int>(3));
cout << "立方和：" << result << endl;
result = accumulate(v.begin(), v.end(), 0, SumPowers<int>(4));
cout << "4次方和：" << result;
```

**在 sort 中使用**

```cpp
// sort 算法有两个版本 , 一个是使用默认规则 < , 一个是使用自定义规则

// a<b的值为 true，则 a 排在 b 前面；
// 如果a<b的值为 false，则还要看b<a是否成立，成立的话 b 才排在 a 前面。
template <class_Randlt>
void sort(_Randlt first, _RandIt last);

// 使用自定义规则 OP
template <class_Randlt, class Pred>
void sort(_Randlt first, _RandIt last, Pred op);



// 例: 自定义类
class A
{
public:
    int v;
    A(int n) : v(n) {}
}a[5] = { 13, 12, 9, 8, 16 };;

// 重载运算符
bool operator < (const A & a1, const A & a2)
{  //重载为 A 的 const 成员函数也可以，重载为非 const 成员函数在某些编译器上会出错
    return a1.v < a2.v;
}
// 使用默认的 < 符号
sort(a.begin(),a.end());

// 自定义增续排序函数
bool GreaterA(const A & a1, const A & a2)
{  //v值大的元素作为较小的数
    return a1.v > a2.v;
}
sort(a.begin(),a.end(), GreaterA);

// 结构体类中的函数对象
struct LessA
{
    bool operator() (const A & a1, const A & a2)
    {  //v的个位数小的元素就作为较小的数
        return (a1.v % 10) < (a2.v % 10);
    }
};
// 使用函数对象
sort(a.begin(), a.end(), LessA());
```
## 5.5. STL 函数对象模板类

| 函数对象类模板   | 成员函数 T operator ( const T & x, const T & y) 的功能 |
| ---------------- | ------------------------------------------------------ |
| `plus <T>`       | return x + y;                                          |
| minus < >        | return x - y;                                          |
| `multiplies <T>` | return x * y;                                          |
| `divides <T>`    | return x / y;                                          |
| `modulus <T>`    | return x % y;                                          |

| 函数对象类模板      | 成员函数 bool operator ( const T & x, const T & y) 的功能 |
| ------------------- | --------------------------------------------------------- |
| `equal_to <T>`      | return x == y;                                            |
| `not_equal_to <T>`  | return x! = y;                                            |
| `greater <T>`       | return x > y;                                             |
| `less <T>`          | return x < y;                                             |
| `greater_equal <T>` | return x > = y;                                           |
| `less_equal <T>`    | return x <= y;                                            |
| `logical_and <T>`   | return x && y;                                            |
| `logical_or <T>`    | return x                                                  |  | y; |


成员函数 T operator( const T & x) 的功能  
`negate <T>`  return - x;  

成员函数 bool operator( const T & x) 的功能  
`logical_not <T>`  return ! x;  


```cpp
// 要求两个 double 型变量 x、y 的乘积
multiplies<double> () (x, y)

// 要判断两个 int 变量 x、y 中 x 是否比 y 小
if( less<int>()(x, y) ) {
    //
}
```

## 5.6. Operator Overloading 运算符重载

能够重载的运算符:  
1. `+  -  *  /  %`
2. `^  &  | ~`  
3. `! ==  != =  <  >  <=  >= +=  -=  *=  /=  %=  ^=  &=  |=`
4. `<<  >>  <<=  >>=`      
5. `&&  ||`
6. `++  --`  
7. `,  ->*  ->  ()  []`
8. `new  new[]  delete  delete[]`

长度运算符 `sizeof` 、条件运算符: `?` 、成员选择符`.` 和域解析运算符`::` 不能被重载。

重载函数的特性:  
1. 重载不能改变运算符的优先级和结合性
2. 重载不会改变运算符的用法，原有有几个操作数、操作数在左边还是在右边，这些都不会改变
3. 运算符重载函数不能有默认的参数
4. 将运算符重载函数作为 `类的成员函数` 时，二元运算符的参数只有一个，一元运算符则不需要参数
5. 将运算符重载函数作为 `全局函数` 时, 参数其中必须有一个参数是对象，好让编译器区分这是程序员自定义的运算符，防止程序员修改用于内置类型的运算符
6. 将运算符重载函数作为全局函数时，一般都需要在类中将该函数声明为友元函数。因为运算符函数大部分情况下都需要使用类的 private 成员。
7. 箭头运算符 `->` 、下标运算符` []`、函数调用运算符 `( )` 、赋值运算符 `=` 只能以成员函数的形式重载

* 优先考虑成员函数，这样更符合运算符重载的初衷
* 有一部分运算符重载必须是全局函数，这样能保证参数的对称性
* C++ 规定，箭头运算符`->`、下标运算符`[ ]`、函数调用运算符`( )`、赋值运算符`=`只能以成员函数的形式重载。


### 5.6.1. 为什么要以全局函数重载运算符 加号 +


**+ 运算符具有左结合性**  
```cpp
// 输入运算
Complex c2 = c1 + 15.6;
// 加号左结合, 此时浮点数是 c1的成员函数的参数, 会被隐式转换
Complex c2 = c1.operator+(Complex(15.6));

// 如果定义为成员函数, 对于运算
Complex c3 = 28.23 + c1;
// 因为 double 类型并没有成员函数重载 + , 因此不能转换复数 c1
// 而且 C++ 只会对成员函数的参数进行类型转换，而不会对调用成员函数的对象进行类型转换
...
// 报错
```
定义的operator+是一个全局函数（一个友元函数），而不是成员函数，这样做是为了保证 + 运算符的操作数能够被对称的处理  


```cpp

// 运算符重载格式 operator 是关键字
returntype operator 运算符名称 (parameter){)

// 在类内部声明一个成员复数加法的运算符重载函数
complex operator+(const complex &A) const;

// 在类外部声明一个全局复数加法
// 在类内部将函数声明成友元这样可以访问私有变量
friend complex operator+(const complex &A, const complex &B);
// 在全局范围内重载
complex operator+(const complex &A, const complex &B){} 
```

## 5.7. const

### 5.7.1. C语言的常量

常量一旦被创建后其值就不能再改变，所以常量必须在定义的同时赋值（初始化），后面的任何赋值行为都将引发错误。
const 也可以和指针变量一起使用，这样可以限制指针变量本身，也可以限制指针指向的数据。  

```c
const int MaxNum = 100; 

// const 和 type 都是用来修饰变量的，它们的位置可以互换
// 建议将常量名的首字母大写，以提醒程序员这是个常量
const type Name = value;

// 常量指针的三种形式
const int *p1;    // 常用这种 , 指针所指向的数据是只读的
int const *p2;    // 这两种也就是 const 和 type 可以互换的类型 , 指针所指向的数据是只读的
int * const p3;   // 指针是只读的，也就是 p3 本身的值不能被修改；

// 还有一种指针本身和它指向的数据都有可能是只读的
const int * const p4;
int const * const p5; // 交换后的写法

```

在C语言中，单独定义 const 变量没有明显的优势，完全可以使用#define命令代替。  
因此: const 通常用在函数形参中，如果形参是一个指针，为了防止在函数内部修改指针指向的数据，就可以用 const 来限制。  

### 5.7.2. C++的 const

C++中的 const 更像编译阶段的 #define
```cpp
const int m = 10;
int n = m;
```
m、n 两个变量占用不同的内存，int n = m;表示将 m 的值赋给 n，这个赋值的过程在C和C++中是有区别的。
1. 在C语言中，编译器会先到 m 所在的内存取出一份数据，再将这份数据赋给 n；
2. 而在C++中，编译器会直接将 10 赋给 m，没有读取内存的过程，和int n = 10;的效果一样
   只不过#define是在预处理阶段替换，而常量是在编译阶段替换。

### 5.7.3. 类的 const 函数

在类中，如果你不希望某些数据被修改，可以使用const关键字加以限定。

const 可以用来修饰成员变量和成员函数。
1. 初始化 const 成员变量只有一种方法，就是通过构造函数的初始化列表
2. const 成员函数可以使用类中的所有成员变量，但是不能修改它们的值，
   这种措施主要还是为了保护数据而设置的。const 成员函数也称为`常成员函数`。


* 常成员函数需要在声明和定义的时候在函数头部的结尾加上 const 关键字  !注意是结尾,括号后面!
* 需要强调的是，必须在成员函数的声明和定义处同时加上 const 关键字    !声明和定义处都要加!
* 我们通常将 get 函数设置为常成员函数.  

```cpp
char *getname() const;
int getage() const;
float getscore() const;
```
### 5.7.4. const 对象

1. const 也可以用来修饰对象，称为常对象。  
2. 一旦将对象定义为常对象之后，就只能调用类的 const 成员（包括 const 成员变量和 const 成员函数）了。  
3. 因为非 const 成员`可能`会修改对象的数据（编译器也会这样`假设`），因此C++禁止这样做。

```cpp
// 定义常量对象的两种 是互换写法 , 常用第一种
const  class  object(params);
class const object(params);
```

## 5.8. Inheritance 类的继承

由一个类继承而来的类被称为 派生类 derived class

```cpp
attr class-or-decltype
// 1 Specifies a non-virtual inheritance with default member accessibility.

attr virtual class-or-decltype 	
// 2 Specifies a virtual inheritance with default member accessibility.

attr access-specifier class-or-decltype 	
// 3 Specifies a non-virtual inheritance with given member accessibility.

attr virtual access-specifier class-or-decltype 	
// 4 Specifies a virtual inheritance with given member accessibility.

attr access-specifier virtual class-or-decltype 	
// 5 Same as 4), virtual and access-specifier can appear in any order.
```

* attr (C++11)          : C++11 的 attributes 特性
* `access-specifier`    : 访问指示符, 即  	one of private, public, or protected
* `class-or-decltype`   : 具体的 类 的名称, 可以是
  *  nested-name-specifier(optional) type-name 
  *  nested-name-specifier template simple-template-id 
  *  decltype-specifier (since C++11)
  *  TODO : An `elaborated type specifier` cannot directly appear as class-or-decltype due to syntax limitations. 


# 6. Templates 模板

Templates 是C++的一个大的特性, 也是各种 STL 的基础  
以 template 为关键字, 方便的创建一套面向多数据类型的 函数/实体

A template is a C++ entity that defines one of the following: 
* 模板类, a family of classes (class template), which may be nested classes
* 模板函数, a family of functions (function template), which may be member functions
* 模板的别名 (since C++11), an alias to a family of types (alias template)  
* 模板变量 (C++14), a family of variables (variable template)
* (since C++20) a concept (constraints and concepts) 


一个模板被 一个或者多个 `template parameters` 来参数化定义
* 数据类型 
* 非数据类型
* 模板 (递归定义)


语法:
```cpp
template < parameter-list > requires-clause(optional) declaration

// C++11 中被废弃的
export template < parameter-list > declaration

// C++20 中新增的 
template < parameter-list > concept concept-name = constraint-expression ; 	
```

语法中的名词解释:


## 6.1. full (explicit) specializations

## 6.2. partial specializations 


# 7. Exceptions C++ 异常

* C++的异常类使用需要调用 `<exception>`头文件

```cpp
try{
    // 可能抛出异常的语句
}catch(exceptionType variable1){
    // 处理异常的语句
}catch(exceptionType variable2){
    //多级 catch
}
```

* catch 部分可以看作一个没有返回值的函数, 异常发生后catch会被调用, 并且可以接受实参, 即异常数据
* catch 和真正的函数调用相比, 多了一个「在运行阶段才将实参和形参匹配」的过程
* 如果不希望catch 处理异常数据, 也可以将 variable 省略掉, 这样就不再会传递异常数据了
* 多级catch在异常匹配的时候会进行向上转型, 因此父异常类要放在子异常类的后面



## 7.1. throw expression

手动抛出异常


```cpp
throw exceptionData;


// 异常规范 Exception specification 于 C++98提出, 于C++11抛弃
// 当前编译器已经不再支持 异常规范
// 该函数只能抛出 int 类型的异常
double func (char param) throw (int);
// 抛出多种异常
double func (char param) throw (int, char, exception);
// 不会抛出任何异常, 即写了 try 也会失效
double func (char param) throw ();

```
* throw 关键字可以显式的抛出异常
* exceptionData 是异常数据, 可以是任何类型
  * int, float, bool, 指针等基本类型
  * 数组, 字符串, 结构体, 类 等聚合类型也可以
* throw除了写在函数体中代表实际抛出异常, 也可以写在函数头的部分
  * 指明该函数可以抛出的异常类型, 这称为 `异常规范 (Exception specification) `
  * 由于使用方法太过模糊, 已经被舍弃




## 7.2. exceptionType exception类

* exceptionType 是异常类型, 代表指定当前 catch 可以处理什么异常
* exception 类是所有异常类的基类, 用 exception 可以接受所有异常, 被称为标准异常 Standard Exception

