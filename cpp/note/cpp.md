- [1. C++ 特性](#1-c-特性)
  - [1.1. 值类别 Value Categories](#11-值类别-value-categories)
    - [1.1.1. lvalue 左值](#111-lvalue-左值)
    - [1.1.2. prvalue 纯右值](#112-prvalue-纯右值)
  - [1.2. 程序参数配置](#12-程序参数配置)
    - [1.2.1. C的输出与C++流的同步](#121-c的输出与c流的同步)
    - [1.2.2. cin 和 cout 的绑定](#122-cin-和-cout-的绑定)
- [2. 类](#2-类)
  - [2.1. 初始化列表](#21-初始化列表)
  - [2.2. 类和struct的区别 C++中增强的struct](#22-类和struct的区别-c中增强的struct)
  - [2.3. 构造函数 转换构造函数](#23-构造函数-转换构造函数)
  - [2.4. 函数对象](#24-函数对象)
    - [2.4.1. 使用定义](#241-使用定义)
  - [2.5. STL 函数对象模板类](#25-stl-函数对象模板类)
  - [2.6. Operator Overloading 运算符重载](#26-operator-overloading-运算符重载)
    - [2.6.1. 为什么要以全局函数重载运算符 加号 +](#261-为什么要以全局函数重载运算符-加号-)
  - [2.7. const](#27-const)
    - [2.7.1. C语言的常量](#271-c语言的常量)
    - [2.7.2. C++的 const](#272-c的-const)
    - [2.7.3. 类的 const 函数](#273-类的-const-函数)
    - [2.7.4. const 对象](#274-const-对象)

# 1. C++ 特性

## 1.1. 值类别 Value Categories

每个C++`表达式`的种类都可以按照两种独立的特性进行区分
1. 类型, 每个值都有的 eg. int, char
2. 值类别



值类别可以进行多段细分, 且与C++的版本更替进行着变更:
* 基础的三大类
  * lvalue
  * prvalue
  * xvalue


### 1.1.1. lvalue 左值

左值是最基本的类型, 特征:
1. 可以进行取址运算 `&a`  `&std::endl`
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


### 1.1.2. prvalue 纯右值

纯右值一般代表一次性的值, 特性
* `12 true nullptr` 等字面量 (除了字符串)
* `a--` 等内建`后置`自增表达式


## 1.2. 程序参数配置

### 1.2.1. C的输出与C++流的同步

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

### 1.2.2. cin 和 cout 的绑定

`std :: cin`默认是与`std :: cout`绑定的，所以每次操作的时候都要调用`fflush`  
这样增加了IO的负担，通过`tie(nullptr)`来解除`cin`和`cout`之间的绑定，进一步加快执行效率。

```cpp
// std::tie是将两个stream绑定的函数，空参数的话返回当前的输出流指针。

std::cin.tie(nullptr);
std::cout.tie(nullptr);
```



# 2. 类

## 2.1. 初始化列表

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


## 2.2. 类和struct的区别 C++中增强的struct

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

## 2.3. 构造函数 转换构造函数

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

## 2.4. 函数对象

### 2.4.1. 使用定义
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
## 2.5. STL 函数对象模板类

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

## 2.6. Operator Overloading 运算符重载

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


### 2.6.1. 为什么要以全局函数重载运算符 加号 +


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

## 2.7. const

### 2.7.1. C语言的常量

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

### 2.7.2. C++的 const

C++中的 const 更像编译阶段的 #define
```cpp
const int m = 10;
int n = m;
```
m、n 两个变量占用不同的内存，int n = m;表示将 m 的值赋给 n，这个赋值的过程在C和C++中是有区别的。
1. 在C语言中，编译器会先到 m 所在的内存取出一份数据，再将这份数据赋给 n；
2. 而在C++中，编译器会直接将 10 赋给 m，没有读取内存的过程，和int n = 10;的效果一样
   只不过#define是在预处理阶段替换，而常量是在编译阶段替换。

### 2.7.3. 类的 const 函数

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
### 2.7.4. const 对象

1. const 也可以用来修饰对象，称为常对象。  
2. 一旦将对象定义为常对象之后，就只能调用类的 const 成员（包括 const 成员变量和 const 成员函数）了。  
3. 因为非 const 成员`可能`会修改对象的数据（编译器也会这样`假设`），因此C++禁止这样做。

```cpp
// 定义常量对象的两种 是互换写法 , 常用第一种
const  class  object(params);
class const object(params);
```

