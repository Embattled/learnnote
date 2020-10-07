# 1. c++ 比较高级的 容易忘记的内容

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



## 1.2. 类

### 1.2.1. 初始化列表

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


### 1.2.2. 类和struct的区别 C++中增强的struct

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



## 1.2.3. 函数对象

### 使用定义
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
### STL 函数对象模板类

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


## 1.3. const

### 1.3.1. C语言的常量

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

### 1.3.2. C++的 const

C++中的 const 更像编译阶段的 #define
```cpp
const int m = 10;
int n = m;
```
m、n 两个变量占用不同的内存，int n = m;表示将 m 的值赋给 n，这个赋值的过程在C和C++中是有区别的。
1. 在C语言中，编译器会先到 m 所在的内存取出一份数据，再将这份数据赋给 n；
2. 而在C++中，编译器会直接将 10 赋给 m，没有读取内存的过程，和int n = 10;的效果一样
   只不过#define是在预处理阶段替换，而常量是在编译阶段替换。

### 1.3.3. 类的 const 函数

在类中，如果你不希望某些数据被修改，可以使用const关键字加以限定。

const 可以用来修饰成员变量和成员函数。
1. 初始化 const 成员变量只有一种方法，就是通过构造函数的初始化列表
2. const 成员函数可以使用类中的所有成员变量，但是不能修改它们的值，
   这种措施主要还是为了保护数据而设置的。const 成员函数也称为`常成员函数`。



* 常成员函数需要在声明和定义的时候在函数头部的结尾加上 const 关键字  
* 需要强调的是，必须在成员函数的声明和定义处同时加上 const 关键字
* 我们通常将 get 函数设置为常成员函数.  

```cpp
char *getname() const;
int getage() const;
float getscore() const;
```
### 1.3.4. const 对象

1. const 也可以用来修饰对象，称为常对象。  
2. 一旦将对象定义为常对象之后，就只能调用类的 const 成员（包括 const 成员变量和 const 成员函数）了。  
3. 因为非 const 成员可能会修改对象的数据（编译器也会这样假设），C++禁止这样做。

```cpp
// 定义常量对象的两种 , 常用第一种
const  class  object(params);
class const object(params);


```


# 2. C++11标准

C++11 标准对 C++ 语言增添了约 140 个新特性

# 3. C++11 空指针

nullptr是C++11版本中新加入的，它的出现是为了解决NULL表示空指针在C++中具有二义性的问题  

C++中关于空指针的定义  
```c++
// C++98/03 标准中，将一个指针初始化为空指针的方式有 2 种

// 将指针明确指向 0（0x0000 0000）这个内存空间。一方面，明确指针的指向可以避免其成为野指针；
// 另一方面，大多数操作系统都不允许用户对地址为 0 的内存空间执行写操作，若用户在程序中尝试修改其内容，则程序运行会直接报错。
int *p2 = 0;

// 值得一提的是，NULL 并不是 C++ 的关键字，它是 C++ 为我们事先定义好的一个宏，并且它的值往往就是字面量 0（#define NULL 0）
int *p1 = NULL; // 需要引入cstdlib头文件

int *p3 = nullptr;
```

1. 关于NULL的定义
   在C++中,NULL被直接定义成整数0  
   而在C中,NULL的定义为 `#define NULL ((void *)0)`,也就是说NULL实际上是一个void* 指针  
   造成这种差异的原因是为了照顾C++的函数重载  
```Cpp
void Func(char *);
void Func(int);

int main()
{
    //C语言没有重载, 是弱类型语言,因此可以允许void*进行隐式转换  
    //C++为了重载, 是强类型语言,因此不允许void*的隐式重载, 将NULL更改为了整数0  
    //而根据传统思维, NULL应该代表的是char* , 但是因为C++中NULL就是整数0, 所以实际执行的是int的函数
    Func(NULL);
}
```
2. nullptr的使用
   注意`nullptr`是在C++11中引入的  
   nullptr可以转化为任意的指针和布尔值, 但是不能转变为整数  

# 4. C++11 的for循环

## 4.1. 语法和使用方法
C++ 11标准之前（C++ 98/03 标准），如果要用 for 循环语句遍历一个数组或者容器，只能套用如下结构
```cpp
for(表达式 1; 表达式 2; 表达式 3){
    //循环体
}
```

而C++11 引用了新的语法结构, 类似于其他脚本语言的for循环
```cpp
// 语法
for (declaration : expression){
    //循环体
}

// 遍历容器
vector<char>myvector(arc, arc + 23);
for (auto ch : myvector) {
  cout << ch;
}

// 遍历用{}大括号初始化的列表
for (int num : {1, 2, 3, 4, 5}) {
  cout << num << " ";
}
```
* declaration : 表示此处要定义一个变量，该变量的类型为要遍历序列中存储元素的类型
  * declaration参数处定义的变量类型可以用 auto 关键字表示
  * 注意如果对象是 stl容器 , auto 后的变量是元素类型而不是迭代器类型  
* expression  : 表示要遍历的序列，常见的可以为事先定义好的**普通数组**或者**容器**，还可以是用 {} 大括号初始化的序列。

同 C++ 98/03 中 for 循环的语法格式相比较，此格式并没有明确限定 for 循环的遍历范围，这是它们最大的区别  
旧格式的 for 循环可以指定循环的范围，而 C++11 标准增加的 for 循环，只会逐个遍历 expression 参数处指定序列中的**每个元素**。  

注意 declaration :
* `for (auto ch : myvector)`  读取出的ch值是拷贝后的值
  * 因此对于 stl map 来说,取出的值是 pair 类型
* `for (auto &ch : myvector)` 使用引用类型才可以修改原本数组的值
因此,常规来说 :
* 如果需要在遍历序列的过程中修改器内部元素的值，就必须定义引用形式的变量
* 反之，建议定义`const &`（常引用）形式的变量（避免了底层复制变量的过程，效率更高）


C++11 for 循环的特殊用法
```cpp
// 基于范围的 for 循环也可以直接遍历某个字符串，比如：
// 也可以遍历 string 类型的字符串，这种情况下同样冒号前定义 char 类型的变量即可
for (char ch : "http://c.biancheng.net/cplus/11/") {
    cout << ch;
}


// 注意，基于范围的 for 循环不支持遍历函数返回的以指针形式表示的数组
char str[] = "http://c.biancheng.net/cplus/11/";
char* retStr() {
    return str;
}
for (char ch : retStr()) //直接报错
{
    cout << ch;
}

```

## 4.2. 注意事项

基于关联式容器（包括哈希容器）底层存储机制的限制：
* 不允许修改 map、unordered_map、multimap 以及 unordered_multimap 容器存储的键的值；
* 不允许修改 set、unordered_set、multiset 以及 unordered_multiset 容器中存储的元素的值。
* 因此，当使用基于范围的 for 循环遍历此类型容器时，切勿修改容器中不允许被修改的数据部分
基于范围的 for 循环完成对容器的遍历，其底层也是借助容器的迭代器实现的
* 因此，在使用基于范围的 for 循环遍历容器时，应避免在循环体中修改容器存储元素的个数。

# 5. C++11 的 右值引用 移动 转发
## 5.1. 右值引用

C++11 新增了一种引用，可以引用右值，因而称为“右值引用”。  
无名的临时变量不能出现在赋值号左边，因而是右值。右值引用就可以引用无名的临时变量。  

能出现在赋值号左边的表达式称为“左值”，不能出现在赋值号左边的表达式称为“右值”。
1. 左值是可以取地址
2. 右值则不可以

* 非 const 的变量都是左值。
* 函数调用的返回值若不是引用，则该函数调用就是右值。
* 一般所学的“引用”都是引用变量的，而变量是左值，因此它们都是“左值引用”.

左值的英文简写为“lvalue”，右值的英文简写为“rvalue”。很多人认为它们分别是"left value"、"right value" 的缩写，其实不然。  
`lvalue` 是 `loactor value` 的缩写，可意为存储在内存中、有明确存储地址（可寻址）的数据.   
`rvalue` 译为 ` read value` ，指的是那些可以提供数据值的数据（不一定可以寻址，例如存储于寄存器中的数据）。  


引用类型        可以引用的值类型 	                         使用场景  
                非常量左值 常量左值 非常量右值 常量右值   
非常量左值引用      Y         N   	    N        N 	        无
常量左值引用        Y         Y   	    Y        Y 	        常用于类中构建拷贝构造函数  
非常量右值引用      N         N   	    Y        N          移动语义、完美转发
常量右值引用        N         N   	    Y        Y          无实际用途

### 5.1.1. 定义

右值往往是没有名称的，因此要使用它只能借助引用的方式。这就产生一个问题，实际开发中我们可能需要对右值进行修改（实现移动语义时就需要），显然左值引用的方式是行不通的。  
C++11 标准新引入了另一种引用方式，称为右值引用，用 `&&` 表示。    


```cpp
// 传统的引用定义
// C++98/03 标准不支持为右值建立非常量左值引用  
int num = 10;
int &b = num; //正确
int &c = 10; //错误

// 但是支持常量引用
int num = 10;
const int &b = num;
const int &c = 10;


// C++11 的右值引用定义
//类型 && 引用名 = 右值表达式;
int num = 10;
//int && a = num;  //右值引用不能初始化为左值
int && a = 10;

// 使用了右值引用就可以对右值进行修改
int && a = 10;
a = 100;
cout << a << endl;


class A{};
A & rl = A();  //错误，无名临时变量 A() 是右值，因此不能初始化左值引用 r1
A && r2 = A();  //正确，因 r2 是右值引用

```



### 5.1.2. 使用场景

引入右值引用的主要目的是提高程序运行的效率。有些对象在复制时需要进行深复制，深复制往往非常耗时。合理使用右值引用可以避免没有必要的深复制操作。  
右值引用主要用于实现移动（move）语义和完美转发  


## 5.2. 移动构造函数

知道了右值引用,就可以学习移动构造函数 ,引入了右值引用的语法，借助它可以实现移动语义。   

### 5.2.1. 定义
在 C++ 11 标准之前（C++ 98/03 标准中），如果想用其它对象初始化一个同类的新对象，只能借助类中的复制（拷贝）构造函数。  

需要注意的是，当类中拥有指针类型的成员变量时，拷贝构造函数中需要以深拷贝（而非浅拷贝）的方式复制该指针成员。  
即拷贝该指针成员本身的同时，还要拷贝指针指向的内存资源。否则一旦`多个对象中的指针成员指向同一块堆空间，这些对象析构时就会对该空间释放多次`，这是不允许的。  


移动构造函数的目的, 是解决函数的无用拷贝  
1. `return demo()`  时, `demo()`  构造函数产生一个匿名对象, 然后复制给返回值对象, 匿名对象析构
2. `a = get_demo()` 时, 又从返回值对象执行复制给a, 然后返回值对象析构
假如 `demo()` 类需要深拷贝,那么一个初始化操作就执行了2次无用拷贝  
事实上, 编译器对这些过程做了专门的优化, 如果使用 VS 2017、codeblocks 等这些编译器运行此程序时，则不会有以上两次无用拷贝, 且并不会影响程序的正确性，因此很少进入程序员的视野。  

此问题一直存留在以 C++ 98/03 标准编写的 C++ 程序中.



### 5.2.2. 使用方法

当类中同时包含拷贝构造函数和移动构造函数时
1. 如果使用临时对象初始化当前类的对象，编译器会优先调用移动构造函数来完成此操作
2. 只有当类中没有合适的移动构造函数时，编译器才会退而求其次，调用拷贝构造函数。
   
在实际开发中，通常在类中自定义移动构造函数的同时，会再为其自定义一个适当的拷贝构造函数
1. 由此当用户利用右值初始化类对象时，会调用移动构造函数
2. 使用左值（非右值）初始化类对象时，会调用拷贝构造函数。

```cpp

//初始化构造函数
demo():num(new int(0)){
     cout<<"construct!"<<endl;
}


//拷贝构造函数 深拷贝
demo(const demo &d):num(new int(*d.num)){
    cout<<"copy construct!"<<endl;
}

//添加移动构造函数
demo(demo &&d):num(d.num){
    // 将原本对象的 指针成员置空, 防止对象析构的时候多次释放空间  
    d.num = NULL;
    cout<<"move construct!"<<endl;
}
```


如果使用左值初始化同类对象，但也想调用移动构造函数完成的时候
1. 默认情况下，左值初始化同类对象只能通过拷贝构造函数完成，如果想调用移动构造函数，则必须使用右值进行初始化。
2. C++11 标准中为了满足用户使用左值初始化同类对象时也通过移动构造函数完成的需求，新引入了 std::move() 函数，它可以将左值强制转换成对应的右值，由此便可以使用移动构造函数。 

## 5.3. move()函数 将左值强制转换为右值

* C++11 标准中借助右值引用可以为指定类添加移动构造函数，这样当使用该类的右值对象（可以理解为临时对象）初始化同类对象时，编译器会优先选择移动构造函数。  
* 那么，用当前类的左值对象（有名称，能获取其存储地址的实例对象）初始化同类对象时，想使用移动构造函数则可以调用 move() 函数。

move 本意为 "移动"，但该函数并不能移动任何数据，它的功能很简单，就是将某个左值强制转化为右值。   

move() 函数的用法也很简单，其语法格式如下：  
`move( arg )`  
其中，arg 表示指定的左值对象。该函数会返回 arg 对象的右值形式。  

使用move来实现传统的 swap函数:
```cpp
void MoveSwap(T & a, T & b) {
    T tmp(move(a));  //std::move(a) 为右值，这里会调用移动构造函数
    a = move(b);  //move(b) 为右值，因此这里会调用移动赋值号
    b = move(tmp);  //move(tmp) 为右值，因此这里会调用移动赋值号
}
```

## 5.4. 完美转发 forward()
 
完美转发: 指的是函数模板可以将自己的参数“完美”地转发给内部调用的其它函数。  
所谓完美，即不仅能准确地转发参数的值，还能保证被转发参数的`左、右值属性`不变。  

如果使用 C++ 98/03 标准下的 C++ 语言，我们可以采用函数模板重载的方式实现完美转发  
使用重载的模板函数实现完美转发也是有弊端的，此实现方式仅适用于模板函数仅有少量参数的情况，否则就需要编写大量的重载函数模板，造成代码的冗余。  

没有完美转发的代码:  
```cpp
template<typename T>
void function(T t) {
    otherdef(t);
}
// 在此情况下, 无论调用 function() 函数模板时传递给参数 t 的是左值还是右值，对于函数内部的参数 t 来说，它有自己的名称，也可以获取它的存储地址，因此它永远都是左值
// 此外, 参数 t 为非引用类型，这意味着在调用 function() 函数时，实参将值传递给形参的过程就需要额外进行一次拷贝操作
```
如果 function() 函数接收到的参数 t 为左值，那么该函数传递给 otherdef() 的参数 t 也是左值  
反之如果 function() 函数接收到的参数 t 为右值，那么传递给 otherdef() 函数的参数 t 也必须为右值  


为了方便用户更快速地实现完美转发，C++ 11 标准中允许在函数模板中使用右值引用来实现完美转发。
C++11 标准中规定，通常情况下右值引用形式的参数只能接收右值，不能接收左值。
* 但对于函数模板中使用右值引用语法定义的参数来说，它不再遵守这一规定，既可以接收右值，也可以接收左值（此时的右值引用又被称为`万能引用`）  
* 由于引用折叠规则存在, 读者只需要知道，在实现完美转发时，只要函数模板的参数类型为 T&&，则 C++ 可以自行准确地判定出实际传入的实参是左值还是右值。


对于函数模板内部来说，形参既有名称又能寻址，因此它都是左值, 为了完美转发给函数内部的函数  
C++11新标准还引入了一个模板函数 `forword<T>()`

使用C++11特性修改后的完美转发
```cpp
template <typename T>
void function(T&& t) {  // 仅仅只是用了右值引用作为参数
    otherdef(forward<T>(t));
}
```

总的来说，在定义模板函数时，我们采用右值引用的语法格式定义参数类型，由此该函数既可以接收外界传入的左值，也可以接收右值  
其次，还需要使用 C++11 标准库提供的 forword() 模板函数修饰被调用函数中需要维持左、右值属性的参数。由此即可轻松实现函数模板中参数的完美转发。   

# 6. lambda 表达式 匿名函数

lambda 源自希腊字母表中第 11 位的 λ，在计算机科学领域，它则是被用来表示一种匿名函数。  
所谓匿名函数，简单地理解就是没有名称的函数，又常被称为 lambda 函数或者 lambda 表达式。  


继 Python、Java、C#、PHP 等众多高级编程语言都支持 lambda 匿名函数后，C++11 标准终于引入了 lambda

## 6.1. 定义方法


```cpp
// Lambda表达式完整的声明格式如下

[capture list] (params list) mutable noexcept/throw() return type { function body }
/* 
    capture list    捕获外部变量列表
    params list     形参列表  如果不需要参数而且不使用 mutable和异常抛出 则括号也可以省略
    mutable         指示符   用来说用是否可以修改捕获的变量  可以省略
    noexcept/throw()异常设定 可以省略
    return type     返回类型
    function body   函数体
 */
```
各部分详细说明
1. `[外部变量方位方式说明符]`
   * `[]` 用于向编译器表明这是一个 lambda 表达式 不能被省略
   * 在方括号的内部 可以注明当前 lambda 函数的函数体可以使用哪些 `外部变量`
   * 这里的外部变量指的是 lambda 表达式作用域的所有局部变量
2. `(参数)` 可省略
   * 和普通函数一样可以接受外部传参
   * 如果不使用 3 和 4 则在不需要传参数的情况下可以省略括号
3. mutable 可省略
   * 对于以值传递的方式引入的外部变量, lambda 也不能修改他们的值 , 相当于 const 常量
   * 如果想修改就使用 mutable (但是如果是值传递的话 仍然是拷贝的部分 不会真正修改外部变量)
4. noexcept/throw() 可省略
   * 和普通函数一样可以抛出任何类型的异常 , 使用 noexcept 表示程序不会抛出任何异常
   * 使用 throw() 可以指定 lambda 函数内部可以抛出的异常类型
   * 如果函数标注了 noexcept 或者 throw() 中没有对应的异常类型, 即非指定类型异常, 这些异常不能用 try-catch捕获
5. -> 返回值类型 可省略
   * 如果 lambda 函数体中有且只有 一个 return 语句, 或者返回 void 则编译器可以自己推出返回值类型 省略 `->类型`
6. 函数体
   * 除了传入的参数 , 全部变量也可以使用, 还可以使用 `指定的外部变量`


**`[外部变量]的具体使用方法`**

如表所示
| 格式写法        | 功能                                             |
| --------------- | ------------------------------------------------ |
| `[]`            | 函数中不导入任何外部变量                         |
| `[=]`           | 只有一个=号,值传递的方式导入所有外部变量         |
| `[val1,val2]`   | 值传递导入指定的外部变量,没有先后次序            |
| `[&]`           | 只有一个&号,引用方式导入所有外部变量             |
| `[&val1,&val2]` | 引用传递的方式导入指定的外部变量                 |
| `[=,&val1]`     | 区别对待,指定的变量使用引用传递,剩余的所有值传递 |
| `[this]`        | 值传递方法传入当前的 this 指针                   |

注意，单个外部变量不允许以相同的传递方式导入多次。例如 `[=，val1]` 中，val1 先后被以值传递的方式导入了 2 次，这是非法的。  

匿名函数的定义和使用
```cpp
// 最简单的函数
[]{}

```

