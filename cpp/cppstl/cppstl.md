# 1. C++ 标准环境

STL是Standard Template Library 的简称  

## 1.1. C++ 编译环境的构成

一个完整的C++环境由 库和编译模块构成

库中包括
* C++标准库     即 STL, 不带`.h`的头文件, 在std命名空间
* C语言兼容库   头文件带`.h` , 不是C++标准的内容,但是C++编译器提供商一般都会提供C的兼容库, 即编译器内置的C库
* 编译器扩展库  每个编译器独有的拓展库 G++和VC++的拓展库就不同 如 `stdafx.h`
编译模块
* C++ 标准语法模块    对C++标准语法的支持
* C++ 扩展语法模块    每个编译器独有的扩展语法的支持 

## 1.2. C++ 标准库

C++ 标准库并不止 容器库



1. C标准库            首字母带C的C语言库
2. 流库               `iostream iomanip ios sstream fstream` 以及C语言兼容的 `cstdio cwchar`
3. 数值操作库         `complex valarray numeric cmath cstdlib`
4. 诊断功能           `stdexcept cassert cerrno`
5. 通用工具
6. 国际化
7. 语言支持功能       `cstddef limits climits cfloat cstdlib new typeinfo exception cstdarg csetjmp csginal`

在此之外的则是 STL容器库

包括字符串 以及带有关联的 `算法-迭代器-容器` 

8. 字符串             `string cctype cwctpye cstring cwchar cstdlib`
9. 容器
10. 迭代器            `iterator`
11. 算法              `algorithm cstdlib ciso646`


根据库中的函数是否属于类 , 标准库还有另一种分类方法
* 标准函数库      通用的独立的,不属于类的函数组成的库,函数基本继承于C语言
* 面向对象库      类及其相关函数的集合

## 1.3. cppreference.com  的C++标准库

从网站上拷贝的最标准的库

1. Language Support

```cpp
#include <cstddef>
#include <cstdlib>
#include <version>
#include <limits>
#include <climits>
#include <cfloat>
#include <cstdint>
#include <new>
#include <typeinfo>
#include <source_location>
#include <exception>
#include <initializer_list>
#include <compare>
#include <coroutine>
#include <csignal>
#include <csetjmp>
#include <cstdarg>
```

2. Concepts
`#include <concepts>`  

3. Diagnostics
```cpp
#include <stdexcept>
#include <cassert>
#include <cerrno>
#include <system_error>
```

4. General utilities

```cpp
#include <utility>
#include <memory>
#include <memory_resource>
#include <scoped_allocator>    
#include <bitset>
#include <tuple>
#include <optional>
#include <any>
#include <variant>
#include <type_traits>
#include <ratio>
#include <chrono>
#include <typeindex>
#include <functional>
#include <ctime>
```

5. Strings

```cpp
#include <string>
#include <string_view>
#include <cstring>
#include <charconv>
#include <format>
#include <cctype>
#include <cwctype>
#include <cwchar>
#include <cuchar>
```
6. Localization

```cpp
#include <locale>
#include <codecvt>
#include <clocale>
```
7.  Containers

```cpp
#include <span>
#include <array>
#include <vector>
#include <deque>
#include <forward_list>
#include <list>
#include <map>
#include <set>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <stack>
```

8. Iterators
```cpp
#include <iterator>
```

9. Ranges
```cpp
#include <ranges>
```

10. Algorithms

```cpp
#include <algorithm>
#include <execution>
```

11. Numerics

```cpp
#include <complex>
#include <random>
#include <valarray>
#include <numeric>
#include <bit>
#include <numbers>
#include <cfenv>
#include <cmath>
```

12. Input/Output

```cpp
#include <iosfwd>
#include <ios>
#include <iomanip>
#include <streambuf>
#include <istream>
#include <ostream>
#include <iostream>
#include <sstream>
#include <fstream>
#include <syncstream>
#include <cstdio>
#include <cinttypes>
#include <strstream>
```

13. Regular expressions

```cpp
#include <regex>
```

14. Filesystem support

```cpp
#include <filesystem>
```

15. Thread support

```cpp
#include <thread>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <condition_variable>
#include <future>
#include <stop_token>
#include <semaphore>
#include <latch>
#include <barrier>
```
16. C compatibility
```cpp
#include <ciso646>
#include <cstdalign>
#include <cstdbool>
#include <ccomplex>
#include <ctgmath>
```
 


# 3. 工程要点
## 3.1. 後藤先生からのアドバイス

1. 功能函数和用户界面应该分开， 例如标准信息输出不应该出现在用户处理函数中
2. 凡是涉及到分配内存的功能都应该检查是否成功
3. 虽然图像被视为一维处理，但是从可读性的角度来看，应该认真地以二维阵列的形式在双循环中处理。
4. 注意要经常使用 unsign 
5. 在多重循环中使用条件分歧会对速度产生不利影响
6. 在内存限制不严重的情况下， 不要用float类型

## 3.2. 关于 new 的检查

通常来说，new 使用 try catch 来捕捉异常  
抛出的异常为 `std::bad_alloc` 类  
```cpp
try {
    while (true) {
        new int[100000000ul];   // throwing overload
  }
}catch (const std::bad_alloc& e) {
    std::cout << e.what() << '\n';
}

```

但是也有阻止抛出异常的方法  
`std::nothrow` 就是专门用来处理new异常的类, new 操作失败时返回的指针为 nullptr 而不是抛出异常  
```cpp
int* p = new(std::nothrow) int[100000000ul]; // non-throwing overload
if (p == nullptr) {
   std::cout << "Allocation returned nullptr\n";
   break;
}
```

# 5. 容器 Containers

```cpp
#include <span>
#include <array>
#include <vector>
#include <deque>
#include <forward_list>
#include <list>
#include <map>
#include <set>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <stack>
```

## 5.1. 容器种类  
STL 提供有 3 类标准容器，分别是序列容器、排序容器和哈希容器  其中后两类容器有时也统称为关联容器  
* 序列容器，是因为元素在容
* 器中的位置同元素的值无关，即容器不是排序的。将元素插入容器时，指定在什么位置，元素就会位于什么位置
* 排序容器中的元素默认是由小到大排序好的，即便是插入元素，元素也会插入到适当位置。所以关联容器在查找时具有非常好的性能。
* 和排序容器不同，哈希容器中的元素是未排序的，元素的位置由哈希函数确定
| 容器种类 | 功能                                                                                                                                                       |
| -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 序列容器 | 主要包括 vector 向量容器、list 列表容器以及 deque 双端队列容器。                                                                                           |
| 排序容器 | 包括 set 集合容器、multiset多重集合容器、map映射容器以及 multimap 多重映射容器。                                                                           |
| 哈希容器 | C++ 11 新加入 4 种关联式容器，分别是 unordered_set 哈希集合、unordered_multiset 哈希多重集合、unordered_map 哈希映射以及 unordered_multimap 哈希多重映射。 |

### 5.1.1. 序列式容器  


* `array<T,N>`（数组容器）：表示可以存储 N 个 T 类型的元素，是 C++ 本身提供的一种容器。此类容器一旦建立，其长度就是固定不变的，这意味着不能增加或删除元素，只能改变某个元素的值；
* `vector<T>`（向量容器）：用来存放 T 类型的元素，是一个长度可变的序列容器，即在存储空间不足时，会自动申请更多的内存。使用此容器，在尾部增加或删除元素的效率最高（时间复杂度为 O(1) 常数阶），在其它位置插入或删除元素效率较差（时间复杂度为 O(n) 线性阶，其中 n 为容器中元素的个数）；
* `deque<T>`（双端队列容器）：和 vector 非常相似，区别在于使用该容器不仅尾部插入和删除元素高效，在头部插入或删除元素也同样高效，时间复杂度都是 O(1) 常数阶，但是在容器中某一位置处插入或删除元素，时间复杂度为 O(n) 线性阶；
* `list<T>`（链表容器）：是一个长度可变的、由 T 类型元素组成的序列，它以**双向链表**的形式组织元素，在这个序列的任何地方都可以高效地增加或删除元素（时间复杂度都为常数阶 O(1)），但访问容器中任意元素的速度要比前三种容器慢，这是因为 `list<T>` 必须从第一个元素或最后一个元素开始访问，需要沿着链表移动，直到到达想要的元素。
* `forward_list<T>`（正向链表容器）：和 list 容器非常类似，只不过它以**单链表**的形式组织元素，它内部的元素只能从第一个元素开始访问，是一类比链表容器快、更节省内存的容器。

    `stack<T> `和 `queue<T>` 本质上也属于序列容器，只不过它们都是在 deque 容器的基础上改头换面而成，通常更习惯称它们为容器适配器


### 5.1.2. 关联式容器

关联式容器在存储元素值的同时，还会为各元素额外再配备一个值（又称为“键”，其本质也是一个 C++ 基础数据类型或自定义类型的元素）  
它的功能是在使用关联式容器的过程中，如果已知目标元素的键的值，则直接通过该键就可以找到目标元素，而无需再通过遍历整个容器的方式。   

也就是说，使用关联式容器存储的元素，都是一个一个的“键值对”·`（ <key,value> ）`  
除此之外，序列式容器中存储的元素默认都是未经过排序的，而使用关联式容器存储的元素，默认会根据各元素的键值的大小做升序排序。   

* map 	    定义在`<map>` 头文件中，使用该容器存储的数据，其各个元素的键必须是唯一的（即不能重复），该容器会根据各元素键的大小，默认进行升序排序`（调用 std::less<T>）。`
* set   	定义在`<set>` 头文件中，使用该容器存储的数据，各个元素键和值完全相同，且各个元素的值不能重复（保证了各元素键的唯一性）。该容器会自动根据各个元素的键（其实也就是元素值）的大小进行升序* 排序`（调用 std::less<T>）。`
* multimap 	定义在 `<map>` 头文件中，和 map 容器唯一的不同在于，multimap 容器中存储元素的键可以重复。
* multiset 	定义在` <set>` 头文件中，和 set 容器唯一的不同在于，multiset 容器中存储元素的值可以重复（一旦值重复，则意味着键也是重复的）。

C++ 11 还新增了 4 种哈希容器，即 unordered_map、unordered_multimap 以及 unordered_set、unordered_multiset  
但哈希容器底层采用的是哈希表，而不是红黑树

### 5.1.3. 自定义关联式容器的排序规则

1. 使用函数对象自定义排序规则
2. 重载关系运算符实现自定义排序


```cpp
//定义函数对象类
class cmp {
public:
    //重载 () 运算符
    bool operator ()(const string &a,const string &b) {
        //按照字符串的长度，做升序排序(即存储的字符串从短到长)
        return  (a.length() < b.length());
    }
};

// 用struct 定义函数对象类
struct cmp {
  //重载 () 运算符
  bool operator ()(const string &a, const string &b) {
      //按照字符串的长度，做升序排序(即存储的字符串从短到长)
      return  (a.length() < b.length());
  }
};

// 定义函数对象模板类
template <typename T>
class cmp {
public:
    //重载 () 运算符
    bool operator ()(const T &a, const T &b) {
        //按照值的大小，做升序排序
        return  a < b;
    }
};

// 创建 set 容器，并使用自定义的 cmp 排序规则
std::set<string, cmp>myset;

```
  
重载关系运算符  
在 STL 标准库中，本就包含几个可供关联式容器使用的排序规则  
| 排序规则                | 功能                                                               |
| ----------------------- | ------------------------------------------------------------------ |
| `std::less<T>`          | 底层采用 < 运算符实现升序排序，各关联式容器默认采用的排序规则。    |
| `std::greater<T>`       | 底层采用 > 运算符实现降序排序，同样适用于各个关联式容器。          |
| `std::less_equal<T>`    | 底层采用 <= 运算符实现升序排序，多用于 multimap 和 multiset 容器。 |
| `std::greater_equal<T>` | 底层采用 >= 运算符实现降序排序，多用于 multimap 和 multiset 容器。 |

1. 当关联式容器中存储的数据类型为自定义的结构体变量或者类对象时，  
   通过对现有排序规则中所用的关系运算符进行重载，也能实现自定义排序规则的目的  
2. 当关联式容器中存储的元素类型为结构体指针变量或者类的指针对象时
   只能使用函数对象的方式自定义排序规则，此方法不再适用

```cpp
//自定义类
class myString {
public:
    //定义构造函数，向 myset 容器中添加元素时会用到
    myString(string tempStr) :str(tempStr) {};
    //获取 str 私有对象，由于会被私有对象调用，因此该成员方法也必须为 const 类型
private:
    string str;
};

//重载 < 运算符，参数必须都为 const 类型
bool operator <(const myString &stra, const myString & strb) {
    //以字符串的长度为标准比较大小
    return stra.getStr().length() < strb.getStr().length();
}

//创建空 set 容器，仍使用默认的 less<T> 排序规则
std::set<myString>myset;
```
### 5.1.4. 无序关联式容器

无序容器是 C++ 11 标准才正式引入到 STL 标准库中的，这意味着如果要使用该类容器，则必须选择支持 C++ 11 标准的编译器。  
它们常被称为 “无序容器”、“哈希容器”或者“无序关联容器”。

unordered_map、unordered_multimap 以及 unordered_set、unordered_multiset   

* 总的来说，实际场景中如果涉及大量遍历容器的操作，建议首选关联式容器；
* 反之，如果更多的操作是通过键获取对应的值，则应首选无序容器。


1. 二者的头文件不同 模板定义不同
```cpp
//map
#include < map >
//unordered_map
#include < unordered_map >

template < class Key,                        //键值对中键的类型
            class T,                          //键值对中值的类型
            class Hash = hash<Key>,           //容器内部存储键值对所用的哈希函数
            class Pred = equal_to<Key>,       //判断各个键值对键相同的规则
            class Alloc = allocator< pair<const Key,T> >  // 指定分配器对象的类型
            > class unordered_map;
/* 
1. Hash = hash<Key> 指明容器在存储各个键值对时要使用的哈希函数 
    STL 标准库提供的 hash<key> 哈希函数 只适用于基本数据类型（包括 string 类型），而不适用于自定义的结构体或者类。
2. Pred = equal_to<Key>  容器中存储的各个键值对的键是不能相等的，而判断是否相等的规则，就由此参数指定。
    默认情况下，使用 STL 标准库中提供的 equal_to<key> 规则，该规则仅支持可直接用 == 运算符做比较的数据类型。
3. 当无序容器中存储键值对的键为自定义类型时，默认的哈希函数 hash 以及比较函数 equal_to 将不再适用，
    只能自己设计适用该类型的哈希函数和比较函数，并显式传递给 Hash 参数和 Pred 参数。
*/
```

2. 内部实现机理不同

      map
      * map内部实现了一个红黑树（红黑树是非严格平衡二叉搜索树，而AVL是严格平衡二叉搜索树）
      * 红黑树具有自动排序的功能，因此map内部的所有元素都是有序的，红黑树的每一个节点都代表着map的一个元素
      * 因此，对于map进行的查找，删除，添加等一系列的操作都相当于是对红黑树进行的操作

      unordered_map
      * unordered_map内部实现了一个哈希表（也叫散列表，通过把关键码值映射到Hash表中一个位置来访问记录，查找的时间复杂度可达到O(1)
      * 其在海量数据处理中有着广泛应用）
      * 因此，其元素的排列顺序是无序的
  
3. 优缺点以及适用处
  
**map的优点**:
* 有序性，这是map结构最大的优点，其元素的有序性在很多应用中都会简化很多的操作
* 红黑树，内部实现一个红黑书使得map的很多操作在lg(n)的时间复杂度下就可以实现，因此效率非常的高
**map的缺点**:
* 空间占用率高，因为map内部实现了红黑树，虽然提高了运行效率，但是因为每一个节点都需要额外保存父节点、孩子节点和红/黑性质，使得每一个节点都占用大量的空间

**map适用处**:
* 对于那些有顺序要求的问题，用map会更高效一些


**unordered_map**:  
优点:
* 因为内部实现了哈希表，因此其查找速度非常的快
缺点: 
* 哈希表的建立比较耗费时间
适用处:
* 对于查找问题，unordered_map会更加高效一些
* 遇到查找问题，常会考虑一下用unordered_map


**总结**:  
* 内存占有率的问题就转化成红黑树 VS hash表 , 还是unorder_map占用的内存要高
* 但是unordered_map执行效率要比map高很多
* 对于unordered_map或unordered_set容器，其遍历顺序与创建该容器时输入的顺序不一定相同，因为遍历是按照哈希表从前往后依次遍历的
  

## 5.2. 迭代器

泛型思维发展的必然结果，类似中介的装置，它除了要具有对容器进行遍历读写数据的能力之外，还要能对外隐藏容器的内部差异，从而以统一的界面向算法传送数据。  

### 5.2.1. 迭代器的种类

STL 标准库为每一种标准容器定义了一种迭代器类型，这意味着，不同容器的迭代器也不同，其功能强弱也有所不同。  

**容器的迭代器的功能强弱** ， 决定了该容器是否支持 STL 中的某种算法。  

针对容器的迭代器有：前向迭代器、双向迭代器、随机访问迭代器  
针对输入输出流的迭代器有：输入迭代器、输出迭代器  

1. 前向迭代器（forward iterator）
    假设 p 是一个前向迭代器，则 p 支持 ++p，p++，*p 操作，还可以被复制或赋值，可以用 == 和 != 运算符进行比较。此外，两个正向迭代器可以互相赋值。

2. 双向迭代器（bidirectional iterator）
    双向迭代器具有正向迭代器的全部功能，除此之外，假设 p 是一个双向迭代器，则还可以进行 --p 或者 p-- 操作（即一次向后移动一个位置）。
    **双向迭代器不支持用“< >”进行比较**  

3. 随机访问迭代器（random access iterator）
    随机访问迭代器具有双向迭代器的全部功能。除此之外，假设 p 是一个随机访问迭代器，i 是一个整型变量或常量，则 p 还支持以下操作：
        p+=i：使得 p 往后移动 i 个元素。
        p-=i：使得 p 往前移动 i 个元素。
        p+i：返回 p 后面第 i 个元素的迭代器。
        p-i：返回 p 前面第 i 个元素的迭代器。
        p[i]：返回 p 后面第 i 个元素的引用。
    此外，两个随机访问迭代器 p1、p2 还可以用 <、>、<=、>= 运算符进行比较。另外，表达式 p2-p1 也是有定义的，其返回值表示 p2 所指向元素和 p1 所指向元素的序号之差（也可以说是 p2 和 p1 之间的元素个数减一）
  

不同容器的迭代器
| 容器                               | 对应的迭代器类型 |
| ---------------------------------- | ---------------- |
| array                              | 随机访问迭代器   |
| vector                             | 随机访问迭代器   |
| deque                              | 随机访问迭代器   |
| list                               | 双向迭代器       |
| set / multiset                     | 双向迭代器       |
| map / multimap                     | 双向迭代器       |
| forward_list                       | 前向迭代器       |
| unordered_map / unordered_multimap | 前向迭代器       |
| unordered_set / unordered_multiset | 前向迭代器       |
| stack                              | 不支持迭代器     |
| queue                              | 不支持迭代器     |

### 5.2.2. 定义中迭代器的种类  

迭代器的 4 种定义方式
| 迭代器定义方式 | 具体格式                                    |
| -------------- | ------------------------------------------- |
| 正向迭代器     | 容器类名::iterator  迭代器名;               |
| 常量正向迭代器 | 容器类名::const_iterator  迭代器名;         |
| 反向迭代器     | 容器类名::reverse_iterator  迭代器名;       |
| 常量反向迭代器 | 容器类名::const_reverse_iterator  迭代器名; |

读取它指向的元素，`*迭代器名` 就表示迭代器指向的元素  
* 对正向迭代器进行 ++ 操作时，迭代器会指向容器中的后一个元素；
* 而对反向迭代器进行 ++ 操作时，迭代器会指向容器中的前一个元素。
**注意，以上 4 种定义迭代器的方式**，并不是每个容器都适用。  
有一部分容器同时支持以上 4 种方式，比如 `array、deque、vector`；  
而有些容器只支持其中部分的定义方式，例如 `forward_list` 容器只支持定义正向迭代器，不支持定义反向迭代器。  

### 5.2.3. 和迭代器有关的成员函数  

| 函数成员  | 函数功能                                                                          |
| --------- | --------------------------------------------------------------------------------- |
| begin()   | 返回指向容器中第一个元素的迭代器。                                                |
| end()     | 返回指向容器最后一个元素所在位置**后一个位置**的迭代器，通常和 begin() 结合使用。 |
| rbegin()  | 返回指向最后一个元素的迭代器。                                                    |
| rend()    | 返回指向第一个元素所在位置**前一个位置**的迭代器。                                |
| cbegin()  | 和 begin() 功能相同，只不过在其基础上，增加了 const 属性，不能用于修改元素。      |
| cend()    | 和 end() 功能相同，只不过在其基础上，增加了 const 属性，不能用于修改元素。        |
| crbegin() | 和 rbegin() 功能相同，只不过在其基础上，增加了 const 属性，不能用于修改元素。     |
| crend()   | 和 rend() 功能相同，只不过在其基础上，增加了 const 属性，不能用于修改元素。       |

以上函数在实际使用时，其返回值类型都可以使用 auto 关键字代替，编译器可以自行判断出该迭代器的类型。

## 5.3. 容器适配器

```cpp
// 假设一个代码模块 A，它的构成如下所示：
class A{
public:
  void f1(){}
  void f2(){}
  void f3(){}
  void f4(){}
};

// 现在我们需要设计一个模板 B，但发现，其实只需要组合一下模块 A 中的 f1()、f2()、f3()，就可以实现模板 B 需要的功能。
// 其中 f1() 单独使用即可，而 f2() 和 f3() 需要组合起来使用
class B{
private:
    A * a;
public:
    void g1(){
        a->f1();
    }
    void g2(){
        a->f2();
        a->f3();
    }
};

// 模板 B 将不适合直接拿来用的模板 A 变得适用了，因此我们可以将模板 B 称为 B 适配器。
```

* 就是将不适用的序列式容器（包括 vector、deque 和 list）变得适用。
* 容器适配器的底层实现和模板 A、B 的关系是完全相同的，即通过封装某个序列式容器，**并重新组合该容器中包含的成员函数**，使其满足某些特定场景的需要。
* 容器适配器本质上还是容器，只不过此容器模板类的实现，利用了大量其它基础容器模板类中已经写好的成员函数。当然，如果必要的话，容器适配器中也可以自创新的成员函数。
* 容器适配器没有迭代器, 访问元素只能遍历容器


STL 提供了 3 种容器适配器, 各适配器所使用的**默认基础容器**以及 **可供用户选择的基础容器** 

1. stack  默认使用的基础容器 deque
      * empty()
   * size()
   * top()                    返回一个栈顶元素的引用，类型为 T&。如果栈为空，程序会报错。
   * pop()                    弹出栈顶元素。
   * push(const T& val)       先复制 val，再将 val 副本压入栈顶。这是通过调用底层容器的 push_back() 函数完成的
   * push(T&& obj)            以移动元素的方式将其压入栈顶。这是通过调用底层容器的有右值引用参数的 push_back() 函数完成的。
   * emplace(arg...)           可以是一个参数，也可以是多个参数，但它们都只用于构造一个对象，并在栈顶直接生成该对象，作为新的栈顶元素。
   * swap(`stack<T> & other_stack`)
2. queue  默认使用的基础容器 deque
   * empty()
   * size()
   * front()
   * back()
   * push()
   * pop()
3. priority_queue  默认使用的基础容器 vector
   * empty()
   * size()
   * front()
   * push_back()
   * pop_front()



### 5.3.1. pair 例外

考虑到“键值对”并不是普通类型数据，C++ STL 标准库提供了 pair 类模板  
专门用来将 2 个普通元素 first 和 second  创建成一个新元素`<first, second>`    

pair 对象重载了 <、<=、>、>=、==、!= 这 6 的运算符，其运算规则是：对于进行比较的 2 个 pair 对象，先比较 pair.first 元素的大小，如果相等则继续比较 pair.second 元素的大小。  

* 注意: pair 类模板定义在 `<utility>` 头文件中

```cpp

// 创建一个 pair
std::make_pair("C语言教程",10)
// 在初始化 map 的时候可以这么使用
std::map<std::string, int>myMap{std::make_pair("C语言教程",10),std::make_pair("STL教程",20)};

```

### 5.3.2. stack 

stack  默认使用的基础容器 deque
   * empty()
   * size()
   * top()                    返回一个栈顶元素的引用，类型为 T&。如果栈为空，程序会报错。
   * pop()                    弹出栈顶元素。
   * push(const T& val)       先复制 val，再将 val 副本压入栈顶。这是通过调用底层容器的 push_back() 函数完成的
   * push(T&& obj)            以移动元素的方式将其压入栈顶。这是通过调用底层容器的有右值引用参数的 push_back() 函数完成的。
   * emplace(arg...)           可以是一个参数，也可以是多个参数，但它们都只用于构造一个对象，并在栈顶直接生成该对象，作为新的栈顶元素。
   * swap(`stack<T> & other_stack`)

和其他序列容器相比，stack 是一类存储机制简单、提供成员函数较少的容器  
stack 栈适配器是一种单端开口的容器  
实际上该容器模拟的就是栈存储结构，即无论是向里存数据还是从中取数据，都只能从这一个开口实现操作。  

```cpp
// stack 有自己的头文件
#include <stack>

// T 为存储元素的类型，Container 表示底层容器的类型
stack<T,Container=deque<T>>

// 创建一个不包含任何元素的 stack 适配器，并采用默认的 deque 基础容器： 
std::stack<int> values;

// 通过指定第二个模板类型参数，我们可以使用出 deque 容器外的其它序列式容器，
// 只要该容器支持 empty()、size()、back()、push_back()、pop_back() 这 5 个成员函数即可
// 序列式容器中同时包含这 5 个成员函数的，只有 vector、deque 和 list 因此，stack 适配器的基础容器可以是它们 3 个中任何一个
std::stack<std::int, std::list<int>> values;


// 可以用一个和 stack 底层使用的基础容器类型相同的基础容器来初始化 stack 适配器
std::list<int> values {1, 2, 3};
// stack 第 2 个模板参数必须显式指定为 list<int>（必须为 int 类型，和存储类型保持一致） 否则无法用 lsit 容器的内容来初始化该 stack 适配器
std::stack<int,std::list<int>> my_stack (values);
// 初始化后的 my_stack 适配器中，栈顶元素为 3，而不是 1


// 还可以用一个 stack 适配器来初始化另一个 stack 适配器，只要它们存储的元素类型以及底层采用的基础容器类型相同即可
std::stack<int, std::list<int>> my_stack2=my_stack;
std::stack<int, std::list<int>> my_stack(my_stack1);
```

### 5.3.3. queue

和 stack 栈容器适配器不同，queue 容器适配器有 2 个开口，其中一个开口专门用来输入数据，另一个专门用来输出数据  
最先进入 queue 的元素，也可以最先从 queue 中出来，  
即用此容器适配器存储数据具有“先进先出（简称 "FIFO" ）”的特点，因此 queue 又称为队列适配器。  

```cpp

// queue 也有自己的头文件
#include <queue>


// 可以指定底层采用的基础容器类型
// queue 容器适配器底层容器可以选择 deque (默认) 和 list。
std::queue<int, std::list<int>> values;

// 用基础容器来初始化 queue 容器适配器 
std::deque<int> values{1,2,3};
std::queue<int> my_queue(values);


// 成员函数
empty();
size();
front();
back();

push()
emplace()
pop()
swap()

while (!my_queue.empty())
{
    cout << my_queue.front() << endl;
    //访问过的元素出队列
    my_queue.pop();
}
```

### 5.3.4. priority_queue

和queue有几点不同
1. 只能访问 priority_queue 中位于队头的元素
2. 只能“从一端进（称为队尾），从另一端出（称为队头）
3. 不是 “First in,First out”（先入先出）原则，而是“First in，Largest out”原则

```cpp
// T：指定存储元素的具体类型
// Container：指定 priority_queue 底层使用的基础容器，vector(默认) 和 deque 可以使用
// 
template <typename T,
      typename Container=std::vector<T>,
      typename Compare=std::less<T> >
class priority_queue{
  //......
}

// 初始化的数组或容器中的数据不需要有序，priority_queue 会自动对它们进行排序。
std::priority_queue<int> values;

empty()
size()
top()
push()
emplace()
pop()
swap()

```


## 5.4. array 升级的数组  

rray 容器是 C++ 11 标准中新增的序列容器，它就是在 C++ 普通数组的基础上，添加了一些成员函数和全局函数。  
在使用上，它比普通数组更安全，且效率并没有因此变差。  

```cpp
#include <array>
//在 array<T,N> 类模板中，T 用于指明容器中的存储的具体数据类型，N 用于指明容器的大小，
//需要注意的是，这里的 N 必须是常量，不能用变量表示, 和普通数组一样。
std::array<double, 10> values;
//这种方式创建的容器中，各个元素的值是不确定的（array 容器不会做默认初始化操作）。

//将所有的元素初始化为 0 或者和默认元素类型等效的值：
std::array<double, 10> values {};

//像创建常规数组那样对元素进行初始化：
std::array<double, 10> values {0.5,1.0,1.5,,2.0};
//剩余的元素都会被初始化为 0.0

```
### 5.4.1. 安全的访问  

1. begin() end()
    C++ 11 标准库还新增加了 begin() 和 end() 这 2 个函数，和 array 容器包含的 begin() 和 end() 成员函数不同的是，标准库提供的这 2 个函数的操作对象，既可以是容器，还可以是普通数组。  
    如果操作对象是普通数组，则 begin() 函数返回的是指向数组第一个元素的指针，同样 end() 返回指向数组中最后一个元素之后一个位置的指针（注意不是最后一个元素）。
```cpp
    auto first = std::begin(values);
    auto last = std::end (values);
```
2. get()  `<array>` 头文件中重载了 get() 全局函数，该重载函数的功能是访问容器中指定的元素，并返回该元素的**引用**。
3. at() 返回容器中 n 位置处元素的引用，该函数自动检查 n 是否在有效的范围内，如果不是则抛出 out_of_range 异常。

```cpp
#include <iostream>
//需要引入 array 头文件
#include <array>
int main()
{
    std::array<int, 4> values{};
    //初始化 values 容器为 {0,1,2,3}, 用at 访问并更改值
    for (int i = 0; i < values.size(); i++) {
        values.at(i) = i;
    }

    //使用 get() 重载函数输出指定位置元素
    cout << get<3>(values) << endl;
    //如果容器不为空，则输出容器中所有的元素
    if (!values.empty()) {
        for (auto val = values.begin(); val < values.end(); val++) {
            cout << *val << " ";
        }
    }
}


```
### 5.4.2. array使用方法总结  

访问array容器中单个元素
1. 可以通过容器名[]的方式直接访问和使用容器中的元素，这和 C++ 标准数组访问元素的方式相同
2. `get<n>` 模板函数，它是一个辅助函数，能够获取到容器的第 n 个元素 , 只能访问模板参数指定的元素，编译器在编译时会对它进行检查
3. array 容器提供了 data() 成员函数，通过调用该函数可以得到指向容器首个元素的指针。
```cpp

values[4] = values[3] + 2.O*values[1];
// 使用 at 越界时  会抛出 std::out_of_range 异常
values.at (4) = values.at(3) + 2.O*values.at(1);

cout << get<3>(words) << endl;

cout << *( words.data()+1);
```

访问array容器中多个元素

1. size() 函数能够返回容器中元素的个数（函数返回值为 size_t 类型）
2. empty() 成员函数，即可知道容器中有没有元素（如果容器中没有元素，此函数返回 true）

```cpp
if(values.empty())
    std::cout << "The container has no elements.\n";
else
    std::cout << "The container has "<< values.size()<<"elements.\n";
```

## 5.5. vector 向量容器

它和 array 容器非常类似，都可以看做是对 C++ 普通数组的“升级版”。不同之处在于，array 实现的是静态数组（容量固定的数组），而 vector 实现的是一个动态数组，即可以进行元素的插入和删除，在此过程中，vector 会动态调整所占用的内存空间，整个过程无需人工干预  

vector 常被称为向量容器，因为`该容器擅长在尾部插入或删除元素`，`时间复杂度为O(1)`；而对于在容器头部或者中部插入或删除元素，则花费时间要长一些（移动元素需要耗费时间），时间复杂度为线性阶`O(n)`。  


vector的源码实现
```cpp
/*
vector就是使用 3 个迭代器（可以理解成指针）来表示的
_Myfirst 指向的是 vector 容器对象的起始字节位置
_Mylast 指向当前最后一个元素的末尾字节
_Myend 指向整个 vector 容器所占用内存空间的末尾字节。

*/
class vector{
    ...
protected:
    pointer _Myfirst;
    pointer _Mylast;
    pointer _Myend;
};
```
当 vector 的大小和容量相等（size==capacity）也就是满载时，如果再向其添加元素，那么 vector 就需要扩容。vector 容器扩容的过程需要经历以下 3 步：
1. 完全弃用现有的内存空间，重新申请更大的内存空间；
2. 将旧内存空间中的数据，按原有顺序移动到新的内存空间中；
3. 最后将旧的内存空间释放。

vector 容器扩容时，不同的编译器申请更多内存空间的量是不同的。以 VS 为例，它会扩容现有容器容量的 50%。  


### 5.5.1. 创建

```cpp
// 最基础的定义
std::vector<double> values;

// 使用 reserve() 来重新分配空间,增加容量
// 重新分配空间后可能导致之前定义的迭代器失效
values.reserve(20);

// 用类似数组的方法定义初始值
std::vector<int> primes {2, 3, 5, 7, 11, 13, 17, 19};
// 定义初始的元素个数
std::vector<double> values(20);

// 定义初始元素个数以及统一的初始值
std::vector<double> values(20, 1.0);

// 和 array 不同, 定义 vector 的时候, 元素个数和初始值都可以是变量
std::vector<double> values(num, value);


// vector的拷贝构造
//可以用一对指针或者迭代器来指定拷贝初始值的范围
int array[]={1,2,3};
std::vector<int>values(array, array+2);//values 将保存{1,2}
std::vector<int>value1{1,2,3,4,5};
std::vector<int>value2(std::begin(value1),std::begin(value1)+3);//value2保存{1,2,3}

```

### 5.5.2. queue
和 stack 栈容器适配器不同，queue 容器适配器有 2 个开口，其中一个开口专门用来输入数据，另一个专门用来输出数据

### 5.5.3. 访问修改元素

1. 下标访问 (有越界可能)
2. at() 访问(抛出 std::out_of_range 异常)
3. front() back() 访问首尾元素 ( 返回引用, 可以修改值 )
4. data 访问
```cpp
// 使用下标进行访问和修改
values[0] = values[1] + values[2];

// at() 方法访问
values.at(0) = values.at(1);

//修改首尾元素 front() 和 back() 函数 ,  
d.front() = 10;
d.back() = 20;

//输出容器中第 3 个元素的值
cout << *(values.data() + 2) << endl;
```

### 5.5.4. 添加元素

1. push_back() 和 emplace_back() 插入尾部
   * push_back() 向容器尾部添加元素时，首先会创建这个元素，然后再将这个元素拷贝或者移动到容器中（如果是拷贝的话，事后会自行销毁先前创建的这个元素）
   * emplace_back() 在实现时，则是直接在容器尾部创建这个元素，省去了拷贝或移动元素的过程。
   * emplace_back()该函数是 C++ 11 新增加的
   * emplace_back() 的执行效率比 push_back() 高。因此，在实际使用时，建议大家优先选用 emplace_back()。
   * 如果程序要兼顾之前的版本，还是应该使用 push_back()。
2. insert() 和 emplace() 插入指定位置

* insert
    iterator insert(pos,elem)           在迭代器 pos 指定的位置之前插入`一个`新元素elem，并返回表示新插入元素位置的迭代器。  
    iterator insert(pos,n,elem)         在迭代器 pos 指定的位置之前插入 `n 个`元素 elem，并返回表示第一个新插入元素位置的迭代器。  
    iterator insert(pos,first,last)     在迭代器 pos 指定的位置之前，插入`其他容器（不仅限于vector）`中位于` [first,last) `区域的所有元素，并返回表示第一个新插入元素位置的迭代器。  
    iterator insert(pos,initlist)       在迭代器 pos 指定的位置之前，插入初始化列表（用大括号{}括起来的多个元素，中间有逗号隔开）中所有的元素，并返回表示第一个新插入元素位置的迭代器。  
* emplace()
  * emplace() 是 C++ 11 标准新增加的成员函数，用于在 vector 容器指定位置之前插入一个新的元素
  * emplace() 每次只能插入一个元素，而不是多个。
  * 同理, emplace() 在插入元素时，是在容器的指定位置直接构造元素，而不是先单独生成，再将其复制（或移动）到容器中。因此，在实际使用中，推荐大家优先使用 emplace()。
`iterator emplace (const_iterator pos, args...);`  
pos 为指定插入位置的迭代器；`args..`. 表示与新插入元素的`构造函数`相对应的多个参数；该函数会返回表示新插入元素位置的迭代器。  

```cpp
std::vector<int> demo{1,2}; // 1 2 

//第一种格式用法
demo.insert(demo.begin() + 1, 3);//{1,3,2}

//第二种格式用法
demo.insert(demo.end(), 2, 5);//{1,3,2,5,5}

//第三种格式用法
std::array<int,3>test{ 7,8,9 };
demo.insert(demo.end(), test.begin(), test.end());//{1,3,2,5,5,7,8,9}

//第四种格式用法
demo.insert(demo.end(), { 10,11 });//{1,3,2,5,5,7,8,9,10,11}

// emplace
demo1.emplace(demo1.begin(), 3);
```

### 5.5.5. 删除元素

成员函数  
* pop_back()     | 删除 vector 容器中最后一个元素，该容器的大小（size）会减 1，但容量（capacity）不会发生改变。
* erase(pos)     | 删除 vector 容器中 pos 迭代器指定位置处的元素，并返回指向被删除元素`下一个`位置元素的迭代器。该容器的大小（size）会减 1，但容量（capacity）不会发生改变。
* erase(beg,end) | 删除 vector 容器中位于迭代器 [beg,end)指定区域内的所有元素，并返回指向被删除区域`下一个`位置元素的迭代器。该容器的大小（size）会减小，但容量（capacity）不会发生改变。
* clear() |删除 vector 容器中所有的元素，使其变成空的 vector 容器。该函数会改变 vector 的大小（变为 0），但不是改变其容量。


remove() 函数，该函数定义在` <algorithm>` 头文件中。
remove()用于删除容器中指定元素时，常和 erase() 成员函数搭配使用。
* remove() |删除容器中所有`和指定元素值相等的元素`，并返回指向最后一个元素下一个位置的迭代器。值得一提的是，调用该函数不会改变容器的大小和容量。
* 因此无法使用之前的方法遍历容器，而是需要向程序中那样，借助 remove() 返回的迭代器完成正确的遍历。
  * remove() 的实现原理是，在遍历容器中的元素时，一旦遇到目标元素，就做上标记，然后继续遍历，直到找到一个非目标元素，即用此元素将最先做标记的位置覆盖掉，
  * 同时将`此非目标元素所在的位置也做上标记`，等待找到新的非目标元素将其覆盖

```cpp

vector<int>demo{ 1,3,3,4,3,5 };

auto iter = std::remove(demo.begin(), demo.end(), 3); // 1 4 5 4 3 5

// 删除排在后面的无用元素  
demo.erase(iter, demo.end());
```

swap() 函数在头文件 `<algorithm>` 和 `<utility>` 中都有定义，使用时引入其中一个即可   
* swap(beg)、pop_back() |先调用 swap() 函数交换要删除的目标元素和容器最后一个元素的位置，然后使用 pop_back() 删除该目标元素。


### 5.5.6. 迭代器

1. 8大迭代器生成函数, 使用auto接受返回的迭代器很方便, begin和end()配合迭代全部元素
2. 可以使用全局的 begin(vector) 和 end(vector) 获取迭代器  
3. size() 配合下标访问全部元素  

      capacity() 成员函数，可以获得当前容器的容量  
      通过 size() 成员函数，可以获得容器当前的大小。  


* 可以调用` reserve()` 成员函数来增加容器的容量（但并不会改变存储元素的个数）
  * 仅仅只是用来增加容量
* 通过调用成员函数` resize() `可以改变容器的大小，并且该函数也可能会导致 vector 容器容量的增加。
  * 会同时生成对应的初始值, 即调用后容器内就有多少个元素

```cpp
value.reserve(20);
cout << "value 容量是(2)：" << value.capacity() << endl;    // 20 当前的容量
cout << "value 大小是(2)：" << value.size() << endl;        // 15 个元素

//将元素个数改变为 21 个，所以会增加 6 个默认初始化的元素
value.resize(21);

//将元素个数改变为 21 个，新增加的 6 个元素默认值为 99。
value.resize(21,99);

//当需要减小容器的大小时，会移除多余的元素。
value.resize(20);
```


一旦 vector 容器的内存被重新分配，则和 vector 容器中元素相关的所有引用、指针以及迭代器，都可能会失效，最稳妥的方法就是重新生成。   

对于 vector 容器而言，当增加新的元素时，有可能很快完成（即直接存在预留空间中）；也有可能会慢一些（扩容之后再放新元素）。  

若需要保存容器的容量或者大小,尽量使用auto  
vector<T> 对象的容量和大小类型都是 vector<T>::size_type  

```cpp
vector<int>::size_type cap = value.capacity();
vector<int>::size_type size = value.size();

auto cap = value.capacity();
auto size = value.size();
```

### 5.5.7. vector 的容量高级操作

vector 容器的扩容过程是非常耗时的，并且当容器进行扩容后，之前和该容器相关的所有指针、迭代器以及引用都会失效。因此在使用 vector 容器过程中，我们应尽量避免执行不必要的扩容操作。
1. 分配一块大小是当前 vector 容量几倍的新存储空间。注意，多数 STL 版本中的 vector 容器，其容器都会以 2 的倍数增长，也就是说，每次 vector 容器扩容，它们的容量都会提高到之前的 2 倍；
2. 将 vector 容器存储的所有元素，依照原有次序从旧的存储空间复制到新的存储空间中；
3. 析构掉旧存储空间中存储的所有元素；
4. 释放旧的存储空间。

resize(n)
* 强制 vector 容器必须存储 n 个元素，注意，
* 如果 n 比 size() 的返回值小，则容器尾部多出的元素将会被析构（删除）
* 如果 n 比 size() 大，则 vector 会借助默认构造函数创建出更多的默认值元素，并将它们存储到容器末尾
* 如果 n 比 capacity() 的返回值还要大，则 vector 会先扩增，在添加一些默认值元素。

reserve(n) 	强制 vector 容器的容量至少为 n。注意，如果 n 比当前 vector 容器的容量小，则该方法什么也不会做；反之如果 n 比当前 vector 容器的容量大，则 vector 容器就会扩容。

* 避免 vector 容器执行不必要的扩容操作的关键在于，在使用 vector 容器初期，就要将其容量设为足够大的值
* 在 vector 容器刚刚构造出来的那一刻，就应该借助 reserve() 成员方法为其扩充足够大的容量。

### 5.5.8. vector 的容量 fit


```cpp

// 使用 shrink_to_fit() 来调整容量到当前的元素内容大小
vector<int>myvector;
myvector.shrink_to_fit();

// 使用 swap 成员函数来
// 该方法的基础功能是交换 2 个相同类型的 vector 容器（交换容量和存储的所有元素）
vector<int>(myvector).swap(myvector);

//清空 myvector 容器
vector<int>().swap(myvector);
```

swap()成员函数的执行流程:  
1) 先执行 vector<int>(myvector)，此表达式会调用 vector 模板类中的拷贝构造函数，从而创建出一个临时的 vector 容器（后续称其为 tempvector）。
2) 值得一提的是，tempvector 临时容器并不为空，因为我们将 myvector 作为参数传递给了复制构造函数，该函数会将 myvector 容器中的所有元素拷贝一份，并存储到 tempvector 临时容器中。

    注意，vector 模板类中的拷贝构造函数只会为拷贝的元素分配存储空间。换句话说，tempvector 临时容器中没有空闲的存储空间，其容量等于存储元素的个数。
3) 然后借助 swap() 成员方法对 tempvector 临时容器和 myvector 容器进行调换，此过程不仅会交换 2 个容器存储的元素，还会交换它们的容量。换句话说经过 swap() 操作，myvetor 容器具有了 tempvector 临时容器存储的所有元素和容量，同时 tempvector 也具有了原 myvector 容器存储的所有元素和容量。
4) 当整条语句执行结束时，临时的 tempvector 容器会被销毁，其占据的存储空间都会被释放。注意，这里释放的其实是原 myvector 容器占用的存储空间。
经过以上步骤，就成功的将 myvector 容器的容量由 100 缩减至 10。  


## 5.6. deque 双端队列

deque 容器和 vecotr 容器有很多相似之处

主要要点:
1. deque 是 double-ended queue 的缩写，又称双端队列容器。
2. deque 容器中存储元素并`不能保证`所有元素都存储到连续的内存空间中。
3. deque 容器也擅长在序列尾部添加或删除元素（时间复杂度为O(1)），而不擅长在序列中间添加或删除元素。
4. deque 容器也可以根据需要修改自身的容量和大小。

当需要向序列两端频繁的添加或删除元素时，应首选 deque 容器。   
和 vector 相比，额外增加了实现在容器头部添加和删除元素的成员函数，同时删除了 `capacity()`、`reserve()` 和 `data()` 成员函数。     


### 5.6.1. deque的底层实现原理

和 vector 容器采用连续的线性空间不同!!  

1. deque 容器存储数据的空间是由许多`段`等长的连续空间构成，各段空间之间并不一定是连续的，可以位于在内存的不同区域  
2. deque 容器用数组（数组名假设为 map）存储着各个连续空间的首地址。也就是说，map 数组中存储的都是指针，指向那些真正用来存储数据的各个连续空间
3. 当 deque 容器需要在头部或尾部增加存储空间时，它会申请一段新的连续空间，同时在 map 数组的开头或结尾添加指向该空间的指针，由此该空间就串接到了 deque 容器的头部或尾部。
4. 如果 map 数组满了, 再申请一块更大的连续空间供 map 数组使用，将原有数据（很多指针）拷贝到新的 map 数组中，然后释放旧的空间。map类似于普通 vector

对迭代器的底层实现, 可以学习运算符重载函数的使用
```cpp
//当迭代器处于当前连续空间边缘的位置时，如果继续遍历，就需要跳跃到其它的连续空间中，该函数可用来实现此功能
void set_node(map_pointer new_node){
  node = new_node;//记录新的连续空间在 map 数组中的位置
  first = *new_node; //更新 first 指针
  //更新 last 指针，difference_type(buffer_size())表示每段连续空间的长度
  last = first + difference_type(buffer_size());
}
//重载 * 运算符
reference operator*() const{return *cur;}
pointer operator->() const{return &(operator *());}
//重载前置 ++ 运算符
self & operator++(){
  ++cur;
  //处理 cur 处于连续空间边缘的特殊情况
  if(cur == last){
      //调用该函数，将迭代器跳跃到下一个连续空间中
      set_node(node+1);
      //对 cur 重新赋值
      cur = first;
  }
  return *this;
}
//重置前置 -- 运算符
self& operator--(){
  //如果 cur 位于连续空间边缘，则先将迭代器跳跃到前一个连续空间中
  if(cur == first){
      set_node(node-1);
      cur == last;
  }
  --cur;
  return *this;
}
```

deque 容器除了维护先前讲过的 map 数组，还需要维护 start、finish 这 2 个 deque 迭代器  
1. start 迭代器记录着 map 数组中首个连续空间的信息
2. start 迭代器中的 cur 指针指向的是连续空间中首个元素
3. finish 迭代器记录着 map 数组中最后一个连续空间的信息
4. 而 finish 迭代器中的 cur 指针指向的是连续空间最后一个元素的下一个位置

deque 的源代码定义
```cpp
//_Alloc为内存分配器
template<class _Ty,
  class _Alloc = allocator<_Ty>>
class deque{
  ...
protected:
  iterator start;
  iterator finish;
  map_pointer map;
...
}
```

### 5.6.2. 定义deque

```cpp
// 空队列
std::deque<int> d;

// 初始元素的个数, 赋予默认值
std::deque<int> d(10);

// 初始元素的个数以及指定初始值
std::deque<int> d(10, 5);

// 拷贝构造
std::deque<int> d2(d1);

// 指定区间拷贝 (从其他任何容器中)
std::array<int, 5>arr{ 11,12,13,14,15 };
std::deque<int>d(arr.begin()+2, arr.end()); //13,14,15
```

### 5.6.3. 迭代器

八大迭代器方法, 不多说  
begin();  end();    
rbegin();  rend();  
cbegin();  cend();  
crbegin();  crend();  

全局函数适用于 deque `begin(deque) end(deque)`  
```cpp
for (auto i = d.begin(); i < d.end(); i++) 
for (auto i = begin(d); i < end(d); i++)
```

### 5.6.4. 访问元素

基本同 vector 一致  

注意，和 vector 容器不同，deque 容器没有提供 data() 成员函数，  
同时 deque 容器在存储元素时，也无法保证其会将元素存储在连续的内存空间中，因此尝试使用指针去访问 deque 容器中指定位置处的元素，是非常危险的。  

```cpp
deque<int>d{ 1,2,3,4 };

// 下标访问
cout << d[1] << endl;

// at() 防止越界访问  抛出 std::out_of_range 异常
d.at(1) = 5;

// vector 也有的 front() 和 back() 函数 ,  
d.front() = 10;
d.back() = 20;
```

### 5.6.5. 修改元素 增删

push_back()、push_front() 或者 resize() 成员函数实现向（空）deque 容器中添加元素。  
在实际应用中，常用 emplace()、emplace_front() 和 emplace_back() 分别代替 insert()、push_front() 和 push_back()  

```cpp
// 单参数 单使用方法函数
push_back()     // 在容器现有元素的尾部添加一个元素，和 emplace_back() 不同，该函数添加新元素的过程是，先构造元素，然后再将该元素移动或复制到容器的尾部。
pop_back()      // 移除容器尾部的一个元素。
push_front()    // 在容器现有元素的头部添加一个元素，和 emplace_back() 不同，该函数添加新元素的过程是，先构造元素，然后再将该元素移动或复制到容器的头部。
pop_front()     // 移除容器尾部的一个元素。
emplace_back()  // C++ 11 新添加的成员函数，其功能是在容器尾部生成一个元素。和 push_back() 不同，该函数直接在容器头部构造元素，省去了复制或移动元素的过程。
emplace_front() // C++ 11 新添加的成员函数，其功能是在容器头部生成一个元素。和 push_front() 不同，该函数直接在容器头部构造元素，省去了复制或移动元素的过程。

//emplace() 需要 2 个参数，第一个为指定插入位置的迭代器，第二个是插入的值。
d.emplace(d.begin() + 1, 4);


insert()    // 在指定的位置直接生成一个元素。和 emplace() 不同的是，该函数添加新元素的过程是，先构造元素，然后再将该元素移动或复制到容器的指定位置。
// insert 有四个重载 具体看vector 一样的

// 移除一个元素或某一区域内的多个元素。
iterator erase( const_iterator pos );
iterator erase( const_iterator first, const_iterator last );

clear()     // 删除容器中所有的元素。

```

## 5.7. list 双向链表

实际场景中，如何需要对序列进行大量添加或删除元素的操作，而直接访问元素的需求却很少，这种情况建议使用 list 容器存储序列。  

1. list底层是以双向链表的形式实现的。这意味着，list 容器中的元素可以分散存储在内存空间里，而不是必须存储在一整块连续的内存空间中。  
2. 每个元素都配备了 2 个指针，分别指向它的前一个元素和后一个元素。其中第一个元素的前向指针总为 null，因为它前面没有元素；同样，尾部元素的后向指针也总为 null。  
3. list 容器具有一些其它容器（array、vector 和 deque）所不具备的优势，即它可以在序列已知的任何位置快速插入或删除元素（时间复杂度为O(1)）。并且在 list 容器中移动元素，也比其它容器的效率高。
4. 缺点是，它不能像 array 和 vector 那样，通过位置直接访问元素。举个例子，如果要访问 list 容器中的第 6 个元素，它不支持容器对象名[6]这种语法格式，正确的做法是从容器中第一个元素或最后一个元素开始遍历容器，直到找到该位置。

头文件: `#include <list>`  

### 5.7.1. 定义  
```cpp

// 空
std::list<int> values;

// 指定大小
std::list<int> values(10);

// 大小和初始值
std::list<int> values(10, 5);

// 拷贝
std::list<int> value2(value1);

// 指定  起始和终止位置  拷贝
std::list<int> values(a, a+5);

//拷贝其它类型的容器，创建 list 容器
std::array<int, 5>arr{ 11,12,13,14,15 };
std::list<int>values(arr.begin()+2, arr.end());
```

### 5.7.2. 迭代器

同理, 也是8大迭代器方法和 全局的 `begin() end()`  

但是和 `array、vector、deque` 容器的迭代器相比，list 容器迭代器最大的不同在于，其配备的迭代器类型为双向迭代器，而不再是随机访问迭代器。  

体现在:
1. 不能通过下标访问 list 容器中指定位置处的元素。`p[i]`
2. 双向迭代器 p1 不支持使用 -=、+=、+、- 运算符. `p[i]`
3. 双向迭代器 p1、p2 不支持使用 <、 >、 <=、 >= 比较运算符。 迭代器之间不能进行大小比较.
   程序中比较迭代器之间的关系，用的是 != 运算符，因为它不支持 < 等运算符。


* list 容器在进行插入（insert()）、接合（splice()）等操作时，都不会造成原有的 list 迭代器失效，甚至进行删除操作，而只有指向被删除元素的迭代器失效，其他迭代器不受任何影响。
* 在进行插入操作之后，仍使用先前创建的迭代器遍历容器，虽然程序不会出错，但由于插入位置的不同，**可能会遗漏新插入的元素**。

```cpp
for (std::list<char>::iterator it = values.begin(); it != values.end(); ++it)

for (std::list<char>::reverse_iterator it = values.rbegin(); it != values.rend();++it)

```
### 5.7.3. 访问元素的方法

list 容器不支持随机访问，未提供下标操作符 [] 和 at() 成员函数，也没有提供 data() 成员函数。  

访问 list 容器中存储元素要么使用 front() 和 back() 成员函数，要么使用 list 容器迭代器.  

```cpp
int &first = mylist.front();
int &last = mylist.back();

//修改
first = 10;
last = 20;

auto it = mylist.begin();
while (it!=mylist.end())
{
    cout << *it << " ";
    ++it;  
}

```
### 5.7.4. 添加元素

list 模板类中，与“添加或插入新元素”相关的成员方法有如下7个：
单语法格式:
* push_front()：向 list 容器首个元素前添加新元素；
* push_back()：向 list 容器最后一个元素后添加新元素；
* emplace_front()：在容器首个元素前直接生成新的元素；
* emplace_back()：在容器最后一个元素后直接生成新的元素；
* emplace(pos,value)：在容器的指定位置直接生成新的元素；
多语法格式:
* insert()：在指定位置插入新元素；
  * insert(pos,elem) 	 在迭代器 pos 指定的位置之前插入一个新元素 elem，并返回表示新插入元素位置的迭代器。
  * insert(pos,n,elem) 	在迭代器 pos 指定的位置之前插入 n 个元素 elem，并返回表示第一个新插入元素位置的迭代器。
  * insert(pos,first,last)  	在迭代器 pos 指定的位置之前，插入其他容器中位于 [first,last) 区域的所有元素，并返回表示第一个新插入元素位置的迭代器。
  * insert(pos,initlist) 	在迭代器 pos 指定的位置之前，插入初始化列表（用大括号 { } 括起来的多个元素，中间有逗号隔开）中所有的元素，并返回表示第一个新插入元素位置的迭代器。
* splice()：将其他 list 容器存储的多个元素添加到当前 list 容器的指定位置处。
  * splice (position, list& x);               将 x 容器中存储的所有元素全部移动当前 list 容器中 position 指明的位置处
  * splice (position, list& x, i);            x 容器中 i 指向的元素移动到当前容器中 position 指明的位置处
  * splice (position, list& x, first, last);  将 x 容器 [first, last) 范围内所有的元素移动到当前容器 position 指明的位置处。


splice() 成员方法移动元素的方式是，将存储该元素的节点从 list 容器底层的链表中摘除，然后再链接到当前 list 容器底层的链表中。  
这意味着，当使用 splice() 成员方法将 x 容器中的元素添加到当前容器的同时，该元素会从 x 容器中删除。  

```cpp
//创建并初始化 2 个 list 容器
list<int> mylist1{ 1,2,3,4 }, mylist2{10,20,30};
list<int>::iterator it = ++mylist1.begin(); //指向 mylist1 容器中的元素 2

//调用第一种语法格式
mylist1.splice(it, mylist2); // mylist1: 1 10 20 30 2 3 4
                             // mylist2:
                             // it 迭代器仍然指向元素 2，只不过容器变为了 mylist1
//调用第二种语法格式，将 it 指向的元素 2 移动到 mylist2.begin() 位置处
mylist2.splice(mylist2.begin(), mylist1, it);   // mylist1: 1 10 20 30 3 4
                                                // mylist2: 2
                                                // it 仍然指向元素 2

//调用第三种语法格式，将 [mylist1.begin(),mylist1.end())范围内的元素移动到 mylist.begin() 位置处                  
mylist2.splice(mylist2.begin(), mylist1, mylist1.begin(), mylist1.end());//mylist1:
                                                                         //mylist2:1 10 20 30 3 4 2
```

### 5.7.5. 其他元素操作


并不是所有的容器都有 `sort()`方法, 通过更改容器中元素的位置，将它们进行排序。
* 有sort方法的容器只有, list 和 forward_list
* 其他容器可以使用`algorithm`里的全局 sort() 函数

`reverse()` 反转容器中元素的顺序。  

forward_list 容器中是不提供 size() 函数的，但如果想要获取 forward_list 容器中存储元素的个数，可以使用头文件 `<iterator>` 中的 distance() 函数。
```cpp
#include <forward_list>
#include <iterator>

td::forward_list<int> my_words{1,2,3,4};
int count = std::distance(std::begin(my_words), std::end(my_words));
```


forward_list 容器迭代器的移动除了使用` ++ `运算符单步移动，还能使用 `advance()` 函数  
```cpp
std::forward_list<int> values{1,2,3,4};
auto it = values.begin();
// 往前迭代2个位置
advance(it, 2);
```

### 5.7.6. 删除元素 

简单方法
* pop_front() 	删除位于 list 容器头部的一个元素。
* pop_back() 	删除位于 list 容器尾部的一个元素。
* clear() 	删除 list 容器存储的所有元素。

指定位置删除
* erase(position) 	该成员函数既可以删除 list 容器中指定位置处的元素，也可以删除容器中某个区域内的多个元素。
* erase(first,last) 	该成员函数既可以删除 list 容器中指定位置处的元素，也可以删除容器中某个区域内的多个元素。
* remove(val) 	删除容器中所有等于 val 的元素。


* unique() 	删除容器中**相邻的**重复元素，只保留一份。
* unique(BinaryPredicate) //传入一个二元谓词函数

* remove_if() 	删除容器中满足条件的元素。


### 5.7.7. forward_list  C++11 单向链表

头文件`#include <forward_list>`  
forward_list 使用的是单链表，而 list 使用的是双向链表  
* 存储相同个数的同类型元素，单链表耗用的内存空间更少，空间利用率更高，并且对于实现某些操作单链表的执行效率也更高。
* 只要是 list 容器和 forward_list 容器都能实现的操作，应优先选择 forward_list 容器。



forward_list 容器具有和 list 容器相同的特性，即擅长在序列的任何位置进行插入元素或删除元素的操作,  
但是由于单链表没有双向链表那样灵活，因此相比 list 容器，forward_list 容器的功能受到了很多限制。  
比如，由于单链表只能从前向后遍历，而不支持反向遍历，因此 forward_list 容器只提供前向迭代器，而不是双向迭代器。  




### 5.7.8. 构造函数

```cpp
// c++ 11 标准之前
// 默认构造函数，即创建空的 pair 对象
pair();

// 直接使用 2 个元素初始化成 pair 对象
pair (const first_type& a, const second_type& b);

// 拷贝（复制）构造函数，即借助另一个 pair 对象，创建新的 pair 对象
template<class U, class V> pair (const pair<U,V>& pr);


// 基于右值引用的构造函数
// 移动构造函数
template<class U, class V> pair (pair<U,V>&& pr);
pair <string, string> pair4(make_pair("C++教程", "http://c.biancheng.net/cplus/"));


// 使用右值引用参数，创建 pair 对象
template<class U, class V> pair (U&& a, V&& b);
```
调用 make_pair() 函数，它也是` <utility>` 头文件提供的，其功能是生成一个 pair 对象。  
当我们将 make_pair() 函数的返回值（是一个临时对象）作为参数传递给 pair() 构造函数时，其调用的是移动构造函数，而不是拷贝构造函数。  

### 5.7.9. 值修改

通过访问 `first second` 就可以访问pair的值  
pair类模板还提供有一个 swap() 成员函数，能够互换 2 个 pair 对象的键值对，其操作成功的前提是这 2 个 pair 对象的键和值的类型要相同  

```cpp
pair1.first = "Java教程";
pair1.second = "http://c.biancheng.net/java/";

pair1.swap(pair2);

```



## 5.8. map multimap

* 向map容器中增添元素，insert()效率更高
* 更新map容器中的键值对，operator[]效率更高

### 5.8.1. 构造函数 定义

```cpp
// map 的定义
template < class Key,                                     // 指定键（key）的类型
           class T,                                       // 指定值（value）的类型
           class Compare = less<Key>,                     // 指定排序规则
           class Alloc = allocator<pair<const Key,T> >    // 指定分配器对象的类型
           > class map;


// 空 map 的建立
map<int, string> mapStudent;

// 初始化的方法 , 使用{ {,},{,} } 的方式
std::map<std::string, int>myMap{ {"C语言教程",10},{"STL教程",20} };

// map 容器中存储的键值对，其本质都是 pair 类模板创建的 pair 对象, 因此也可以
std::map<std::string, int>myMap{std::make_pair("C语言教程",10),std::make_pair("STL教程",20)};
std::map<std::string, int>myMap{std::pair<string,int>{"C语言教程",10},std::pair<string,int>{"STL教程",20}};


// 拷贝构造
std::map<std::string, int>newMap(myMap);

// 移动构造 disMap()是返回一个 map 的函数
std::map<std::string, int>newMap(disMap());

// 指定区域拷贝构造
std::map<std::string, int>myMap{ {"C语言教程",10},{"STL教程",20} };
std::map<std::string, int>newMap(++myMap.begin(), myMap.end());


// 指定排序规则
std::map<std::string, int, std::greater<std::string> >myMap{ {"C语言教程",10},{"STL教程",20} };
```

### 5.8.2. 获取值 迭代器

map 类模板中对[ ]运算符进行了重载，这意味着，类似于借助数组下标可以直接访问数组中元素，通过指定的键，我们可以轻松获取 map 容器中该键对应的值
1. 有当 map 容器中确实存有包含该指定键的键值对，借助重载的 [ ] 运算符才能成功获取该键对应的值
2. 若当前 map 容器中没有包含该指定键的键值对，则此时使用 [ ] 运算符将不再是访问容器中的元素，而变成了向该 map 容器中增添一个键值对
   * 键用 [ ] 运算符中指定的键
   * 值取决于值的数据类型，如果是基本数据类型，则值为 0；如果是 string 类型，其值为 ""，即空字符串（即使用该类型的默认值作为键值对的值）
  
调用  at() 成员方法
1. 成功找到键的键值对，并返回该键对应的值
2. 没有键的键值对，会导致 at() 成员方法查找失败，并抛出 `out_of_range` 异常   

借助 find() 成员方法间接实现此目的
1. 该方法返回的是一个迭代器
2. 如果查找成功，该迭代器指向查找到的键值对
3. 反之，则指向 map 容器最后一个键值对之后的位置（和 end() 成功方法返回的迭代器一样
   
```cpp
// 两个函数  find()  和 end()
// find 返回迭代器指向当前查找元素的位置否则返回map::end()位置
iter = mapStudent.find("123");
if(iter != mapStudent.end())
  cout<<"Find, the value is"<<iter->second<<endl;
else
  cout<<"Do not Find"<<endl;

```
find 函数返回迭代器对象 , 可以使用 `->` 直接获取数据


* lower_bound() upper_bound() equal_range()
```cpp
// lower_bound() upper_bound()  返回一个迭代器, 用来指向第一个[索引值]大于参数的迭代器
const std::map<int, const char*> m{ 
  { 0, "zero" },
  { 1, "one" },
  { 2, "two" },
};


// euqal_range() 则是返回一个pair 用来包含键值等于参数的始末迭代器
auto p = m.equal_range(1);
for (auto& q = p.first; q != p.second; ++q) {
    std::cout << "m[" << q->first << "] = " << q->second << '\n';
}
// If there are no elements not less than key, past-the-end (see end()) iterator is returned as the first element. 
// Similarly if there are no elements greater than key, past-the-end iterator is returned as the second element. 

```



### 5.8.3. 插入元素

```cpp
// 定义一个map对象
map<int, string> mapStudent;
 
// 第一种 用insert函數插入 pair
mapStudent.insert(pair<int, string>(000, "student_zero"));
 
// 第二种 用insert函数插入 value_type 数据
mapStudent.insert(map<int, string>::value_type(001, "student_one"));
 
// 第三种 用"array"方式插入
mapStudent[123] = "student_first";
mapStudent[456] = "student_second";

// 关于 Insert 函数的返回值

// 这是 insert 函数的构造定义，返回一个pair对象
pair<iterator,bool> insert (const value_type& val);

// 要获取是否成功 , 需要建立一个对应的 pair 
pair<map<int, string/*这里需要和原本的map对象相同*/>::iterator, bool> Insert_Pair;
Insert_Pair = mapStudent.insert(map<int, string>::value_type (001, "student_one"));

//如果插入成功的话Insert_Pair.second应该是true的，否则为false
if(!Insert_Pair.second)cout << ""Error insert new element" << endl;

```
使用 insert 函数插入和使用数组方式插入主要区别在于数据的唯一性  
若关键字已存在 , insert 会返回插入失败 
而数组方式会直接覆盖  



除此之外，insert() 方法还支持向 map 容器的指定位置插入新键值对，该方法的语法格式如下
```cpp
iterator insert (const_iterator position, const value_type& val);
// 这里 insert() 方法返回的是迭代器
// 如果插入成功，insert() 方法会返回一个指向 map 容器中已插入键值对的迭代器；而不再是pair
// 如果插入失败，insert() 方法同样会返回一个迭代器，该迭代器指向 map 容器中和 val 具有相同键的那个键值对。


// insert() 方法还支持向当前 map 容器中插入其它 map 容器指定区域内的所有键值对，该方法的语法格式如下：
template <class InputIterator> void insert (InputIterator first, InputIterator last);

// 除了以上一种格式外，insert() 方法还允许一次向 map 容器中插入多个键值对，其语法格式为：
// 其中，vali 都表示的是键值对变量。
void insert ({val1, val2, ...});

```

emplace类函数 实现相同的插入操作，无论是用 emplace() 还是 emplace_hont()，都比 insert() 方法的效率

```cpp
// 和 insert() 方法相比，emplace() 和 emplace_hint() 方法的使用要简单很多，因为它们各自只有一种语法格式

template <class... Args> pair<iterator,bool> emplace (Args&&... args);
// 这里只需要将创建新键值对所需的数据作为参数直接传入即可，此方法可以自行利用这些数据构建出指定的键值对
// 另外，该方法的返回值也是一个 pair 对象


template <class... Args> iterator emplace_hint (const_iterator position, Args&&... args);
// 和 emplace() 语法格式相比 该方法不仅要传入创建键值对所需要的数据，还需要传入一个迭代器作为第一个参数，指明要插入的位置
// 返回值是一个迭代器，而不再是 pair 对象


```

### 5.8.4. 删除元素 
```cpp
iterator erase( const_iterator pos );         //(since C++11)
iterator erase( iterator pos );               //(since C++17)
iterator erase( const_iterator first, const_iterator last );  //(since C++11)
size_type erase( const key_type& key );

/* 
Parameters
pos 	- 	iterator to the element to remove
first, last 	- 	range of elements to remove
key 	- 	key value of the elements to remove 
*/

//迭代器作为参数刪除 , 返回指向下一个元素的迭代器
iter = mapStudent.find("123");
iter= mapStudent.erase(iter);

//用关键字作为参数刪除 , 返回删除的元素个数
int n = mapStudent.erase("123");

//两个迭代器对象作为参数, 表示范围删除 , 下行代码表示清空 map
mapStudent.erase(mapStudent.begin(), mapStudent.end());

//等同于
mapStudent.clear()
```
### 5.8.5. 获取大小

```cpp
//函数定义
size_type size() const noexcept;

//一般使用方法
int nSize = mapStudent.size();
```
### 5.8.6. 其他常用函数
begin()         返回指向map头部的迭代器  
end()           返回指向map末尾的迭代器  
rbegin()        返回一个指向map尾部的逆向迭代器  
rend()          返回一个指向map头部的逆向迭代器

empty()         如果map为空则返回true
clear()         清空    
swap()          交换两个map的内容  
size()          返回map中元素的个数  
max_size()      返回可以容纳的最大元素个数 和操作系统有关, 返回值会不同    
at()            相比于[]访问更安全的访问方法, 越界会抛出异常  

count()         返回指定key出现的次数   因为key唯一, 因此返回值最大为 1  
insert()        容器中插入键值对  
emplace()       高效插入  
emplace_hind()  需要一个指向位置的迭代器作为第一个参数  

find(key)           返回指向键为key的双向迭代器  
lower_bound(key)    返回指向当前容器第一个大于或等于key的双向迭代器  
upper_bound()       返回指向当前容器第一个大于key的双向迭代器  
equal_range()       返回一个 pair 分别包含了 lower_bound()和 equal_range() 的返回值  最多包含一个键值对    

get_allocator() 返回map的配置器  
key_comp()      返回比较元素key的函数  
value_comp()     返回比较元素value的函数  

### 5.8.7. multimap

multimap 容器中指定的键可能对应多个键值对，而不再是 1 个。  

* 和 map 容器相比，multimap 未提供 at() 成员方法，也没有重载 [] 运算符。  
* 这意味着，map 容器中通过指定键获取指定指定键值对的方式，将不再适用于 multimap 容器。
* 但是只要是 multimap 容器提供的成员方法，map 容器都提供

由于 multimap 容器可存储多个具有相同键的键值对，  
因此表 1 中的 `lower_bound()、upper_bound()、equal_range()` 以及 `count()` 成员方法会经常用到  


## 5.9. set multiset 

set 容器定义于`<set>`头文件  

* 和 map、multimap 容器不同，使用 set 容器存储的各个键值对，要求键 key 和值 value 必须相等。  当使用 set 容器存储键值对时，只需要为其提供各键值对中的 value 值（也就是 key 的值）即可。  
* map、multimap 容器都会自行根据键的大小对存储的键值对进行排序， set 容器也会如此，只不过 set 容器中各键值对的键 key 和值 value 是相等的，根据 key 排序，也就等价为根据 value 排序。  


用 set 容器存储的各个元素的值`必须各不相同`。更重要的是，从语法上讲 set 容器并没有强制对存储元素的类型做 const 修饰，即 set 容器中存储的元素的值是可以修改的。  
但是，C++ 标准为了防止用户修改容器中元素的值，对所有可能会实现此操作的行为做了限制，使得在正常情况下，用户是无法做到修改 set 容器中元素的值的。  
* 切勿尝试直接修改 set 容器中已存储元素的值，这很有可能破坏 set 容器中元素的有序性，
* 最正确的修改 set 容器中元素值的做法是：先删除该元素，然后再添加一个修改后的元素。

### 5.9.1. 创建set

定义对象
```cpp

template < class T,                        // 键 key 和值 value 的类型
           class Compare = less<T>,        // 指定 set 容器内部的排序规则
           class Alloc = allocator<T>      // 指定分配器对象的类型
           > class set;

// 默认构造函数
// 由于 set 容器支持随时向内部添加新的元素，因此创建空 set 容器的方法是经常使用的。
std::set<std::string> myset;

// 定义的同时初始化
std::set<std::string> myset{"java","stl","python"};

// 拷贝构造函数
std::set<std::string> copyset(myset);
// 等同于
std::set<std::string> copyset = myset


// 移动构造函数 retSet() 函数的返回值是一个临时 set 容器
std::set<std::string> copyset(retSet());
//或者
std::set<std::string> copyset = retSet();


// 拷贝部分元素初始化
std::set<std::string> copyset(++myset.begin(), myset.end());


// 指定排序方法 降序
std::set<std::string,std::greater<string> > myset{"java","stl","python"};

```

### 5.9.2. 迭代器

1. C++ STL 中的 set 容器类模板中未提供 at() 成员函数，也未对 [] 运算符进行重载。因此，要想访问 set 容器中存储的元素，只能借助 set 容器的迭代器。  
2.  set 容器配置的迭代器类型为双向迭代器。这意味着，假设 p 为此类型的迭代器，则其只能进行 ++p、p++、--p、p--、*p 操作，并且 2 个双向迭代器之间做比较，也只能使用 == 或者 != 运算符。
3.  成员方法返回的迭代器，指向的只是 set 容器中存储的元素，而不再是键值对。
4.  成员方法返回的迭代器，无论是 const 类型还是非 const 类型，都不能用于修改 set 容器中的值。


八大迭代器函数  

如果只想遍历 set 容器中指定区域内的部分数据  
其他迭代器方法: 如果 set 容器用 const 限定，则该方法返回的是 const 类型的双向迭代器。
1. find(val)          如果成功找到，则返回指向该元素的双向迭代器；反之，则返回和 end() 方法一样的迭代器。
2. lower_bound(val)   返回一个指向当前 set 容器中第一个大于或等于 val 的元素的双向迭代器。
3. upper_bound(val)   返回一个指向当前 set 容器中第一个大于 val 的元素的迭代器。
4. equal_range(val)   
   * 该方法返回一个 pair 对象（包含 2 个双向迭代器），其中 pair.first 和 lower_bound() 方法的返回值等价，pair.second 和 upper_bound() 方法的返回值等价。
   * set 容器中各个元素是唯一的，因此该范围最多包含一个元素
  
虽然 C++ STL 标准中，set 类模板中包含 lower_bound()、upper_bound()、equal_range() 这 3 个成员函数，但它们更适用于 `multiset` 容器，几乎不会用于操作 set 容器。  

### 5.9.3. 插入元素

`.insert()` 方法 val 表示要添加的新元素，该方法的返回值为 pair 类型  
插入单个元素
```cpp
//普通引用方式传参
pair<iterator,bool> insert (const value_type& val);

//右值引用方式传参
pair<iterator,bool> insert (value_type&& val);


// 指定插入位置的重载方法 这 2 种语法格式中，insert() 函数的返回值为迭代器：

//以普通引用的方式传递 val 值
iterator insert (const_iterator position, const value_type& val);
//以右值引用的方式传递 val 值
iterator insert (const_iterator position, value_type&& val);
```

* 当向 set 容器添加元素成功时，该迭代器指向 set 容器新添加的元素，bool 类型的值为 `true；`
* 如果添加失败，即证明原 set 容器中已存有相同的元素，此时返回的迭代器就指向容器中`相同的此元素` ，同时 bool 类型的值为 `false。`
* 注意，使用 insert() 方法将目标元素插入到 set 容器指定位置后，如果该元素破坏了容器内部的有序状态，set 容器还会自行对新元素的位置做进一步调整。
  * 即insert() 方法中指定新元素插入的位置，并不一定就是该元素最终所处的位置。


插入多个元素
```cpp
// insert() 方法支持向当前 set 容器中插入其它 set 容器指定区域内的所有元素，只要这 2 个 set 容器存储的元素类型相同即可。
template <class InputIterator> void insert (InputIterator first, InputIterator last);
// first 和 last 都是迭代器，它们的组合 [first,last) 可以表示另一 set 容器中的一块区域，该区域包括 first 迭代器指向的元素，但不包含 last 迭代器指向的元素


// 一次向 set 容器中添加多个元素：
void insert ( {E1, E2,...,En} );
myset.insert({ "stl","python","java"});

```
### 5.9.4. 插入元素2

`emplace()` 和 `emplace_hint()` 是 C++ 11 标准加入到 set 类模板中的，相比具有同样功能的 insert() 方法，完成同样的任务，emplace() 和 emplace_hint() 的效率会更高。  


```cpp
// 参数 (Args&&... args) 指的是，只需要传入构建新元素所需的数据即可，该方法可以自行利用这些数据构建出要添加的元素。
// 若 set 容器中存储的元素类型为自定义的结构体或者类，则在使用 emplace() 方法向容器中添加新元素时，构造新结构体变量（或者类对象）需要多少个数据，就需要为该方法传入相应个数的数据。
template <class... Args> pair<iterator,bool> emplace (Args&&... args);
template <class... Args> iterator emplace_hint (const_iterator position, Args&&... args);

pair<set<string, string>::iterator, bool> ret = myset.emplace("http://c.biancheng.net/stl/");
```

`emplace` 的返回值类型为 pair 类型，其包含 2 个元素，一个迭代器和一个 bool 值：
* 当该方法将目标元素成功添加到 set 容器中时，其返回的迭代器指向新插入的元素，同时 bool 值为 true；
* 当添加失败时，则表明原 set 容器中已存在相同值的元素，此时返回的迭代器指向容器中具有相同键的这个元素，同时 bool 值为 false。
`emplace_hint`和 emplace() 方法相比，有以下 2 点不同
* 该方法需要额外传入一个迭代器，用来指明新元素添加到 set 容器的具体位置（新元素会添加到该迭代器指向元素的前面）；
* 返回值是一个迭代器，而不再是 pair 对象。当成功添加元素时，返回的迭代器指向新添加的元素；反之，如果添加失败，则迭代器就指向 set 容器和要添加元素的值相同的元素。


### 5.9.5. 删除元素


set也是使用 `erase` 删除元素  `clear()`;
```cpp
//删除 set 容器中值为 val 的元素  注意是删除值
size_type erase (const value_type& val);

//删除 position 迭代器指向的元素
iterator  erase (const_iterator position);

//删除 [first,last) 区间内的所有元素
iterator  erase (const_iterator first, const_iterator last);
```
* 第 1 种格式的 erase() 方法，其返回值为一个整数，表示成功删除的元素个数；
* 后 2 种格式的 erase() 方法，返回值都是迭代器，其指向的是 set 容器中删除元素之后的第一个元素
* 如果要删除的元素就是 set 容器最后一个元素，则 erase() 方法返回的迭代器等价于 end() 方法返回的迭代器。

### 5.9.6. multiset
回忆一下，set 容器具有以下几个特性：
* 不再以键值对的方式存储数据，因为 set 容器专门用于存储键和值相等的键值对，因此该容器中真正存储的是各个键值对的值（value）；
* set 容器在存储数据时，会根据各元素值的大小对存储的元素进行排序（默认做升序排序）；
* 存储到 set 容器中的元素，虽然其类型没有明确用 const 修饰，但正常情况下它们的值是无法被修改的；
* set 容器存储的元素必须互不相等。

C++ STL 标准库中还提供有一个和 set 容器相似的关联式容器， 也定义在`<set>`头文件  `multiset` 容器可以存储多个值相同的元素。  
唯一的差别在于，multiset 容器允许存储多个值相同的元素，而 set 容器中只能存储互不相同的元素。  

虽然 multiset 容器和 set 容器拥有的成员方法**完全相同**，但由于 multiset 容器允许存储多个值相同的元素，因此诸如 
1. count()
2. find()
3. lower_bound()
4. upper_bound()
5. equal_range() 等方法，更常用于 multiset 容器。

# 6. 数值库 Numerics

```cpp
#include <complex>
#include <random>
#include <valarray>
#include <numeric>
#include <bit>
#include <numbers>
#include <cfenv>
#include <cmath>
```

标准C++数值库包含了常规数学函数以及类型， 以及一些特殊化的数列和随机数生成。
包括了一系列的头文件  

## 6.1. cmath 通用数学函数

包括了从 C语言继承来的一些 通用数学运算  

### 基础运算函数



### 三角函数

### 指数函数

### 对数函数

## 6.3. number 数学常量

## 6.4. complex 复数运算
