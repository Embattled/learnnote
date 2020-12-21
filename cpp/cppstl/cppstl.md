# 1. C语言标准库

C语言的标准库一共只有二十多个


1. 一般库提供方不直接向用户提供目标文件，而是将多个相关的目标文件打包成一个静态链接库（Static Link Library），例如 Linux 下的 `.a` 和 Windows 下的 `.lib`。
2. C语言在发布的时候已经将标准库打包到了静态库，并提供了相应的头文件，例如 stdio.h、stdlib.h、string.h 等。   
3. Linux 一般将静态库放在`/lib`和`/usr/lib`,头文件放在`/usr/include` 下

最早的 ANSI C 标准共定义了 15 个头文件，称为“C标准库”，所有的编译器都必须支持，如何正确并熟练的使用这些标准库，可以反映出一个程序员的水平：  
* 合格程序员：`<stdio.h>`、`<ctype.h>`、`<stdlib.h>`、`<string.h>`
* 熟练程序员：`<assert.h>`、`<limits.h>`、`<stddef.h>`、`<time.h>`
* 优秀程序员：`<float.h>`、`<math.h>`、`<error.h>`、`<locale.h>`、`<setjmp.h>`、`<signal.h>`、`<stdarg.h>`

# 2. C++ 标准环境

STL是Standard Template Library 的简称  

## 2.1. C++ 编译环境的构成

一个完整的C++环境由 库和编译模块构成

库中包括
* C++标准库     即 STL, 不带`.h`的头文件, 在std命名空间
* C语言兼容库   头文件带`.h` , 不是C++标准的内容,但是C++编译器提供商一般都会提供C的兼容库, 即编译器内置的C库
* 编译器扩展库  每个编译器独有的拓展库 G++和VC++的拓展库就不同 如 `stdafx.h`
编译模块
* C++ 标准语法模块    对C++标准语法的支持
* C++ 扩展语法模块    每个编译器独有的扩展语法的支持 

## 2.2. C++ 标准库

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

## 2.3. cppreference.com  的C++标准库

从网站上拷贝的最全的标准库

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
