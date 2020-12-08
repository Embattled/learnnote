# 1. c++ string

在最新标准的C++中, string 指代 3种数据类型
1. std::basic_string        一个模板类被设计用来处理任何字符的字符串
2. Null-terminated strings  一个由特殊字符 `null` 结尾的字符序列  C 语言的字符串
3. std::basic_string_view (C++17) 一个轻量级的, 不占有空间,只读, 用于表示一个字符串的子序列


字符串实际上是使用 null 字符 `'\0'` 终止的一维字符数组.因此, 一个以 null 结尾的字符串, 包含了组成字符串的字符.

C++ 中，独立的几个 string 对象可以占据也可以不占据各自特定的物理存储区，但是，如果采用引用计数避免了保存同一数据的拷贝副本，那么各个独立的对象（在处理上）必须看起来并表现得就像独占地拥有各自的存储区一样.   

只有当字符串被修改的时候才创建各自的拷贝，这种实现方式称为写时复制（copy-on-write）策略。当字符串只是作为值参数（value parameter）或在其他只读情形下使用，这种方法能够节省时间和空间。



# 2. header <string>

This header is part of the strings library. 

## 2.1. std::char_traits

定义了适用于多种平台下的 字符类型
`template<class CharT> class char_traits;`

## 2.2. std::basic_string

`std::basic_string` generalizes how sequences of characters are manipulated and stored.   
String creation, manipulation, and destruction are all handled by a `convenient set of class methods` and `related functions`.   

```cpp
template<
    class CharT,
    class Traits = std::char_traits<CharT>,
    class Allocator = std::allocator<CharT>
> class basic_string;

namespace pmr {
    template <
        class CharT, 
        class Traits = std::char_traits<CharT>
     >
    using basic_string = 
        std::basic_string< CharT, Traits,std::polymorphic_allocator<CharT>>
} // Since C++17


```

常用的字符串类型
| Type                           | Definition                         |
| ------------------------------ | ---------------------------------- |
| `std::string                 ` | `std::basic_string<char>    `      |
| `std::wstring                ` | `std::basic_string<wchar_t> `      |
| `std::u8string (since C++20) ` | `std::basic_string<char8_t> `      |
| `std::u16string (since C++11)` | `std::basic_string<char16_t>`      |
| `std::u32string (since C++11)` | `std::basic_string<char32_t>`      |
| pmr 类型                       |                                    |
| `std::pmr::string (C++17) `    | `std::pmr::basic_string<char>`     |
| `std::pmr::wstring (C++17) `   | `std::pmr::basic_string<wchar_t>`  |
| `std::pmr::u8string (C++20) `  | `std::pmr::basic_string<char8_t>`  |
| `std::pmr::u16string (C++17) ` | `std::pmr::basic_string<char16_t>` |
| `std::pmr::u32string (C++17) ` | `std::pmr::basic_string<char32_t>` |
除此之外该10个类型还有分别的 std::hash 版本  


## 2.3. 元素访问成员函数

* at
* operator[]
* front  C++11
* back   C++11
* data
* c_str
* operator basic_string_view : (C++17) returns a non-modifiable string_view into the entire string
除了最后一个都很熟悉


## 2.4. 操作函数


同容器一样, string 类型也有多种类似的访问数据的方法

```cpp
//访问字符串中的字符

//string 可以通过下标来访问字符串中的字符
for(int i=0,len=s.length(); i<len; i++){
    cout<<s[i]<<" ";
}

// 通过 at 函数 越界时会抛出异常
try {
  // throw, even if capacity allowed to access element
  s.at(3) = 'x';
}
catch (std::out_of_range const& exc) {
  std::cout << exc.what() << '\n';
}

// 通过 front 和 back C++11
std::string s("Exemplary");
char& f = s.front();


// 通过 data()
data() + i == std::addressof(operator[](i)) for every i in [0, size()]. //(since C++11)
```

**字符串拼接**  使用 `+` 和 `+=`  
1.  两边可以都是 string 字符串
2.  一个 string 字符串和一个C风格的字符串
3.  一个 string 字符串和一个字符数组
4.  一个 string 字符串和一个单独的字符


## 2.5. 容量函数

注意 string 是有 size 和 length 两个函数的  

## 2.6. 构造函数

```cpp
//----------定义:
std::string s2 = "c plus plus";
// 初始化为sssss
std::string s4 (5, 's');

//----转为C风格字符指针
//返回该字符串的 const 指针（const char*）
string path = "D:\\demo.txt";
FILE *fp = fopen(path.c_str(), "rt");


//-----流输入
//使用流对字符串进行输入的时候, 会把空格作为输入结束  
string s;
cin>>s;
```

	

## 2.7. char 转变成字符串的方法 

char 不能直接转成 string

```cpp
char c='A'

// 构造函数
string s1(1,c);

// string 
stringstream ss;
string s2;
ss << c;
ss >> s2;

// 成员函数 push_back
string s3;
s3.push_back(c);

// += 运算符
string s4;
s4+=c;

// = 运算符
string s5=c;

// 成员函数 append
string s6;
s6.append(1,c);

// 成员函数 assign
string s7;
s7.assign(1,c);

// 成员函数 insert
string s8;
s8.insert(0,1,c);

// 成员函数 replace
s9.replace(0,1,1,c);

// c_str 转换
const char* str=&c;
string s10(str,1);
```


## 2.8. 关于子字符串的操作
```cpp
//获得子函数, pos 开始位置, 默认截取到字符串尾
string substr (size_t pos = 0, size_t len = npos) const;

//删除字符串中的一段 , 不用担心越界
string& erase (size_t pos = 0, size_t len = npos);

//插入字符串, 在指定位置插入另一个字符串
//第一个参入要注意不能越界, 否则会抛出异常
string& insert (size_t pos, const string& str);
```
## 2.9. 查找

```cpp
//查询函数, pos 表开始位置, 无视pos之前的匹配项
//返回值为第一个匹配项出现的位置, 若无则返回 string::npos
size_t find (const string& str, size_t pos = 0) const;
size_t find (const char* s, size_t pos = 0) const;


//rfind() 函数 和find的第二个参数功能相反
//第二个参数表示最多查找到第二个参数处, 若无则返回无穷大值
int index = s1.rfind(s2,6);

//查找子字符串和字符串共同具有的字符在字符串中首次出现的位置
//注意是任意一个共有的字符首次出现的位置
constexpr size_type find_first_of( const basic_string& str,size_type pos = 0 ) const noexcept;


// 其他函数
find
rfind
find_first_of
find_first_not_of
find_last_of
find_last_not_of

```
# 3. header <cstring> "string.h"

以下函数为c原生函数,包含在`<string.h>`头文件中, 在C++中也在 `<cstring>`中  
使用`<string.h>` 不需要在函数名前加`std::`  
| 函数名                   | 功能                                                                                                     |
| ------------------------ | -------------------------------------------------------------------------------------------------------- |
| 字符串操作函数           |                                                                                                          |
| `strcpy(s1, s2);`        | 复制字符串 s2 到字符串 s1。                                                                              |
| `strncpy(s1, s2, n);`    | 复制s1特定数量的字符到s2,若n大于s1的长度,不足的部分补`\0`,若n小于s1的长度,则s2是一个非`\0`结尾的字符数组 |
| `strcat(s1, s2);`        | 连接字符串 s2 到字符串 s1 的末尾。                                                                       |
| `strncat(s1, s2, n);`    | 连接字符串 s2 的 n 个字符 到字符串 s1 的末尾。                                                           |
| `strtok(str, delim)`     | 根据字符delim分隔字符串str,该函数返回被分解的第一个子字符串，如果没有可检索的字符串，则返回一个`空指针`  |
| 字符串验证函数           |                                                                                                          |
| `strlen(s1);    `        | 返回字符串 s1 的长度。                                                                                   |
| `strcmp(s1, s2);`        | 如果 s1 和 s2 是相同的，则返回 0；如果 s1\<s2 则返回值小于 0；如果 s1\>s2 则返回值大于 0。               |
| `strncmp(s1, s2);`       | 如果 s1 和 s2 的前n个字符是相同的，则返回 0；如果 s1\<s2 则返回值小于 0；如果 s1\>s2 则返回值大于 0。    |
| `strcoll(s1,s2)`     \*  | 功能和来strcmp类似,用法也一样,但是strcoll()会依环境变量LC_COLLATE所指定的文字排列次序来比较              |
| 字符串查找函数           |                                                                                                          |
| `strchr(s1, ch);`        | 返回一个char指针，指向字符串 s1 中字符 ch 的第一次出现的位置。                                           |
| `strrchr(s1, ch);`       | 返回一个char指针，指向字符串 s1 中字符 ch 的最后一次出现的位置。                                         |
| `strstr(s1, s2);`        | 返回一个指针，指向字符串 s1 中字符串 s2 的第一次出现的位置。                                             |
| 字符串字典查找函数       |                                                                                                          |
| `strspn(dest,scr)`       | scr指向的字符串作为合法字典,返回一个长度值,dest从起始开始的合法字符的长度                                |
| `strcspn(dest,scr)`      | scr指向的字符串作为不合法字符字典,返回一个长度值,dest从起始开始出现不合法字符的第一个位置度              |
| `strpbrk(dest,breakset)` | breakset作为终止字典, 返回dest第一次出现终止字符的位,没找到就返回`NULL`                                  |
| 内存操作函数             |                                                                                                          |
| `memchr (ptr,ch,count)`  | 找字符第一次出现的位置, 没找到就返回`NULL`                                                               |
| `memcmp (lhs,rhs,count)` | 与字符串比较类似                                                                                         |
| `memset (dest,ch,count)` | 注意 `int ch`                                                                                            |
| `memcpy (dest,scr,c)`    | copies one buffer to another                                                                             |
| `memmove(dest,scr,c)`    | 函数的功能同memcpy基本一致，但是memmove先创建temp内存,当src区域和dst内存区域重叠时, 可以安全拷贝         |

* 若LC_COLLATE为"POSIX"或"C"，则strcoll()与strcmp()作用完全相同 
* 按照 C94 及 C99 标准的规定，程序在启动时设置 locale 为 "C"。在 "C" locale 下，字符串的比较就是按照内码一个字节一个字节地进行，这时 strcoll 与 strcmp 函数没有区别。在其他 locale 下，字符串的比较方式则不同了，例如在简体中文 locale 下，strcmp 仍然按内码比较，而 strcoll 对于汉字则是按拼音进行的（这也跟操作系统有关，Windows 还支持按笔划排序，可以在“区域和语言设置”里面修改
  
## 3.1. string 与数字类型的转换

### 3.1.1. 字符串转为数字

除了使用C语言风格的函数 `ato*()` 和 `strto*` C++风格的函数为 `sto*()`

```cpp
#include <string> //string 头文件

// idx不是空指针，则该函数还将idx的值设置为该数字后str中第一个字符的位置。
int stoi (const string&  str, size_t* idx = 0, int base = 10);
long stol (const string&  str, size_t* idx = 0, int base = 10);
unsigned long stoul (const string&  str, size_t* idx = 0, int base = 10);
long long stoll (const string&  str, size_t* idx = 0, int base = 10);
unsigned long long stoull (const string&  str, size_t* idx = 0, int base = 10);
float stof (const string&  str, size_t* idx = 0);
double stod (const string&  str, size_t* idx = 0);
long double stold (const string&  str, size_t* idx = 0);

```
### 3.1.2. 数字转成字符串

```cpp
//  to_string 函数。将 val 解释为 string，并返回转换结果
string to_string (int val);
string to_string (long val);
string to_string (long long val);
string to_string (unsigned val);
string to_string (unsigned long val);
string to_string (unsigned long long val);
string to_string (float val);
string to_string (double val);
string to_string (long double val);

// 同样的参数还有 to_wstring()
wstring to_wstring (int val);


```
