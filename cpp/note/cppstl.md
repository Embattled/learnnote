# C++ 的文件读写

iostream 标准库,它提供了 cin 和 cout 方法分别用于从标准输入读取流和向标准输出写入流.  
从文件读取流和向文件写入流,这就需要用到 C++ 中另一个标准库 `fstream`  

| 数据类型 | 描述                                                                                                                         |
| -------- | ---------------------------------------------------------------------------------------------------------------------------- |
| ofstream | 该数据类型表示输出文件流, 用于创建文件并向文件写入信息。                                                                     |
| ifstream | 该数据类型表示输入文件流, 用于从文件读取信息。                                                                               |
| fstream  | 该数据类型通常表示文件流, 且同时具有 ofstream 和 ifstream 两种功能, 这意味着它可以创建文件, 向文件写入信息, 从文件读取信息。 |


要在 C++ 中进行文件处理, 必须在 C++ 源代码文件中包含头文件 `iostream` 和 `fstream`

## 1. 文件打开与关闭

`ofstream` 和 `fstream` 对象都可以用来打开文件进行***写操作***, 如果**只需要打开文件进行读操作**, 则使用 `ifstream` 对象

```C++
//打开文件时可以先定义对象再调用函数,也可以直接在定义对象时打开
myfile.open(const char *filename, ios::openmode mode);
ofstream myfile ("example.bin", ios::out | ios::app | ios::binary);
//关闭文件时使用 close 方法
myfile.close();
```
| 模式标志    | 描述                                                                   |
| ----------- | ---------------------------------------------------------------------- |
| ios::app    | 追加模式。所有写入都追加到文件末尾。                                   |
| ios::ate    | 文件打开后定位到文件末尾。                                             |
| ios::in     | 打开文件用于读取。                                                     |
| ios::out    | 打开文件用于写入。                                                     |
| ios::trunc  | 如果该文件已经存在, 其内容将在打开文件之前被截断, 即把文件长度设为 0。 |
| ios::binary | 二进制模式打开.                                                        |

## 2. 文件读写

**对于字符文件**  
对`ofstream` 或 `fstream` 对象,使用流插入运算符`（ << ）`向文件写入信息  
对`ifstream` 或 `fstream` 对象使用流提取运算符`（ >> ）`从文件读取信息  
还有类似于getline();

**对于二进制文件**
```cpp
write ( memory_block, size );
read ( memory_block, size );
//memory_block 是一个 char* 用于指向读取到内存的地址或写出到文件的内容源,size是文件块的大小,可以传入 streampos 类型
```

## 3. 检查函数
每一个流对象都有一个 `flag` 用于保存操作时的各种状态  
使用 `clear()` 来清除`flag`

对特定状态的检查函数,返回值都是布尔类型
|        |                                                          |                                                                  |
| ------ | -------------------------------------------------------- | ---------------------------------------------------------------- |
| bad()  | Returns true 如果有读写失败                              | 例如对一个没有以写入标志打开的流执行写入或者写入的磁盘已没有空间 |
| fail() | Returns true 在`bad()`的基础上检查格式问题               | 例如文件读出来的是字符但是传输给了一个整数变量                   |
| eof()  | 检查是否到了文件末尾.                                    |
| good() | 最常用的函数, 对上面所有函数返回`true`的时候,返回`false` | `good()`与`bad()`不是对立函数,good一次检查更多的flag             |
---

## 4. 文件位置指针操作

`tellg() and tellp()`  返回输入或者输出流当前的文件位置指针,返回值的类型是`streampos`

`seekg() and seekp()`  istream 的 seekg 和 ostream 的 seekp  
可以设定输入位置指针,有两个重载  

```cpp
seekg ( streampos ); 
seekp ( streampos ); //直接设定位置

//offset是streamoff类型, 是一个 long long 整型
seekg ( offset, direction );
seekp ( offset, direction );//根据偏移方向设定指针位置
```
| `direction` | 查找方向                                        |
| ----------- | ----------------------------------------------- |
| ios::beg    | offset counted from the beginning of the stream |
| ios::cur    | offset counted from the current position        |
| ios::end    | offset counted from the end of the stream       |

下面的程序可以获得一个文件的大小 `bytes` 形式  
注意用来表示文件大小的`size`是`streampos size;` 使用这个类型可以安全的加减以及转换为数字  

```cpp
// obtaining file size
#include <iostream>
#include <fstream>
using namespace std;

int main () {
  streampos begin,end;
  ifstream myfile ("example.bin", ios::binary);
  begin = myfile.tellg();
  myfile.seekg (0, ios::end);
  end = myfile.tellg();
  myfile.close();
  cout << "size is: " << (end-begin) << " bytes.\n";
  return 0;
```



# string 字符串操作

字符串实际上是使用 null 字符 `'\0'` 终止的一维字符数组.因此, 一个以 null 结尾的字符串, 包含了组成字符串的字符.

## **基础的字符串操作**  

以下函数为c++原生函数,不需要包含 string 头文件  
| 函数名          | 功能                                                                                     |
| --------------- | ---------------------------------------------------------------------------------------- |
| `strcpy(s1, s2);` | 复制字符串 s2 到字符串 s1。                                                              |
| `strcat(s1, s2);` | 连接字符串 s2 到字符串 s1 的末尾。                                                       |
| `strlen(s1);    ` | 返回字符串 s1 的长度。                                                                   |
| `strcmp(s1, s2);` | 如果 s1 和 s2 是相同的，则返回 0；如果 s1<s2 则返回值小于 0；如果 s1>s2 则返回值大于 0。 |
| `strchr(s1, ch);` | 返回一个指针，指向字符串 s1 中字符 ch 的第一次出现的位置。                               |
| `strstr(s1, s2);` | 返回一个指针，指向字符串 s1 中字符串 s2 的第一次出现的位置。                             |

## **string 类的函数**

**基础8种迭代函数**  
返回对应的迭代器
```cpp
.begin();  //开头
.end();    //末尾
.r*();     //反向迭代器
.c*();     //c++ 11 新标准,const 迭代器,防止更改字符串内容
.cr*();    //顺序为cr
```

**实用操作函数**
```cpp
//获得子函数, pos 开始位置, 默认截取到字符串尾
string substr (size_t pos = 0, size_t len = npos) const;

//查询函数, pos 表开始位置, 无视pos之前的匹配项
//返回值为第一个匹配项出现的位置, 若无则返回 string::npos
size_t find (const string& str, size_t pos = 0) const;


```



