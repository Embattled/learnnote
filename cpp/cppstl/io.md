# 1. C 语言的流

C语言的标准输入输出和文件输入输出都在 `stdio.h` 中  
在C++ 环境下可以使用 `<cstdio>` 使用 C++ 风格的头文件会将所有函数放入 std 命名空间  


## 1.1. 类型 type

### 1.1.1. FILE

object type, capable of holding all information needed to control a C I/O stream   
提供了控制一个 C流所需要的所有信息  

### 1.1.2. fpos_t

complete non-array object type, capable of uniquely specifying a position in a file, including its multibyte parse state   
能够唯一的定位一个文件中的位置  

## 1.2. 常量

`EOF`       : 用来表示已经到达文件结束的负整数,在读写时发生错误也会返回这个宏值  
`FOPEN_MAX` : 用来表示该系统中可以同时打开的文件个数  


## 1.3. File access 文件访问  


### 1.3.1. fopen 文件打开

控制流打开返回一个 `FILE` 对象, 创建一个新的文件或者打开一个已有的文件  
`FILE *fopen( const char * filename, const char * mode );`  
访问模式 mode 的值可以是下列值中的一个  
| 模式 | 描述                                                                 |
| ---- | -------------------------------------------------------------------- |
| r    | 只读                                                                 |
| w    | 只写,从头开始写,如果文件已存在,则会被从开头覆盖,不存在则会创建新文件 |
| a    | 追加写,不存在会创建新文件                                            |
| +    | 代表可读写,追加在上面模式之后                                        |
| b    | 代表二进制读写,追加在上面模式之后                                    |

如果错误则返回 null pointer

```cpp
FILE* fp = std::fopen("test.txt", "r");
if(!fp) {
    std::perror("File opening failed");
    return EXIT_FAILURE;
}
```

### 1.3.2. fclose 关闭文件

`int fclose( FILE *fp );`    
关闭流stream,会清空缓冲区中的数据,关闭文件,并释放用于该文件的所有内存  

* 成功关闭文件, 返回 `0` 
* 关闭错误, 返回 `EOF`

### 1.3.3. 文件增删改

创建一个文件用文件流打开的函数 `fopen` 即可完成

删除文件  
`int remove(const char *filename)`  

重命名或者移动文件  
`int rename(const char *old_filename, const char *new_filename)`  

这两个函数如果成功,则返回零。如果错误,则返回 -1,并设置 errno  

## 1.4. Unformatted io

`int fputc(int char, FILE *stream)`  
把参数 char 指定的字符（一个无符号字符）写入到指定的流 stream 中,并把位置标识符往前移动  
`int fputs(const char *str, FILE *stream)`  
把字符串写入到指定的流 stream 中,但不包括空字符,把一个以 null 结尾的字符串写入到流中  

`int fgetc( FILE * fp );`  
fgetc() 函数从 fp 所指向的输入文件中读取一个字符。返回值是读取的字符,如果发生错误则返回 EOF

`char *fgets( char *buf, int n, FILE *fp );`   
从输入流中读入 ***n - 1*** 个字符,并在最后追加一个 null 字符来终止字符串, 总计 n 个字符  
如果这个函数在读取最后一个字符之前就遇到一个换行符 '\n' 或文件的末尾 EOF,则只会返回读取到的字符,包括换行符

## 1.5. Formatted io

`int fscanf(FILE *stream, const char *format, ...)`  

`int fprintf(FILE *stream, const char *format, ...)`

与普通的 `print scanf` 类似的输入,只不过前面加入了文件流参数  


## 1.6. Direct io 二进制读写

用于存储块的读写 - 通常是数组或结构体
```cpp
size_t fread(void *ptr, size_t size_of_elements, size_t number_of_elements, FILE *a_file);
              
size_t fwrite(const void *ptr, size_t size_of_elements, size_t number_of_elements, FILE *a_file);
```
元素**总数**会以 size_t 对象返回, 如果与输入的元素总数不一致,则可能是遇到了文件结尾  




### 1.6.1. 1.3 C 文件流的指针操作

**判断文件指针是否到末尾**
`int feof(FILE *stream)`  当已经读到末尾时返回一个非零值  
`if( feof ( FILE ) ) break;`  用来跳出读取  


# 2. C++的流 Input/Output

## 2.1. fstream C++ 的文件读写流 

iostream 标准库,它提供了 cin 和 cout 方法分别用于从标准输入读取流和向标准输出写入流.  
从文件读取流和向文件写入流,这就需要用到 C++ 中另一个标准库 `fstream`  

| 数据类型 | 描述                                                                                                                         |
| -------- | ---------------------------------------------------------------------------------------------------------------------------- |
| ofstream | 该数据类型表示输出文件流, 用于创建文件并向文件写入信息。                                                                     |
| ifstream | 该数据类型表示输入文件流, 用于从文件读取信息。                                                                               |
| fstream  | 该数据类型通常表示文件流, 且同时具有 ofstream 和 ifstream 两种功能, 这意味着它可以创建文件, 向文件写入信息, 从文件读取信息。 |


要在 C++ 中进行文件处理, 必须在 C++ 源代码文件中包含头文件 `iostream` 和 `fstream`

### 2.1.1. 文件打开与关闭

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

### 2.1.2. 文件读写

**对于字符文件**  
对`ofstream` 或 `fstream` 对象,使用流插入运算符`（ << ）`向文件写入信息  
对`ifstream` 或 `fstream` 对象使用流提取运算符`（ >> ）`从文件读取信息  
还有类似于getline(); 读取一整行的函数

**对于二进制文件**
```cpp
write ( memory_block, size );
read ( memory_block, size );
//memory_block 是一个 char* 用于指向读取到内存的地址或写出到文件的内容源,size是文件块的大小,可以传入 streampos 类型
```

**基础读写**
```cpp
get();  //extracts characters 
unget(); //Makes the most recently extracted character available again. 
putback(char);// Puts the character ch back to the input stream so the next extracted character will be ch. 
//功能是一样的，最大的区别在参数上 unget没有参数，是把已经从流读取出来的那个字符放回去，下次读取的时候可以读到这个字符 而putback是把参数c放入流中
peek(); //reads the next character without extracting it 
```
### 2.1.3. 检查函数
每一个流对象都有一个 `flag` 用于保存操作时的各种状态  
使用 `clear()` 来清除`flag`

对特定状态的检查函数,返回值都是布尔类型
|        |                                            |                                                                  |
| ------ | ------------------------------------------ | ---------------------------------------------------------------- |
| bad()  | Returns true 如果有读写失败                | 例如对一个没有以写入标志打开的流执行写入或者写入的磁盘已没有空间 |
| fail() | Returns true 在`bad()`的基础上检查格式问题 | 例如文件读出来的是字符但是传输给了一个整数变量                   |
| eof()  | 检查是否到了文件末尾.                      |
| 1.3.   | 1.3.                                       | 1.3.                                                             | 1.2. | 1.2. | 1.2. | 1.2. | 1.2. | 1.2. | 1.2. | 1.2. | 1.2. | 1.2. | 1.2. | 1.2. | 1.2. | 1.2. | 1.2. | 1.2. | 1.4. | 1.4. | 1.4. | 1.4. | 1.4. | 1.4. | 1.4. | 1.4. | 1.4. | 1.4. | 1.4. | 1.4. | 1.4. | 1.4. | 1.4. | 1.4. | 1.4. | 1.4. | 1.4. | good() | 最常用的函数, 对上面所有函数返回`true`的时候,返回`false` | `good()`与`bad()`不是对立函数,good一次检查更多的flag |

### 2.1.4. 文件位置指针操作
tellp() —— 与tellg()有同样的功能，但它用于写文件时。
* 当我们读取一个文件，并要知道内置指针的当前位置时，应该使用tellg()
* 当我们写入一个文件，并要知道内置指针的当前位置时，应该使用tellp()


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

## 2.2. sstream 字符串流 

头文件 `<sstream>`  
* stringstream  同时可以支持C风格的串流的输入输出操作 stringstream则是从  `iostream` (输入输出流类)和和 `stringstreambase` （c++字符串流基类）派生而来
* ostringstream 输出        是从ostream（输出流类）和stringstreambase（c++字符串流基类）派生而来
* istringstream 输入        是从istream（输入流类）和stringstreambase（c++字符串流基类）派生而来

事实上,在C++有两种字符串流，一种在`sstream`中定义，另一种在`strstream`中定义。它们实现的东西基本一样。
```cpp
//strstream里包含
class strstreambuf;
class istrstream;
class ostrstream;
class strstream;

它们是基于C类型字符串char*编写的

//sstream中包含
class istringstream;
class ostringstream;
class stringbuf;
class stringstream;
它们是基于std::string编写的

ostrstream::str();//返回的是char*类型的字符串
ostringstream::str();//返回的是std::string类型的字符串
```
在使用的时候要注意到二者的区别，一般情况下推荐使用std::string类型的字符串
当然如果为了保持和C的兼容，使用strstream也是不错的选择  


### 2.2.1. 从string中读取字符

stringstream对象可以绑定一行字符串，然后以空格为分隔符把该行分隔开来
```cpp
// 构造函数
istringstream::istringstream(string str);

// 建立一个字符串
std::string str = "I am coding ...";
// 绑定字符串到流
std::istringstream is(str);
do
{
    std::string substr;
    is>>substr;
    std::cout << substr << std::endl;
} while (is);

```
### 2.2.2. 用来进行数据类型转换

传入参数和目标对象的类型会被自动推导出来，所以不存在错误的格式化符的问题。相比c库的数据类型转换，sstream更加安全、自动和直接  

```cpp
stringstream sstream;
string strResult;
int nValue = 1000;

// 将int类型的值放入输入流中
sstream << nValue;
// 从sstream中抽取前面插入的int类型的值，赋给string类型
sstream >> strResult;

```
### 2.2.3. 字符串流的高级操作

* 清空字符串流的方式 `sstream.str("")`  `clear()`
* 使用 `str()` 方法，将stringstream类型转换为string类型
* 将多个字符串放入stringstream，实现字符串的拼接目的

```cpp
// 将多个字符串放入 sstream 中
sstream << "first" << " " << "string,";
sstream << " second string";
cout<< sstream.str() << endl;
// 清空 sstream
sstream.str("");

```
