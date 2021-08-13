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

C语言stdio定义了文件流控制相关的一些常量以及三个标准流
* `NULL`            : C语言好多头文件都定义了, 没啥好说的
* `EOF`             : 用来表示已经到达文件结束的负整数,在读写时发生错误也会返回这个宏值  
* `FOPEN_MAX`       : 用来表示该系统中可以同时打开的文件个数  
* `FILENAME_MAX`    : 用来表示该系统支持的最长的文件名, 可以用该常量来分配存储文件名的字符串空间
* `BUFSIZ`          : `std::setbuf` 使用的缓存大小
* `TMP_MAX`         : 通过 `std::tmpnam` 可以生成的唯一文件名的最大个数, 有点特殊的常量
* `L_tmpnam`        : 通过 `std::tmpnam` 生成的文件名的长度, 用该常量来初始化字符串


### 1.2.1. 标准流

这三个文件流在程序运行的时候就已经被隐式的打开,但是事实上他们是标准输出输入, 并不是文件  

* stdin       fully buffered
* stdout      fully buffered
* stderr      not fully buffered
* 可以通过对这三个流的读写来实现自己定义的 printf 函数
```cpp
#define stdin  /* implementation-defined */
#define stdout /* implementation-defined */
#define stderr /* implementation-defined */


#include <cstdarg>
#include <cstdio>
 
int my_printf(const char * fmt, ...)
{
    std::va_list vl;
    va_start(vl, fmt);
    int ret = std::vfprintf(stdout, fmt, vl);
    va_end(vl);
    return ret;
}
 
int main()
{
    my_printf("Rounding:\t%f %.0f %.32f\n", 1.5, 1.5, 1.3);
    my_printf("Padding:\t%05.2f %.2f %5.2f\n", 1.5, 1.5, 1.5);
    my_printf("Scientific:\t%E %e\n", 1.5, 1.5);
    my_printf("Hexadecimal:\t%a %A\n", 1.5, 1.5);
}
```
### 1.2.2. SEEK 常量

作为 fseek 函数可以接受的第三个输入参数  



## 1.3. 标准输入输出

* scanf   reads formatted input from stdin, a file stream or a buffer 
* printf  prints formatted output to stdout, a file stream or a buffer 

### 1.3.1. 输入 scanf

返回值:
* non-zero  : Number of receiving arguments successfully assigned
* zero      : matching failure occurred before the first receiving argument was assigned
* EOF       : input failure occurs before the first receiving argument was assigned. 
  
```c
// Reads the data from stdin
int scanf( const char* format, ... );

// Reads the data from file stream stream
int fscanf( std::FILE* stream, const char* format, ... );

// Reads the data from null-terminated character string buffer
int sscanf( const char* buffer, const char* format, ... );

```


### 1.3.2. 输出 printf

输出同样有返回值:  
1. 成功时   : 输出的字符个数
2. 出错时   : 负数
3. 对于输出到字符串, null character 不统计在返回值中
4. 对于 snprintf , 成功时的返回值一定是小于 buf_size 的
5. 对于 snprintf , 注意 buf_size 为0 时的特殊返回值
```cpp
// 标准输出  输出到 stdout
int printf( const char* format, ... );

// 输出到文件
int fprintf( std::FILE* stream, const char* format, ... );

// 输出到字符串
int sprintf( char* buffer, const char* format, ... );

// (since C++11) At most buf_size - 1 characters are written
// 指定最大输出大小的 输出到字符串  最后一个字符会设置成 '/0'
// 如果 buf_size 是 0 , 注意
// the return value (number of bytes that would be written not including the null terminator) is still calculated and returned.
int snprintf( char* buffer, std::size_t buf_size, const char* format, ... );


// 可以用 snprintf 来计算必要的缓存大小, 因为对于数字数据来说, 并不确定字符长度
const char *fmt = "sqrt(2) = %f";
int sz = std::snprintf(nullptr, 0, fmt, std::sqrt(2));
// note +1 for null terminator
std::vector<char> buf(sz + 1); 
std::snprintf(&buf[0], buf.size(), fmt, std::sqrt(2));
```

### 1.3.3. variable argument list



## 1.4. format string

a null-terminated character string specifying how to read the input.
格式化输入输出的核心  

由三部分组成
1. 非空格的所有字符(除了`%`)
2. 空格字符 包括 `'\n', ' ', '\t' `
   * 在scanf中, format string 的空格' '可以接受并消耗流中所有连续的空白字符
3. conversion specifications  输入时和输出时的可选参数是不同的

### 1.4.1. scanf
conversion specifications 
   * introductory % character 
   * (可选) assignment-suppressing character *
   * (可选)integer number (greater than zero) , 定义输入或输出最大位宽, 输入时定义该转义符在进行数据转换时可以消耗掉的流中字符的最大值
   * (可选)length modifier that specifies the size of the receiving argument
   * conversion format specifier 
   
### 1.4.2. printf

## 1.5. conversion specifiers


空格字符的特殊化处理示例
```cpp
std::scanf("%d", &a);
std::scanf("%d", &b);
/* 
    对于两个连续的整数输入, 输入时两个整数可以在不同行输入或者隔一个空格(tab也算)输入
    这是因为 %d 会消耗掉前方所有的空白字符, 直到接收到整数或者不匹配的字符报错
*/

std::scanf("%d", &a);
std::scanf(" %c", &c);
/* 
    对于其他不会消耗空白字符的 conversion specifiers, 比如说 %c
    可以在 %c 前面加一个空格, 利用该空格消耗掉流中前部的所有连续空白字符, 确保正确接收到
    如果不加的化 c 会直接接收到 %d 遗留下来的换行符
*/

```


## 1.6. File access 文件访问  


### 1.6.1. fopen 文件打开

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

### 1.6.2. fclose 关闭文件

`int fclose( FILE *fp );`    
关闭流stream,会清空缓冲区中的数据,关闭文件,并释放用于该文件的所有内存  

* 成功关闭文件, 返回 `0` 
* 关闭错误, 返回 `EOF`

## 1.7. File operation

分两部分
1. 删除和重命名文件 `remove rename`
   * stl 中 algorithm 头文件也有 remove 函数 , 不过不是同一个东西
2. 自动生成的文件名 `tmpfile tmpnam`
   * `tmpfile` 直接就返回了一个文件流, 包含了 tmpnam 的功能
   * `tmpnam`  只是返回了自动生成的文件名 **文件名里包含了 /tmp/ 路径**
```cpp
// 删除一个文件
int remove(const char *filename);
// 如果要删除的文件正在被使用
// POSIX systems unlink the file name
// Windows does not allow the file to be deleted
// return 0 表示成功,  非0 表示失败

// 重命名一个文件
int rename(const char *old_filename, const char *new_filename)
// return 0 表示成功,  非0 表示错误
// If new_filename exists, the behavior is implementation-defined. 


// Creates a unique filename that does not name a currently existing file
char* tmpnam( char* filename );
// return value: 
// 如果 已经没有文件名可以生成, 返回 NULL
// 如果 指针filename 是空指针, 则生成的文件名作为返回值返回, 是一个内部的静态缓存地址
// 如果 指针filename 不是空指针至少且有 L_tmpnam 大小, 文件名就存在filename 且返回值就是filename

// --------------例子
std::string name1 = std::tmpnam(nullptr);
// /tmp/fileDjwifs   文件名示例


// Creates and opens a temporary file with a unique auto-generated filename. 
std::FILE* tmpfile();
// 打开方式: by std::fopen with access mode "wb+"
// 可能生成的文件名当然是和 tmpnam 共享的
// 返回值 : 失败时返回 NULL 
// Linux系统下  : 因为是 tmp file , 所以该文件不能和其他程序或者进程共享
// Windows下    : 该函数需要申请文件读写权限 
	
// --------------例子
// Linux-specific method to display the tmpfile name
#include <filesystem>
namespace fs = std::filesystem;
std::FILE* tmpf = std::tmpfile();
std::cout << fs::read_symlink(
                fs::path("/proc/self/fd") / std::to_string(fileno(tmpf))
            ) << '\n';

```

## 1.8. File positionin

文件读取过程中读头位置相关函数  
1. ftell        : 返回值获取当前位置, long 类型
2. fseek        : long 类型指定文件读头位置
3. fgetpos      : 指针写入获取当前位置, fpos_t 类型
4. 
5. rewind       : 文件读头返回到文件开始


```cpp
// 如果文件是二进制打开的, 该值表示  the number of bytes from the beginning of the file. 
// 如果是文本模式打开的, 则没有任何实际意义, 只能作为参数输入到 std::fseek
long ftell( std::FILE* stream );
// return -1L if failure occurs. Also sets errno on failure


// origin : position to which offset is added. 
// 是一个预定义量 包括  SEEK_SET, SEEK_CUR, SEEK_END
int fseek( std::FILE* stream, long offset, int origin );


// pos : pointer to a fpos_t object to store the file position indicator to 
// return 0 upon success, nonzero value otherwise. Also sets errno on failure. 
// 注意这个 pos 的值只能作为参数输入 std::fsetpos
int fgetpos( std::FILE* stream, std::fpos_t* pos );


// pos 只能是 fgetpos 获取的, 只能作用于相同的文件上  
int fsetpos( std::FILE* stream, const std::fpos_t* pos );


// 最简单的函数
// 相当于  std::fseek(stream, 0, SEEK_SET);
// 而且会清除文件结尾和错误的 符号位
void rewind( std::FILE* stream );
	
```
## 1.9. error handling

文件读写中会发生很多错误, 比如文件读取到末尾就算其中之一  



## 1.10. Unformatted io

`int fputc(int char, FILE *stream)`  
把参数 char 指定的字符（一个无符号字符）写入到指定的流 stream 中,并把位置标识符往前移动  
`int fputs(const char *str, FILE *stream)`  
把字符串写入到指定的流 stream 中,但不包括空字符,把一个以 null 结尾的字符串写入到流中  

`int fgetc( FILE * fp );`  
fgetc() 函数从 fp 所指向的输入文件中读取一个字符。返回值是读取的字符,如果发生错误则返回 EOF

`char *fgets( char *buf, int n, FILE *fp );`   
从输入流中读入 ***n - 1*** 个字符,并在最后追加一个 null 字符来终止字符串, 总计 n 个字符  
如果这个函数在读取最后一个字符之前就遇到一个换行符 '\n' 或文件的末尾 EOF,则只会返回读取到的字符,包括换行符



## 1.11. Direct io 二进制读写

用于存储块的读写 - 通常是数组或结构体
```cpp
size_t fread(void *ptr, size_t size_of_elements, size_t number_of_elements, FILE *a_file);
              
size_t fwrite(const void *ptr, size_t size_of_elements, size_t number_of_elements, FILE *a_file);
```
元素**总数**会以 size_t 对象返回, 如果与输入的元素总数不一致,则可能是遇到了文件结尾  




### 1.11.1. 1.3 C 文件流的指针操作

**判断文件指针是否到末尾**
`int feof(FILE *stream)`  当已经读到末尾时返回一个非零值  
`if( feof ( FILE ) ) break;`  用来跳出读取  


# 2. C++的流 Input/Output

* C++的流以及流控制分散在了多个头文件中, 而且有分级的继承关系  
* C++流的操作是将文件读写, 内存读写以及其他输出抽象成虚拟设备, 可以用同一套代码进行输入输出  


类的层级结构:
* ios_base
  * basic_ios
    * basic_ostream
      * basic_ostringstream
      * basic_ofstream
    * basic_istream
      * basic_istringstream
      * basic_ifstream
    * 二者组合的 basic_iostream
      * basic_stringstream
      * basic_fstream

# 3. 基础层 <ios>

* 定义了 C++ 流的所有基类 ios 和 basic_ios
* 该头文件的类函数基本适用于所有C++流对象

## 3.1. std::ios_base

**所有** io 流类的基类, 保存的数据有:
1. 流的状态, stream status flags
2. 控制信息, flags that control formatting of both input and output sequences and the imbued locale
3. 私有存储, 是一个可拓展索引的指针, 支持 long and void* members, 可用来存储流的私有数据
4. callbacks: arbitrary number of user-defined functions to be called from `imbue(), copyfmt(), and ~ios_base()`

### 3.1.1. flags成员变量

四个基础控制flags变量, 都是 bitmap 格式  

1. `std::ios_base::openmode`  available file open flags
   BitmaskType, 可以进行或运算
     * app    追加, 在写入操作执行之前将文件指针放到文件末尾
     * ate    追加, 打开文件时立即将文件指针放到文件末尾
     * binary 二进制模式
     * in     允许读
     * out    允许写

2. `std::ios_base::fmtflags`  available formatting flags
   同样是 BitmaskType 
     * dec          use decimal base for integer I/O: see std::dec
     * oct          use octal base for integer I/O: see std::oct
     * hex          use hexadecimal base for integer I/O: see std::hex
     * basefield 	dec|oct|hex. Useful for masking operations
     * left         left adjustment (adds fill characters to the right): see std::left
     * right        right adjustment (adds fill characters to the left): see std::right
     * internal 	internal adjustment (adds fill characters to the internal designated point): see std::internal
     * adjustfield 	left|right|internal. Useful for masking operations
     * `scientific` 	generate floating point types using scientific notation, or hex notation if combined with fixed: see std::scientific
     * `fixed` 	    generate floating point types using fixed notation, or hex notation if combined with scientific: see std::fixed
     * floatfield 	scientific|fixed. Useful for masking operations
     * boolalpha 	insert and extract bool type in alphanumeric format: see std::boolalpha
     * showbase 	generate a prefix indicating the numeric base for integer output, require the currency indicator in monetary I/O: see std::showbase
     * showpoint 	generate a decimal-point character unconditionally for floating-point number output: see std::showpoint
     * showpos 	    generate a + character for non-negative numeric output: see std::showpos
     * skipws 	    skip leading whitespace before certain input operations: see std::skipws
     * unitbuf 	    flush the output after each output operation: see std::unitbuf
     * uppercase 	replace certain lowercase letters with their uppercase , equivalents  in certain output operations: see std::uppercase

3. `std::ios_base::iostate`  stream state flags.
   BitmaskType 总共就4种状态  但是相关的测试函数没有定义在 `ios_base` 中  
    * goodbit
    * badbit
    * failbit
    * eofbit
    * 详细的情况说明定义在了文档中(链接)[https://en.cppreference.com/w/cpp/io/ios_base/iostate]

4. `std::ios_base::seekdir` file seeking direction type
   专门用来指定 seekg 和 seekp 的搜索方法
   * beg 从流的开头搜索
   * end 从末尾开始搜索
   * cur 从当前位置开始搜索 

### 3.1.2. format flags 操作函数

1. .setf
2. .flags
3. .unsetf
```cpp
// std::ios_base::flags
// 返回当前的 flags
fmtflags flags() const;  
// 替代的方式设置新 flags
fmtflags flags( fmtflags flags ); 

// std::ios_base::setf
// 在现有格式fl上添加参数 flags , fl = fl | flags
fmtflags setf( fmtflags flags ); 
// 用mask选定更新的位  fl = (fl & ~mask) | (flags & mask)
fmtflags setf( fmtflags flags, fmtflags mask );

// std::ios_base::unsetf
// 清除 flags 所定义的所有位, 反向赋值
void unsetf( fmtflags flags );
	

int num = 150;
// flags的值可以使用所有子类的域, 还可以使用对象
std::cout.setf(std::ios_base::hex, std::ios_base::basefield);
std::cout.setf (std::ios::hex , std::ios::basefield);
std::cout.setf(std::cout.hex, std::cout.basefield);

// using fmtflags type  获取 fmtflags 修改后再赋值进去
std::ios_base::fmtflags ff;
ff = std::cout.flags();
ff &= ~std::cout.basefield;   // unset basefield bits
ff |= std::cout.hex;          // set hex
ff |= std::cout.showbase;     // set showbase
std::cout.flags(ff);
```

### 3.1.3. width precision 宽度和精度控制

控制精度和宽度
* The default precision, as established by std::basic_ios::init, is 6
* width 是严格控制最小和最大位宽的设置  

```cpp
// std::ios_base::precision
// 返回当前输出的浮点数精度
streamsize precision() const;
// 设置浮点数精度, 返回旧的精度
streamsize precision( streamsize new_precision );

// std::ios_base::width
// 返回当前的位宽
streamsize width() const;
// 设置新位宽, 返回旧位宽
streamsize width( streamsize new_width );


double d = 1.2345678901234;
std::cout << std::cout.precision()<<endl; // 6
std::cout<< d <<endl; // 1.23457
std::cout.precision(12);
std::cout<< d <<endl; // 23456789012
```

### 3.1.4. 地区?字符集 设置 imbue

* imbue   设置 locale
* getloc  返回当前 locale

这个部分不太懂, 和宽字符有关  
locale 要参照 `<locale>` 头文件  

```cpp
// std::ios_base::imbue
// 设置新 locale , 返回旧的 locale
std::locale imbue( const std::locale& loc );

// std::ios_base::getloc
// 返回当前 locale
std::locale getloc() const;
```


## 3.2. std::basic_ios 公有继承 ios_base

```cpp
template<
    class CharT,
    class Traits = std::char_traits<CharT>
> class basic_ios : public std::ios_base
```
* basic_ios 提供了和 std::basic_streambuf 对象的接口  
* 多个 basic_ios 对象可以指向同一个 std::basic_streambuf 对象
* `std::basic_streambuf` 是 abstracts a raw device 


### 3.2.1. state flags 函数


* `clear()`  清除` state flag`
  

对特定状态的检查函数,返回值都是布尔类型
* good()  最近的IO操作没有任何是错误的情况下返回 `true`, 此时其他所有状态函数都是 `false`
* bad()   Returns true 如果有读写失败                 例如对一个没有以写入标志打开的流执行写入或者写入的磁盘已没有空间 
* fail()  Returns true 在`bad()`的基础上检查格式问题  例如文件读出来的是字符但是传输给了一个整数变量                   
* eof()   检查是否到了文件末尾.                      


# 4. <fstream> C++ 的文件读写流 

iostream 标准库,它提供了 cin 和 cout 方法分别用于从标准输入读取流和向标准输出写入流.  
从文件读取流和向文件写入流,这就需要用到 C++ 中另一个标准库 `fstream`  

| 数据类型 | 描述                                                                                                                         |
| -------- | ---------------------------------------------------------------------------------------------------------------------------- |
| ofstream | 该数据类型表示输出文件流, 用于创建文件并向文件写入信息。                                                                     |
| ifstream | 该数据类型表示输入文件流, 用于从文件读取信息。                                                                               |
| fstream  | 该数据类型通常表示文件流, 且同时具有 ofstream 和 ifstream 两种功能, 这意味着它可以创建文件, 向文件写入信息, 从文件读取信息。 |


要在 C++ 中进行文件处理, 必须在 C++ 源代码文件中包含头文件 `iostream` 和 `fstream`

要实现以二进制形式读写文件，<< 和 >> 将不再适用，需要使用 C++ 标准库专门提供的 read() 和 write() 成员方法。其中，read() 方法用于以二进制形式从文件中读取数据；write() 方法用于以二进制形式将数据写入文件。 

### 4.0.1. 文件打开与关闭

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

### 4.0.2. 文件读写

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


### 4.0.3. 文件位置指针操作
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

# 5. <sstream> 字符串流 

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


### 5.0.1. 从string中读取字符

stringstream 对象可以绑定一行字符串，然后以空格为分隔符把该行分隔开来
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
### 5.0.2. 用来进行数据类型转换

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
### 5.0.3. 字符串流的高级操作

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

# <iostream> 标准流

只有在该文件中定义了标准输入输出  

**Include**
* ios
* streambuf
* istream
* ostream

**对象**
标准IO
* cin
* cout
* cerr  无缓存的输出到 stderr
* clog  输出到 stderr

宽字符IO
* wcin
* wcout
* wcerr
* wclog

# 6. <iomanip> 流格式控制

该头文件里只有函数, 均是用于控制流格式的函数  

## std::setw

设置输出位宽  

```cpp
// 以下两个函数功能相同
str << setw(n);
str.width(n);



```
