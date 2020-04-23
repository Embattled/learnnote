# 对C语言和C++的复习补充笔记

c++对默认参数, 只需要在声明或者实现部分任意一处设定即可, 不能两处都设定  
除法要想保留小数, 必须是用浮点数相除, 若是两个操作数都为整型, 结果会自动抹去小数  



## 关于main函数的参数的使用
关于  
`int argc, char const *argv[]`  
* argc是命令行总的参数个数  
* argv[]是argc个参数,其中第0个参数是程序的全名,以后的参数是用户输入的参数  
* main 的返回值为 `0` 代表程序正常退出; 非 `0` 代表不正常,在shell中视为假.


# C语言的标准库
C语言的标准库一共只有二十多个
## 1. stdio.h 

`EOF` 是一个定义在头文件 stdio.h 中的常量, 用来表示已经到达文件结束的负整数,在读写时发生错误也会返回这个宏值  

### 1.1 C 语言的基础文件读写
#### 文件打开
控制流打开与 `FILE` 对象, 创建一个新的文件或者打开一个已有的文件  
`FILE *fopen( const char * filename, const char * mode );`  
访问模式 mode 的值可以是下列值中的一个  
模式|描述  
--|--  
r|只读
w|只写,从头开始写,如果文件已存在,则会被从开头覆盖,不存在则会创建新文件
a|追加写,不存在会创建新文件
+|代表可读写,追加在上面模式之后
b|代表二进制读写,追加在上面模式之后

#### 文件读写(通过字符)

`int fputc(int char, FILE *stream)`  
把参数 char 指定的字符（一个无符号字符）写入到指定的流 stream 中,并把位置标识符往前移动  
`int fputs(const char *str, FILE *stream)`  
把字符串写入到指定的流 stream 中,但不包括空字符,把一个以 null 结尾的字符串写入到流中  

`int fgetc( FILE * fp );`  
fgetc() 函数从 fp 所指向的输入文件中读取一个字符。返回值是读取的字符,如果发生错误则返回 EOF

`char *fgets( char *buf, int n, FILE *fp );`   
从输入流中读入 ***n - 1*** 个字符,并在最后追加一个 null 字符来终止字符串, 总计 n 个字符  
如果这个函数在读取最后一个字符之前就遇到一个换行符 '\n' 或文件的末尾 EOF,则只会返回读取到的字符,包括换行符

#### 文件读写(通过格式化流)

`int fscanf(FILE *stream, const char *format, ...)`  

`int fprintf(FILE *stream, const char *format, ...)`

与普通的 `print scanf` 类似的输入,只不过前面加入了文件流参数  


#### 二进制读写

用于存储块的读写 - 通常是数组或结构体
```cpp
size_t fread(void *ptr, size_t size_of_elements, size_t number_of_elements, FILE *a_file);
              
size_t fwrite(const void *ptr, size_t size_of_elements, size_t number_of_elements, FILE *a_file);
```
元素**总数**会以 size_t 对象返回, 如果与输入的元素总数不一致,则可能是遇到了文件结尾  

#### 文件流的操作


#### 关闭文件
`int fclose( FILE *fp );`    
关闭流stream,会清空缓冲区中的数据,关闭文件,并释放用于该文件的所有内存  
如果成功关闭文件,fclose( ) 函数返回零,如果关闭文件时发生错误,函数返回 EOF

### 1.2 C 语言的文件操作

创建一个文件用文件流打开的函数 `fopen` 即可完成

删除文件  
`int remove(const char *filename)`  

重命名或者移动文件  
`int rename(const char *old_filename, const char *new_filename)`  

这两个函数如果成功，则返回零。如果错误，则返回 -1，并设置 errno  

### 1.3 C 文件流的指针操作

#### 判断文件指针是否到末尾
`int feof(FILE *stream)`  当已经读到末尾时返回一个非零值  
`if( feof ( FILE ) ) break;`  用来跳出读取




## 2. string.h / cstring

注意, c语言的字符串头文件只有 `string.h` 一个  
`cstring` 是 c++ 中对 `string.h`的增强实现,属于C++库,而`string` 则是原生C++库

