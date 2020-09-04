
# 1. C语言的标准库
C语言的标准库一共只有二十多个
## 1.1. stdio.h 

`EOF` 是一个定义在头文件 stdio.h 中的常量, 用来表示已经到达文件结束的负整数,在读写时发生错误也会返回这个宏值  

### 1.1.1. C 语言的基础文件读写

1. 文件打开
控制流打开与 `FILE` 对象, 创建一个新的文件或者打开一个已有的文件  
`FILE *fopen( const char * filename, const char * mode );`  
访问模式 mode 的值可以是下列值中的一个  
| 模式 | 描述                                                                 |
| ---- | -------------------------------------------------------------------- |
| r    | 只读                                                                 |
| w    | 只写,从头开始写,如果文件已存在,则会被从开头覆盖,不存在则会创建新文件 |
| a    | 追加写,不存在会创建新文件                                            |
| +    | 代表可读写,追加在上面模式之后                                        |
| b    | 代表二进制读写,追加在上面模式之后                                    |

2. 文件读写(通过字符)

`int fputc(int char, FILE *stream)`  
把参数 char 指定的字符（一个无符号字符）写入到指定的流 stream 中,并把位置标识符往前移动  
`int fputs(const char *str, FILE *stream)`  
把字符串写入到指定的流 stream 中,但不包括空字符,把一个以 null 结尾的字符串写入到流中  

`int fgetc( FILE * fp );`  
fgetc() 函数从 fp 所指向的输入文件中读取一个字符。返回值是读取的字符,如果发生错误则返回 EOF

`char *fgets( char *buf, int n, FILE *fp );`   
从输入流中读入 ***n - 1*** 个字符,并在最后追加一个 null 字符来终止字符串, 总计 n 个字符  
如果这个函数在读取最后一个字符之前就遇到一个换行符 '\n' 或文件的末尾 EOF,则只会返回读取到的字符,包括换行符

3. 文件读写(通过格式化流)

`int fscanf(FILE *stream, const char *format, ...)`  

`int fprintf(FILE *stream, const char *format, ...)`

与普通的 `print scanf` 类似的输入,只不过前面加入了文件流参数  


4. 二进制读写

用于存储块的读写 - 通常是数组或结构体
```cpp
size_t fread(void *ptr, size_t size_of_elements, size_t number_of_elements, FILE *a_file);
              
size_t fwrite(const void *ptr, size_t size_of_elements, size_t number_of_elements, FILE *a_file);
```
元素**总数**会以 size_t 对象返回, 如果与输入的元素总数不一致,则可能是遇到了文件结尾  

5. 文件流的操作


6. 关闭文件
`int fclose( FILE *fp );`    
关闭流stream,会清空缓冲区中的数据,关闭文件,并释放用于该文件的所有内存  
如果成功关闭文件,fclose( ) 函数返回零,如果关闭文件时发生错误,函数返回 EOF

### 1.1.2. 1.2 C语言的文件增删改

创建一个文件用文件流打开的函数 `fopen` 即可完成

删除文件  
`int remove(const char *filename)`  

重命名或者移动文件  
`int rename(const char *old_filename, const char *new_filename)`  

这两个函数如果成功,则返回零。如果错误,则返回 -1,并设置 errno  

### 1.1.3. 1.3 C 文件流的指针操作

#### 1.1.3.1. 判断文件指针是否到末尾
`int feof(FILE *stream)`  当已经读到末尾时返回一个非零值  
`if( feof ( FILE ) ) break;`  用来跳出读取




## 1.2. string.h / cstring

注意, c语言的字符串头文件只有 `string.h` 一个  
`cstring` 是 c++ 中对 `string.h`的增强实现,属于C++库,而`string` 则是原生C++库


## 1.3. dlfcn.h  显示调用动态链接库

在 C/C++ 程序中显示调用动态链接库时，无需引入和动态链接库相关的头文件  

```cpp
// 相关头文件
#include <dlfcn.h>

// 读取动态链接库 将库文件装载到内存中，为后续使用做准备
// 使用 dlopen 函数 filename 参数用于表明目标库文件的存储位置和库名
/*
flag 参数的值有 2 种
    RTLD_NOW：将库文件中所有的资源都载入内存
    RTLD_LAZY：暂时不降库文件中的资源载入内存，使用时才载入
*/
void *dlopen (const char *filename, int flag);


// dlsym() 函数可以获得指定函数在内存中的位置 , 如果查找失败则返回 NULL
// hanle 参数表示指向已打开库文件的指针
// symbol 参数用于指定目标函数的函数名
void *dlsym(void *handle, char *symbol);

// dlopen() 相对地，借助 dlclose() 函数可以关闭已打开的动态链接库
// 当函数返回 0 时，表示函数操作成功；反之，函数执行失败
// handle 表示已打开的库文件指针
int dlclose (void *handle);


// 查错函数 dlerror() 
// 获得最近一次 dlopen()、dlsym() 或者 dlclose() 函数操作失败的错误信息
const char *dlerror(void);
```
**要点**
* filename 参数
  * 如果用户提供的是以 / 开头，即以绝对路径表示的文件名，则函数会前往该路径下查找库文件
  * 反之，如果用户仅提供文件名，则该函数会依次前往 `LD_LIBRARY_PATH` 环境变量指定的目录、`/etc/ld.so.cache` 文件中指定的目录、`/usr/lib、/usr/lib64、/lib、/lib64` 等默认搜索路径中查找。
* 目标库彻底释放
  * 调用 dlclose() 函数并不一定会将目标库彻底释放
  * 它只会是目标库的引用计数减 1，当引用计数减为 0 时，库文件所占用的资源才会被彻底释放


使用实例
```cpp

//打开库文件
void* handler = dlopen("libmymath.so",RTLD_LAZY);
if(dlerror() != NULL){
    printf("%s",dlerror());
}


//获取库文件中的 add() 函数
int(*add)(int,int)=dlsym(handler,"add");
if(dlerror()!=NULL){
    printf("%s",dlerror());
}

// 通过函数指针使用库函数
int sum=add(1,2);

//关闭库文件
dlclose(handler);

```