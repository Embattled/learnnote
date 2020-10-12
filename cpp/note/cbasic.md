# 1. C语言的标准库
C语言的标准库一共只有二十多个

## 1.2. string.h / cstring

注意, c语言的字符串头文件只有 `string.h` 一个  
`cstring` 是 c++ 中对 `string.h`的增强实现,属于C++库,而`string` 则是原生C++库

## 1.3. stdlib.h

头文件 `<stdlib.h>` 在C++中为 `<cstdlib>`  


## 1.4. dlfcn.h  显示调用动态链接库

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