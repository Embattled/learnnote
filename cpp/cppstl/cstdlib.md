
# 1. stdlib.h /cstdlib

因为该头文件的内容太杂, 单独列出文件
This header provides miscellaneous utilities.   
Symbols defined here are used by several library components.   


# 2. Functions 函数
## 2.1. 随机数生成

所有和计算机有关的随机都是数学意义上的伪随机,真正的随机基本都是基于物理上的分子熵增,白噪声等因素作为种子来生成随机数  

在 c语言 stdlib 中有随机数生成函数

* RAND_MAX  : 和机器有关的常量, 反映了随机数能得到的最大的值, 一般最低也有32767 
* rand() : 返回随机数 范围是`[0,RAND_MAX]`
* srand(): 给随机函数 rand() 投喂种子,如果没有投喂,则程序默认已经执行 srand(1),一般使用`time(0)`来获取当前时间作为种子

```cpp

std::srand(std::time(0)); //use current time as seed for random generator
int random_variable = std::rand();
std::cout << "Random value on [0 " << RAND_MAX << "]: " << random_variable << '\n';
```


## 2.2. Numeric string conversion  字符串转换成数字类型

因为是C语言库, 函数参数都是 `const char * str`  

```c
// 转换为 int 类型  有很好的鲁棒性
// 该函数首先丢弃尽可能多的空白字符，直到找到第一个非空白字符。 然后，从该字符开始，接受可选的正负号，后跟尽可能多的基数为10的数字，并将它们解释为数值。
int atoi (const char * str);
// 例子
char buffer[256];
printf ("Enter a number: ");
fgets (buffer, 256, stdin);
int i = atoi (buffer);

// 转换为 long 类型
long int atol ( const char * str );

// 转换为 long long 类型
long long int atoll ( const char * str );

// 转换为 浮点 类型
// 转为浮点数的函数会首先忽略前导空白，并接受正负号、指数（e/E）、十六进制数（0x/0X开头）。
double atof (const char* str);
double atod (const char* str);

// 另一种功能更健全的转换函数 , 转为 long   
// base:指定基数   如果将base设为0，则该函数将根据 str 的格式来自动决定基数
// 如果 endptr 不为空指针，则该函数还将 endptr 的值设置为指向该数字后面的第一个字符. 即可以连续读入 
long int strtol (const char* str, char** endptr, int base);

// 转换为 unsigned long 类型
unsigned long int strtoul (const char* str, char** endptr, int base);

// 转为 long long类型的整数
long long int strtoll (const char* str, char** endptr, int base);

// 转换为 unsigned long long 类型
unsigned long long int strtoull (const char* str, char** endptr, int base);

// 转换为 浮点 类型
float strtof (const char* str, char** endptr);
double strtod (const char* str, char** endptr);
long double strtold (const char* str, char** endptr);
```
