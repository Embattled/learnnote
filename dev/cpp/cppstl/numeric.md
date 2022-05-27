# 1. 数值库 Numerics

```cpp
#include <cmath>
#include <cfenv>
#include <climit>

#include <limits>
#include <complex>
#include <random>
#include <valarray>
#include <numeric>
#include <bit>
#include <numbers>
```

标准C++数值库包含了常规数学函数以及类型， 以及一些特殊化的数列和随机数生成。
包括了一系列的头文件  稍微有点多  

# 2. limit 库

和数值库配套的 limit 两个库提供了所有基本类型的相关数字范围
* climit
* limits

## 2.1. <climits>

This header was originally in the C standard library as `<limits.h>`.

This header is part of the type support library, in particular it's part of the C numeric limits interface. 


定义了全部的 `整数` 数字界限 


注意常用的 INT_MAX 是C++11才定义的  

| 名称                    | 定义                                     | 备注            |
| ----------------------- | ---------------------------------------- | --------------- |
| CHAR_BIT                | 一个byte中有多少bit                      |
| MB_LEN_MAX              | 一个multibyte character中最大有多少bytes |
| CHAR_MAX                | char 类型的最大值                        |
| CHAR_MIN                | char 类型的最小值                        |
| 在这下面的是C++11加入的 |                                          |
| *_MIN                   | 该类型的最小值                           | *号代表多种类型 |
| *_MAX                   | 该类型的最大值                           | *号代表多种类型 |
| U*_MAX                  | 该类型的最大值                           | *号代表多种类型 |

*号的对应表

| *号填入 | 对应数据类型 |
| ------- | ------------ |
| SCHAR   | signed char  |
| SHRT    | short        |
| INT     | int          |
| LONG    | long         |
| LLONG   | long long    |

只有这么点内容，很容易记

## 2.2. <limits>

C++的数据界限对应库,定义了一个类和两个枚举类型

### 2.2.1. std::numeric_limits

# 3. <cfenv> 浮点数环境

非标准C头文件, 用于控制 浮点数的各种错误符号位  

1. 在标准下:
    * 该头文件只有在 `#pragma STDC FENV_ACCESS`  生效时才能够被使用
    * 否则的话所有设置模式都是默认, 并且所有相关符号位不会进行测试即修改  
2. 事实上:
   * 少数编译器 `HP aCC, Oracle Studio, or IBM XL` 遵循了上面的标准, 只有定义了 pragma才能使用
   * 大多数编译器直接就可以使用该 floating-point environment

原文: 
```
The floating-point environment is the set of floating-point status flags and control modes supported by the implementation. It is thread-local, each thread inherits the initial state of its floating-point environment from the parent thread. Floating-point operations modify the floating-point status flags to indicate abnormal results or auxiliary information. The state of floating-point control modes affects the outcomes of some floating-point operations.

The floating-point environment access and modification is only meaningful when #pragma STDC FENV_ACCESS is supported and is set to ON. Otherwise the implementation is free to assume that floating-point control modes are always the default ones and that floating-point status flags are never tested or modified. In practice, few current compilers, such as HP aCC, Oracle Studio, or IBM XL, support the #pragma explicitly, but most compilers allow meaningful access to the floating-point environment anyway. 

```

# 4. cmath math.h 通用数学函数

包括了从 C语言继承来的一些 通用数学运算  

`cmath` 和 `math.h` 的差异
* 库中的函数分类大致相同
* 因为C语言没有重载, 所以 cmath 里有一些独有的方便函数, 面向多类型重载
* 有些函数是 C99 中加入到 math.h 里的, 但是在 C++11里才被加入到 cmath, 需要注意


对于末尾不加类型修饰的函数名
* C 语言版本针对 double
* C++ 语言版本为多类型重载对应

## 4.1. 类型

Cmath 里只定义了两个类型， 都是C++11新加入的
1. `float_t`    
2. `double_t`

most efficient floating-point type at least as wide as float  
most efficient floating-point type at least as wide as double  


## 4.2. 常量与宏

math.h 的常量基本都与浮点数相关  
注意 : 全部都是 C11 新加入的  

### 4.2.1. 浮点特殊值
IEEE 754 中关于特殊值的表示方法
1. 正负无穷     E 段全1 , M 段 全0 , S 表示正负
2. NaN          E 段全1 , M段非全 0
3. 正负0        E 段和 M 段都是全0 , S 表示正负
4. 非规格化数   E段全0 M段非全零

这几个特殊值相关的宏:
* 系统中有没有实现浮点数的无穷会影响以下宏的效果
1. `INFINIT`
   * 如果系统实现了无穷 表示 float 的 positive or unsigned infinity
   * 没有实现, 该宏可以保证在编译时溢出 float 的范围 并产生编译警报
2. `NAN`
   * evaluates to a quiet not-a-number (QNaN) value
   * If the implementation does not support QNaNs, this macro constant is not defined.
   * 对于不同的 NAN 具体由相关函数的符号位来表示

### 4.2.2. 越界

作为函数和操作的返回值  
compare equal to the values returned by floating-point functions and operators in case of overflow  
* HUGE_VALF     超过了 float 的范围
* HUGE_VAL      超过了 double 的范围
* HUGE_VALL     超过了 long double 的范围



* MATH_ERRNO    
* MATH_ERREXCEPT    
* math_errhandling      
```c
#define MATH_ERRNO        1  

#define MATH_ERREXCEPT    2

#define math_errhandling  /*implementation defined*/
```

### 4.2.3. 浮点数类型

和该头文件下的 `fpclassify` 进行配合, 作为该函数的返回值  
* `FP_NORMAL`       常规浮点数
* `FP_SUBNORMAL`    subnormal
* `FP_ZERO`         正负0
* `FP_INFINITE`     正负无穷
* `FP_NAN`          NaN

用法:
```cpp
const char* show_classification(double x) {
    switch(std::fpclassify(x)) {
        case FP_INFINITE:  return "Inf";
        case FP_NAN:       return "NaN";
        case FP_NORMAL:    return "normal";
        case FP_SUBNORMAL: return "subnormal";
        case FP_ZERO:      return "zero";
        default:           return "unknown";
    }
}
int main()
{
    // 1 除以 0 是无穷
    std::cout << "1.0/0.0 is " << show_classification(1/0.0) << '\n'
    // 0 除以 0 是NaN
              << "0.0/0.0 is " << show_classification(0.0/0.0) << '\n'
    // subnormal
              << "DBL_MIN/2 is " << show_classification(DBL_MIN/2) << '\n'
    // 零
              << "-0.0 is " << show_classification(-0.0) << '\n'
    // Normal
              << "1.0 is " << show_classification(1.0) << '\n';
}
```


## 4.3. 基础运算函数

### 4.3.1. 绝对值 abs

```cpp
// 同时也定义在了 `cstdlib` 的函数 C++重载的版本
int             abs(int j);
long int        abs(long int j);
long long int   abs(long long int j);
float           abs(float j);
double          abs(double j);
long double     abs(long double j);

// 定义在 cmath 的原生 c 函数
float       fabsf( float arg );
double      fabs ( double arg );
long double fabsl( long double arg );
double      fabs ( IntegralType arg );
```
### 4.3.2. 取模 mod
### 4.3.3. 取最大最小 max min
## 4.4. 指数对数函数

这些函数的命名规则都大致相同  
都有重载版本和针对不同浮点数类型单独命名的版本  
* 重载版本接受 float, double, long double
* 单独版本 功能名假设为 x
  * x  为 double 版本
  * xf 为 float 版本
  * xl 为 long double 版本
  * 还有一个额外的 x , 返回值为 double , 接受 `IntegralType` 类型

### 4.4.1. e指数
```cpp
// 重载通用版本
float       exp ( float arg );
double      exp ( double arg );
long double exp ( long double arg );

// 单独版本
float       expf( float arg );
long double expl( long double arg );
double      exp ( IntegralType arg );
```

### 4.4.2. 2指数
```cpp
// 重载通用版本
float       exp2 ( float arg );
double      exp2 ( double arg );
long double exp2 ( long double arg );

// 单独版本
float       exp2f( float arg );
long double exp2l( long double arg );
double      exp2 ( IntegralType arg );
```


### 4.4.3. e对数
```cpp
// 重载通用版本
float       log ( float arg );
double      log ( double arg );
long double log ( long double arg );

// 单独版本
float       logf( float arg );
long double logl( long double arg );
double      log ( IntegralType arg );
```

### 4.4.4. 2和10对数
```cpp
// 重载通用版本
float       log10 ( float arg );
double      log10 ( double arg );
long double log10 ( long double arg );

// 单独版本
float       log10f( float arg );
long double log10l( long double arg );
double      log10 ( IntegralType arg );

// 重载通用版本
float       log2 ( float arg );
double      log2 ( double arg );
long double log2 ( long double arg );

// 单独版本
float       log2f( float arg );
long double log2l( long double arg );
double      log2 ( IntegralType arg );
```

## 4.5. 幂函数 power function

* pow 函数是开平方和开立方的完整函数
* pow 函数支持输入负数幂用来开任意次方

1. pow 函数, C++11之前只支持整数
2. sqrt
3. cbrt C++11新函数
4. hypot C++11新函数

### 4.5.1. 幂函数 pow

```cpp
// 大部分都是从 C++11 支持
// C++11以前的幂只支持整数

float       pow ( float base, float exp );
double      pow ( double base, double exp );
long double pow ( long double base, long double exp );
Promoted    pow ( Arithmetic1 base, Arithmetic2 exp )

float       powf( float base, float exp );
long double powl( long double base, long double exp );
```

### 4.5.2. 开平方 sqrt
```cpp
float       sqrt ( float arg );
double      sqrt ( double arg );
long double sqrt ( long double arg );
double      sqrt ( IntegralType arg );

float       sqrtf( float arg );
long double sqrtl( long double arg );
```
### 4.5.3. 开立方 cbrt
```cpp
float       cbrt ( float arg );
double      cbrt ( double arg );
long double cbrt ( long double arg );
double      cbrt ( IntegralType arg );

float       cbrtf( float arg );
long double cbrtl( long double arg );
```
### 4.5.4. 开勾股 hypot

* C++17 开始支持三个数的开勾股

```cpp
// C++11 开始
float       hypot ( float arg );
double      hypot ( double arg );
long double hypot ( long double arg );
Promoted    hypot ( Arithmetic1 x, Arithmetic2 y );

float       hypotf( float arg );
long double hypotl( long double arg );

// C++17 开始
float       hypot ( float x, float y, float z );
double      hypot ( double x, double y, double z );
long double hypot ( long double x, long double y, long double z );
Promoted    hypot ( Arithmetic1 x, Arithmetic2 y, Arithmetic3 z );
```

## 4.6. 三角函数 trigonometric functions

特殊方法获取 PI:
`const double pi = std::acos(-1);`   


同样都有重载版本和针对不同浮点数类型单独命名的版本  
* 重载版本接受 float, double, long double
* 单独版本 功能名假设为 x
  * x  为 double 版本
  * xf 为 float 版本
  * xl 为 long double 版本
  * 还有一个额外的 x , 返回值为 double , 接受 `IntegralType` 类型


三角函数特点: 
1. 参数 arg 为 弧度


* sin
* cos
* tan
* asin
* acos
* atan          没有象限判断的版本, 返回值为正负二分之派
* atan2         有象限判断的版本, 返回值为正负派

## 4.7. 最近整数运算 Nearest integer floating-point operations 

版本信息:  
* 原生函数只有 `ceil()` 和 `floor()`, 其他剩余所有都是 C++11 (cmath) 或 C99 (math.h) 提供
* 返回值的类型基本输入类型相同, 并不是整数类型, 需要再次手动转换, 除了 rint 和 nearbyint 的 ll 形式


标准最近值, 有另外的 `f` `l` 版本, 分别针对 float 类型和 long double 
* ceil        : 天花板
* floor       : 地板
* trunc       : 直接删除掉小数部分的整数
* nearbyint   : 根据 `fenv.h/ cfenv` 中的模式来返回
  * `int fesetround( int round );` 设定模式
  * `int fegetround();` 获取模式
  * 四种模式分别是
    * `FE_DOWNWARD`   = floor
    * `FE_TONEAREST`  不等于 round, 是 五舍五入, 确保结果是偶数
    * `FE_TOWARDZERO` = trunc
    * `FE_UPWARD`     = ceil


特殊最近值
* round : 标准四舍五入, rounding away from zero in halfway cases 
* rint  : 五舍五入相当于 nearbyint, 但是有拥有返回值是整数的派生函数
* round 和 rint 具有返回值是整数类型的函数 函数名首追加 `l` long, `ll` long long 

# 5. random 随机数

C语言的随机数定义在 `cstdlib` 中
* 如果学习了 C++的随机数, 就不推荐再使用C的随机数生成器了
  * rand() 生成伪随机数, 是一个整数
  * srand() 输入种子
  * RAND_MAX 生成的最大可能值


* 所有随机数引擎都可以指定地播种, 序列化和反序列化, 以用于可重复的模拟器
* C++的随机数库, 提供:
  * `Uniform random bit generators (URBGs)`, 生成伪随机数, 若有真随机数设备也可以调用
  * `Random number distributions` , 将生成的随机数转化成相应的统计分布
  * 这两个类会相互调用



```cpp
// random_device 获取真随机值(若可能) 用于播种
std::random_device r;

// 播种一个 default_random_engine 
std::default_random_engine e1(r());

// 均匀分布 1 与 6 间的随机数
std::uniform_int_distribution<int> uniform_dist(1, 6);
// 获取随机值
int mean = uniform_dist(e1);
```

## 5.1. 随机数生成系统 Uniform random bit generators 

C++的 random 定义了一套复杂的随机数引擎系统


### 5.1.1. 预定义随机数生成器

* 可以直接调用的方便的标准库类, 日常使用的对象
* 具体的类型需要参照类的定义
  
| 生成器名(C++11)       | 简单说明                   |
| --------------------- | -------------------------- |
| minstd_rand0          | 1988采纳的最低标准         |
| minstd_rand           | 1993更新的最低标准         |
| mt19937               | 32为梅森缠绕器, 1998年发布 |
| mt19937_64            | 64位梅森缠绕器             |
| ranlux24_base         | 无说明                     |
| ranlux48_base         | 无说明                     |
| ranlux24              | 24位RANLUX生成器           |
| ranlux48              | 48位RANLUX生成器           |
| knuth_b               | shuffle order engine       |
| default_random_engine | 用于实现定义的默认         |

```cpp

// minstd_rand0
std::linear_congruential_engine<std::uint_fast32_t, 16807, 0, 2147483647> 
// minstd_rand
std::linear_congruential_engine<std::uint_fast32_t, 48271, 0, 2147483647>

// mt19937
std::mersenne_twister_engine<std::uint_fast32_t, 32, 624, 397, 31,
                             0x9908b0df, 11,
                             0xffffffff, 7,
                             0x9d2c5680, 15,
                             0xefc60000, 18, 1812433253>
// mt19937_64
std::mersenne_twister_engine<std::uint_fast64_t, 64, 312, 156, 31,
                             0xb5026f5aa96619e9, 29,
                             0x5555555555555555, 17,
                             0x71d67fffeda60000, 37,
                             0xfff7eee000000000, 43, 6364136223846793005>

// ranlux24_base
std::subtract_with_carry_engine<std::uint_fast32_t, 24, 10, 24>
// ranlux48_base
std::subtract_with_carry_engine<std::uint_fast64_t, 48, 5, 12>

// ranlux24
std::discard_block_engine<std::ranlux24_base, 223, 23>
// ranlux48
std::discard_block_engine<std::ranlux48_base, 389, 11>

// knuth_b
std::shuffle_order_engine<std::minstd_rand0, 256>

```

### 5.1.2. 随机数引擎 Random number engines 

* 所谓引擎?
  * 以种子数据为熵源生成伪随机数
  * C++定义了几种不同的随机数算法, 并将它们实现成可以定制的模板类
* 引擎的选择需要进行权衡
  * 线性同余 (linear_congruential)   一般的快, 存储要求小
  * 延迟斐波那契 (lagged Fibonacci ) 在任何设备上(无先进指令集)都能最快执行, 存储要求大, 有时会有` 不太想要的谱特性`
  * 梅森缠绕器 (Mersenne twister)    最慢, 存储要求大, 但是能生产最长的不重复序列, 且能得到`想要的谱特性`
* 类名称(C++11)
  * `linear_congruential_engine`  线性同余
  * `subtract_with_carry_engine` 带进位减(延迟斐波那契)
  * `mersenne_twister_engine`     梅森缠绕器



### 5.1.3. 随机数引擎适配器 Random number engine adaptors

* 何为引擎适配器?
  * 引擎已经可以生成随机数, 适配器用于进一步打乱随机数的输出结果
  * 这种适配器主要的目的是改名引擎的 `谱特性`
* 类名称(C++11)
  * `discard_block_engine`     舍弃随机数引擎的某些输出 
  * `independent_bits_engine`  将一个随机数引擎的输出打包为指定位数的块 
  * `shuffle_order_engine`     以不同顺序发送一个随机数引擎的输出 

### 5.1.4. 非确定随机数 Non-deterministic random numbers 

`std::random_device`
* 最简单的随机数生成器, 均匀分布的生成一个整数
  * 一般不适用该类作为具体的随机数生成器
  * 而是用该类只生成一个数, 作为其他标准生成器(类似mt19937) 的种子
* 某种程度的真随机数
  * 如果`硬件允许`, 使用硬件熵源来生成随机数
  * 否则仍然使用上文的伪随机数引擎

成员函数:
* () : 返回随机生成的值
* entropy()  返回该生成器的熵估计. 
  * 确定的随机数生成器（例如伪随机数生成器）拥有零熵
  * 设备熵的值，或若不可应用则为零.
* min()  返回固定值, 意为该随机数生成器的最小值
* max()  返回固定值, 意为该随机数生成器的最大值

```cpp
// 默认构造函数定义
random_device() : random_device(/*implementation-defined*/) {}
// 带有 token 的随机生成器, 这里 token 是另一个知识点
explicit random_device(const std::string& token);

// 随机生成器不可以被复制
random_device(const random_device& ) = delete;


// example
std::random_device rd; // 使用 RDRND 或 /dev/urandom
std::random_device rd2("/dev/random"); // Linux 上更慢
std::uniform_int_distribution<int> dist(0, 9);

std::map<int, int> hist;
// 一般不适用 random_device 生成大量的值
for (int n = 0; n < 20000; ++n) {
  // 统计随机生成的值
  ++hist[dist(rd)];
}
```
## 5.2. 随机数分布

* 所谓一种后处理 post-processes , 将 URBG 的输出结果按照定义的统计概率密度函数分布
* 所有标准库的分布都满足 `C++具名要求: 随机数分布 (RandomNumberDistribution)`
* 分布的分类:
  * 均匀分布
  * 伯努利分布
  * 泊松分布
  * 正态分布
  * 采样分布



## 5.3. C++20新内容 

* 指定类型的 `uniform_random_bit_generator` 
```cpp
template <class G>
concept uniform_random_bit_generator =
  std::invocable<G&> && std::unsigned_integral<std::invoke_result_t<G&>> &&
  requires {
    { G::min() } -> std::same_as<std::invoke_result_t<G&>>;
    { G::max() } -> std::same_as<std::invoke_result_t<G&>>;
    requires std::bool_constant<(G::min() < G::max())>::value;
  };


```

# 6. <number> 数学常量



# 7. <numeric> 数学运算库

numerics library 的一部分， 包含一些特殊的常用函数，可以简化编程流程  

## 7.1. accumulate 

自动累加 first 到 last 之间的元素  

```cpp

// 使用 operator+ 来进行相加
template< class InputIt, class T >
constexpr T accumulate( InputIt first, InputIt last, T init );

// uses the given binary function op
template< class InputIt, class T, class BinaryOperation >
constexpr T accumulate( InputIt first, InputIt last, T init,BinaryOperation op );


/* 
first, last 	- 	the range of elements to sum
init 	        - 	initial value of the sum
op 	          - 	binary operation function object that will be applied. The binary operator takes the current a    
                  ccumulation value a (initialized to init) and the value of the current element b. 


The signature of the function should be equivalent to the following:
  Ret fun(const Type1 &a, const Type2 &b);

The signature does not need to have const &.
The type Type1 must be such that an object of type T can be implicitly converted to Type1. The type Type2 must be such that an object of type InputIt can be dereferenced and then implicitly converted to Type2. The type Ret must be such that an object of type T can be assigned a value of type Ret. ​ 

*/

// 返回值
// Return value
// 1) The sum of the given value and elements in the given range.
// 2) The result of left fold of the given range over op

/* 
Notes:
std::accumulate performs a left fold. In order to perform a right fold, 
one must reverse the order of the arguments to the binary operator, and use reverse iterators. 
*/

// 两个版本的内部实现(可能的) 其一
template<class InputIt, class T>
constexpr // since C++20
T accumulate(InputIt first, InputIt last, T init)
{
    for (; first != last; ++first) {
        init = std::move(init) + *first; // std::move since C++20
    }
    return init;
}

// 其二

template<class InputIt, class T, class BinaryOperation>
constexpr // since C++20
T accumulate(InputIt first, InputIt last, T init, 
             BinaryOperation op)
{
    for (; first != last; ++first) {
        init = op(std::move(init), *first); // std::move since C++20
    }
    return init;
}


// e.g.

int main()
{
   std::vector<int> v{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
   int sum = std::accumulate(v.begin(), v.end(), 0);
   //  sum: 55

   int product = std::accumulate(v.begin(), v.end(), 1, std::multiplies<int>());
  //  product: 3628800

}

```

## 7.2. inner_product

内积, 类似于向量点乘  

使用方法:
```cpp

int main()
{
    std::vector<int> a{0, 1, 2, 3, 4};
    std::vector<int> b{5, 4, 2, 3, 1};
 
    int r1 = std::inner_product(a.begin(), a.end(), b.begin(), 0);

    int r2 = std::inner_product(a.begin(), a.end(), b.begin(), 0, std::plus<>(), std::equal_to<>());

}

```

函数定义: 
```cpp

template< class InputIt1, class InputIt2, class T >
constexpr T inner_product( InputIt1 first1, InputIt1 last1,
                           InputIt2 first2, T init );

// 自定义操作函数的版本
template<class InputIt1, class InputIt2, class T,class BinaryOperation1, class BinaryOperation2>
constexpr T inner_product( InputIt1 first1, InputIt1 last1,
                           InputIt2 first2, T init,
                           BinaryOperation1 op1,
                           BinaryOperation2 op2 );

```

## 7.3. gcd lcm 公倍数 公约数 since c++17

C++17 才加入的新函数   
不会抛出异常  

```cpp
// greatest common divisor
template< class M, class N>
constexpr std::common_type_t<M, N> gcd(M m, N n);

// least common multiple
template< class M, class N>
constexpr std::common_type_t<M, N> lcm(M m, N n);


/* 
Return value : 

gcd:  如果 m和n 都是0, 返回0, 否则正常返回 |m|和|n|的gcd
lcm:  如果 m和n 中有一个是0, 则返回0, 否则正常返回绝对值的 lcm
*/

/* 
Rmark:

gcd:
    1. 如果 m和n 不全是整数, 则该程序是 ill-formed.
    2.  If either |m| or |n| is not representable as a value of type std::common_type_t<M, N>, the behavior is undefined.

lcm: 
    1. 如果 m和n 不全是整数, 则该程序是 ill-formed.
    2. The behavior is undefined if |m|, |n|, or the least common multiple of |m| and |n| is not representable as a value of type std::common_type_t<M, N>. 


*/
```

## 7.4. iota  自动生成加1数列


```cpp
// until c++20
template< class ForwardIt, class T >
void iota( ForwardIt first, ForwardIt last, T value )

// since c++20
template< class ForwardIt, class T >
constexpr void iota( ForwardIt first, ForwardIt last, T value );

// Fills the range [first, last) with sequentially increasing values, starting with value and repetitively evaluating ++value. 
// 只能生成加1数列, 不能其他等比数列

// 由于内部是用 value++ 实现的, 因此可以自加的数据类型都可以作为 value 输入 


// 给 list 赋值 -4 到 5
    std::list<int> l(10);
    std::iota(l.begin(), l.end(), -4);
 
// 自动将list的每个值的地址存到 vector 中 
    std::vector<std::list<int>::iterator> v(l.size());
    std::iota(v.begin(), v.end(), l.begin());
```
### 7.4.1. midpoint since c++20

Computes the midpoint of the integers, floating-points, or pointers a and b.   
计算中点  

```cpp

template< class T >
constexpr T midpoint(T a, T b) noexcept;
template< class T >
constexpr T* midpoint(T* a, T* b);

// a, b 	- 	integers, floating-points, or pointer values
// 如果是指针的话, a和b 必须是同一个序列的指针
/* 
Return:
1) Half the sum of a and b. No overflow occurs. 
  If a and b have integer type and the sum is odd, the result is rounded towards a. 
  If a and b have floating-point type, at most one inexact operation occurs.
  如果 a 和 b 的差是奇数, 则返回的值偏向 a 

2) If a and b point to, respectively, x[i] and x[j] of the same array object x (for the purpose of pointer arithmetic), returns a pointer to x[i+(j-i)/2] (or, equivalently x[std::midpoint(i, j)]) where the division rounds towards zero. 
  If a and b do not point to elements of the same array object, the behavior is undefined.
*/


// 可以无溢出的计算 a和b 的中值
int main()
{
    std::uint32_t a = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t b = std::numeric_limits<std::uint32_t>::max() - 2;
    // a: 4294967295
    // b: 4294967293
    std::cout << "Incorrect (overflow and wrapping): " << (a + b) / 2 << '\n'
              << "Correct: " << std::midpoint(a, b) << "\n\n";
    // Incorrect (overflow and wrapping): 2147483646
    // Correct: 4294967294
}

```

## 7.5. partial_sum

累加数列的生成  

```cpp
template< class InputIt, class OutputIt >
constexpr OutputIt partial_sum( InputIt first, InputIt last, OutputIt d_first );

// 自定义操作版本
template< class InputIt, class OutputIt, class BinaryOperation >
constexpr OutputIt partial_sum( InputIt first, InputIt last, OutputIt d_first,
                                BinaryOperation op );

/* 
range ：[first, last)

*(d_first)   = *first;
*(d_first+1) = *first + *(first+1);
*(d_first+2) = *first + *(first+1) + *(first+2);
*(d_first+3) = *first + *(first+1) + *(first+2) + *(first+3);
...
 */

// 注意这里的 d_first 不一定是原本的数列位置

std::vector<int> v = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
std::partial_sum(v.begin(), v.end(), std::ostream_iterator<int>(std::cout, " "))

// 自定义操作，改成累乘
std::partial_sum(v.begin(), v.end(), v.begin(), std::multiplies<int>());
```

# 8. <complex> 复数运算
