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

# 4. <cmath> 通用数学函数

包括了从 C语言继承来的一些 通用数学运算  


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


# 5. <random> 随机数

* 所有随机数引擎都可以指定地播种, 序列化和反序列化, 以用于可重复的模拟器
* C++的随机数库, 提供:
  * `Uniform random bit generators (URBGs)`, 生成伪随机数, 若有真随机数设备也可以调用
  * `Random number distributions` , 将生成的随机数转化成相应的统计分布
  * 这两个类会相互调用

## 预定义的

## 随机数引擎 Random number engines 


## 随机数引擎适配器 Random number engine adaptors


## C++20新内容 

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
