
# 1. numeric 数学运算库

### 1.1. iota 自动生成加1数列 since C++11


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
### 1.2. gcd lcm 最大公约数 最小公倍数 since c++17

```cpp
// greatest common divisor
template< class M, class N>
constexpr std::common_type_t<M, N> gcd(M m, N n);
// If either |m| or |n| is not representable as a value of type std::common_type_t<M, N>, the behavior is undefined.

// least common multiple
template< class M, class N>
constexpr std::common_type_t<M, N> lcm(M m, N n);
// The behavior is undefined if |m|, |n|, or the least common multiple of |m| and |n| is not representable as a value of type std::common_type_t<M, N>. 

// 这两个函数都是 如果 m和n 都是 0 ， 则返回 0
// If either M or N is not an integer type, or if either is (possibly cv-qualified) bool, the program is ill-formed.

```
### 1.3. accumulate 

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


// 两个版本的内部实现(可能的)
template<class InputIt, class T>
constexpr // since C++20
T accumulate(InputIt first, InputIt last, T init)
{
    for (; first != last; ++first) {
        init = std::move(init) + *first; // std::move since C++20
    }
    return init;
}

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

### 1.4. midpoint since c++20

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

