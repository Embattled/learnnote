# 1. algorithm

头文件 `algorithm` 定义了 cppstl 中绝大多数的算法和常用功能处理, 有其他小部分定义在了 `numeric` 和 `memory` 中


使用STL算法的好处

* 算法函数通常比自己写的循环结构效率更高；
* 自己写循环比使用算法函数更容易出错；
* 相比自己编写循环结构，直接调用算法函数的代码更加简洁明了。
* 使用算法函数编写的程序，可扩展性更强，更容易维护；


# 2. Non-modifying sequence operations 

## 2.1. std::find std::find_if std::find_if_not

注意只有 map multimap set multiset string 拥有 find 成员函数! 其他容器需要使用该全局函数  

find()的函数定义相对简单, 该函数适用于所有的序列式容器, 复杂度 O(n)  

值得一提的是，find() 函数的底层实现，其实就是用==运算符将 val 和 [first, last) 区域内的元素逐个进行比对。  
这也就意味着，[first, last) 区域内的元素必须支持==运算符。  

如果不支持的话应该自己重载 == 运算符  

```cpp
// find
template< class InputIt, class T >
constexpr InputIt find( InputIt first, InputIt last, const T& value );
// find_if
template< class InputIt, class UnaryPredicate >
constexpr InputIt find_if( InputIt first, InputIt last,UnaryPredicate p );
// find_if_not 在 C++ 11才被加入
template< class InputIt, class UnaryPredicate >
constexpr InputIt find_if_not( InputIt first, InputIt last,UnaryPredicate q );
// 如果没有 C++11 则可以用 std::not1 代替 
InputIt find_if_not(InputIt first, InputIt last, UnaryPredicate q)
{ return std::find_if(first, last, std::not1(q)); }



// C++ 17 加入的 policy 重载
template< class ExecutionPolicy, class ForwardIt, class T >
ForwardIt find( ExecutionPolicy&& policy, ForwardIt first, ForwardIt last, const T& value );

template< class ExecutionPolicy, class ForwardIt, class UnaryPredicate >
ForwardIt find_if( ExecutionPolicy&& policy, ForwardIt first, ForwardIt last,UnaryPredicate p );

template< class ExecutionPolicy, class ForwardIt, class UnaryPredicate >
ForwardIt find_if_not( ExecutionPolicy&& policy, ForwardIt first, ForwardIt last,UnaryPredicate q );

// 该函数会返回一个输入迭代器，当 find() 函数查找成功时，其指向的是在 [first, last) 区域内查找到的第一个目标元素；如果查找失败，则该迭代器的指向和 last 相同。


// find() 的底层实现 可能是:
template<class InputIterator, class T>
InputIterator find (InputIterator first, InputIterator last, const T& val)
{
  while (first!=last) {
      if (*first==val) return first;
      ++first;
  }
  return last;
}
// 同理
// find_if
for (; first != last; ++first) {
  if (p(*first)) {
    return first;
  }
}
// find_if_not
if (!q(*first)) { return first; }
```

**函数功能** 在first 和 last 中找到符合条件的元素并返回:
1. `find` searches for an element `equal` to value
2. `find_if` searches for an element for which `predicate p` returns true
3. `find_if_not` searches for an element for which `predicate q` returns false
4. policy 的重载只有在开启 policy 运行的时候 的时候才会生效
*  `std::is_execution_policy_v<std::decay_t<ExecutionPolicy>>`  C++20 之前  
*  `std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> ` C++ 20开始

**参数**:
* first, last `class InputIt` 类型的范围指示器
* value
* policy      将被使用的 policy
* p           unary predicate which returns ​true for the required element. 
* q           unary predicate which returns ​false for the required element.
* p 和 q 必须是返回 bool 的操作, 接受范围内的数据 `v` 以 `p(v)` 的形式操作, 并且操作内不能更改原本的值, 不能传入引用
* `InputIt` 必须是 `LegacyInputIterator`
* `ForwardIt ` 必须是 `LegacyForwardIterator`
* `UnaryPredicate` 必须是 `Predicate`

**返回值**:
Iterator to the first element satisfying the condition or last if no such element is found.   


```cpp
// find() 函数除了可以作用于序列式容器，还可以作用于普通数组
char stl[] ="http://c.biancheng.net/stl/";
char * p = find(stl, stl + strlen(stl), 'c');
if (p != stl + strlen(stl)) {
  cout << p << endl;
}

```
## 2.2. std::all_of std::any_of std::none_of

算是 bool 返回值的 find 函数, 不再返回迭代器而是直接返回 true false  

```cpp
// all_of
template< class InputIt, class UnaryPredicate >
constexpr bool all_of( InputIt first, InputIt last, UnaryPredicate p );  
// any_of C++11 加入
template< class InputIt, class UnaryPredicate >
constexpr bool any_of( InputIt first, InputIt last, UnaryPredicate p );
// none_of C++11 加入
template< class InputIt, class UnaryPredicate >
constexpr bool none_of( InputIt first, InputIt last, UnaryPredicate p );


// // C++ 17 加入的各个函数的 policy 重载
template< class ExecutionPolicy, class ForwardIt, class UnaryPredicate >
bool all_of( ExecutionPolicy&& policy, ForwardIt first, ForwardIt last, UnaryPredicate p );

template< class ExecutionPolicy, class ForwardIt, class UnaryPredicate >
bool any_of( ExecutionPolicy&& policy, ForwardIt first, ForwardIt last, UnaryPredicate p );

template< class ExecutionPolicy, class ForwardIt, class UnaryPredicate >
bool none_of( ExecutionPolicy&& policy, ForwardIt first, ForwardIt last, UnaryPredicate p );


// 该系列函数内部可能也用到了 find 系列
bool all_of(InputIt first, InputIt last, UnaryPredicate p)
{
  return std::find_if_not(first, last, p) == last;
} 

```

**函数参数** : first, last, policy, p 和 find 中的没什么区别, 输入值要求也相同  
* `InputIt` 必须是 `LegacyInputIterator`
* `ForwardIt ` 必须是 `LegacyForwardIterator`
* `UnaryPredicate` 必须是 `Predicate`

**返回值** : 针对输入的规则 p
* all_of  只有对所有的元素 v, p(v) 都返回 true, 才返回 true, 如果 range 为空 也会返回 true
* any_of  至少有一个 p(v) 返回true, 函数就会返回 true, 如果 range 为空也会返回 false
* none_of 必须所有元素的 p(v) 都返回 false, 函数才会返回 true, 如果 range 为空也返回 true


# 3. Modifying sequence operations 

## 3.1. std::transform

```cpp
// 一元 transform
template< class InputIt, class OutputIt, class UnaryOperation >
constexpr OutputIt transform( InputIt first1, InputIt last1, OutputIt d_first, UnaryOperation unary_op );
// 二元 transform
template< class InputIt1, class InputIt2, class OutputIt, class BinaryOperation >
constexpr OutputIt transform( InputIt1 first1, InputIt1 last1, InputIt2 first2, OutputIt d_first, BinaryOperation binary_op );

// C++ 17 加入的 policy 重载
template< class ExecutionPolicy, class ForwardIt1, class ForwardIt2, class UnaryOperation >
ForwardIt2 transform( ExecutionPolicy&& policy, ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 d_first, UnaryOperation unary_op );
template< class ExecutionPolicy, class ForwardIt1, class ForwardIt2, class ForwardIt3, class BinaryOperation >
ForwardIt3 transform( ExecutionPolicy&& policy, ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2, ForwardIt3 d_first, BinaryOperation binary_op );
```

* std::transform applies the given function to a range and stores the result in `another range`, beginning at d_first  
* `unary_op` and `binary_op` must not invalidate any iterators, including the end iterators, or modify any elements of the ranges involved. 

**函数参数**:
* first1, last1   : the first range of elements to transform
* first2          : the beginning of the `second` range of elements to transform
* d_first         : the beginning of the `destination` range, may be equal to first1 or first2
* policy          : the execution policy to use. See execution policy for details.
* unary_op        : unary operation function object that will be applied.
* binary_op       : binary operation function object that will be applied. 

操作函数的样例, 输入序列元素必须能转换成 `Type`, 输出结果`Ret` 必须能转换成 `OutputIt` 的元素 :  
`Ret fun(const Type &a);`   
`Ret fun(const Type1 &a, const Type2 &b);`  

返回值 : Output iterator to the element past the `last element` transformed.   

**注意**: 该函数不保证针对序列中每个元素的 transform 顺序执行, 如果对执行顺序有要求, 使用 `for_each`   



## 3.2. std::fill std::fill_n

fill() 和 fill_n() 算法提供了一种为元素序列填入给定值的简单方式
* fill() 会填充整个序列
* fill_n() 则以给定的迭代器为起始位置，为指定个数的元素设置值

```cpp

// fill
template< class ForwardIt, class T >
constexpr void fill( ForwardIt first, ForwardIt last, const T& value );
// fill_n
template< class OutputIt, class Size, class T >
constexpr OutputIt fill_n( OutputIt first, Size count, const T& value );


// C++17 加入的 policy 
template< class ExecutionPolicy, class ForwardIt, class T >
void fill( ExecutionPolicy&& policy, ForwardIt first, ForwardIt last, const T& value );

template< class ExecutionPolicy, class ForwardIt, class Size, class T >
ForwardIt fill_n( ExecutionPolicy&& policy, ForwardIt first, Size count, const T& value );


// 函数输入的迭代器必须是正向迭代器

// 函数 Complexity: Exactly last - first assignments. 

// (可能的)实现方法
void fill(ForwardIt first, ForwardIt last, const T& value)
{
    for (; first != last; ++first) {
        *first = value;
    }
}

// Container has 12 elements
std::vector<string> data {12}; 
// 给整个vector 赋值字符串 "none"
std::fill (std::begin (data), std::end (data), "none"); 





// 从first 开始 给 count 个值赋值 value
// Complexity: Exactly count assignments, for count > 0. 
// 返回值 : 最后一个操作的对象的迭代器的的下一个  类似于 容器的end()


std::vector<int> v1{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}; 
std::fill_n(v1.begin(), 5, -1);
std::copy(begin(v1), end(v1), std::ostream_iterator<int>(std::cout, " "));
// Output: -1 -1 -1 -1 -1 5 6 7 8 9

```

## 3.3. std::generate std::generate_n

为序列填充同一函数调用值  

```cpp

// generate

template< class ForwardIt, class Generator >
constexpr void generate( ForwardIt first, ForwardIt last, Generator g );
template< class ExecutionPolicy, class ForwardIt, class Generator >
void generate( ExecutionPolicy&& policy, ForwardIt first, ForwardIt last, Generator g );


/* 
first, last 	- 	the range of elements to generate
policy       	- 	the execution policy to use. See execution policy for details.
g            	- 	generator function object that will be called.

The signature of the function should be equivalent to the following:  Ret fun();
The type Ret must be such that an object of type ForwardIt can be dereferenced and assigned a value of type Ret. ​ 
ForwardIt must meet the requirements of LegacyForwardIterator. 

*/

// 可能的实现方法:
template<class ForwardIt, class Generator>
void generate(ForwardIt first, ForwardIt last, Generator g)
{
    while (first != last) {
        *first++ = g();
    }
}


// e.g.

int f()
{ 
    static int i = 1;
    return i++;
}
int main()
{
    std::vector<int> v(5);
    std::generate(v.begin(), v.end(), f);
    // v: 1 2 3 4 5

    // 直接给函数传入 lambda 函数
    // 该调用相当于 numeric 里的 iota(v.begin(), v.end(), 0)
    std::generate(v.begin(), v.end(), [n = 0] () mutable { return n++; });
    // v: 0 1 2 3 4
}


// generate_n
template< class OutputIt, class Size, class Generator >
constexpr OutputIt generate_n( OutputIt first, Size count, Generator g );
template< class ExecutionPolicy, class ForwardIt , class Size, class Generator >
ForwardIt generate_n( ExecutionPolicy&& policy, ForwardIt first, Size count, Generator g );

// return value : Iterator one past the last element assigned if count>0, first otherwise.

// 同理 也是将 last 替换成要填入的值的个数, 返回最后一个填入值的下一个位置的迭代器

int main()
{
    std::mt19937 rng; // default constructed, seeded with fixed seed
    std::generate_n(std::ostream_iterator<std::mt19937::result_type>(std::cout, " "),
                    5, std::ref(rng));
    // Output: 3499211612 581869302 3890346734 3586334585 545404204
}

```

# 4. Sorting operations 

STL 有很多排序算法, 用于适用不同的应用场景  

* `sort (first, last)`
  * 对容器或普通数组中 [first, last) 范围内的元素进行排序，默认进行升序排序。                                                                                                                        
  * `stable_sort (first, last)` 函数功能相似，稳定排序
* `partial_sort (first, middle, last)`
  * 从 [first,last) 范围内，筛选出 `muddle-first` 个最小的元素并排序存放在 [first，middle) 区间中。
* `partial_sort_copy (first, last, result_first, result_last)`
  * 从 [first, last) 范围内筛选出 result_last-result_first 个元素排序并存储到 [result_first, result_last) 指定的范围中。
* `is_sorted (first, last)  `
  * 检测 [first, last) 范围内是否已经排好序，默认检测是否按升序排序。                              
  * `is_sorted_until (first, last)` 如果没有排好序，则该函数会返回指向首个不遵循排序规则的元素的迭代器。                   
* `void nth_element (first, nth, last)`
  * 找到 [first, last) 范围内按照排序规则（默认按照升序排序）应该位于第 nth 个位置处的元素，并将其放置到此位置。
  * 同时使该位置左侧的所有元素都比其存放的元素小，该位置右侧的所有元素都比其存放的元素大。
  * 类似于快速排序内部的单词迭代

应用场景总结:
1. 如果需要对所有元素进行排序，则选择 sort() 或者 stable_sort() 函数；
2. 如果需要保持排序后各元素的相对位置不发生改变，就只能选择 stable_sort() 函数，而另外 3 个排序函数都无法保证这一点；
3. 如果需要对最大（或最小）的 n 个元素进行排序，则优先选择 partial_sort() 函数；
4. 如果只需要找到最大或最小的 n 个元素，但不要求对这 n 个元素进行排序，则优先选择 nth_element() 函数。

nth_element() > partial_sort() > sort() > stable_sort()       <--从左到右，性能由高到低  


sort 函数受到底层实现方式的限制 需要有以下三个条件才能使用
5. 容器支持的迭代器类型必须为**随机访问迭代器**。这意味着，sort() 只对 `array、vector、deque` 这 3 个容器提供支持.
   当操作对象为 `list` 或者 `forward_list` 序列式容器时，其容器模板类中都提供有 `sort()` 排序方法，借助此方法即可实现对容器内部元素进行排序。
6. 如果对容器中指定区域的元素做默认升序排序，则元素类型必须支持<小于运算符；
   同样，如果选用标准库提供的其它排序规则，元素类型也必须支持该规则底层实现所用的比较运算符；
7. sort() 函数在实现排序时，需要交换容器中元素的存储位置。
   如果容器中存储的是自定义的类对象，则该类的内部必须提供移动构造函数和移动赋值运算符。


### 4.0.1. sort() 

sort() 是基于快速排序实现的  复杂度:N*log2(N)    

排序规则:
1. comp 可以是 C++ STL 标准库提供的排序规则（比如 `std::greater<T>`） 或者自定义的规则
2. 可以直接定义一个具有 2 个参数并返回 bool 类型值的函数作为排序规则
3. 给 sort() 函数指定排序规则时，需要为其传入一个函数名,例如 `mycomp` 或者函数对象,例如 `std::greater<int>()` 或者 `mycomp2()`

两种定义
```cpp
void sort (RandomAccessIterator first, RandomAccessIterator last);
// 可以自定义排序规则
void sort (RandomAccessIterator first, RandomAccessIterator last, Compare comp);


// 普通函数作为排序规则
bool mycomp(int i, int j) {
    return (i < j);
}
//以函数对象的方式实现自定义排序规则
class mycomp2 {
public:
    bool operator() (int i, int j) {
        return (i < j);
    }
};


// 使用
std::sort(myvector.begin(), myvector.begin() + 4, std::greater<int>());
std::sort(myvector.begin(), myvector.end(), mycomp);
std::sort(myvector.begin(), myvector.end(), mycomp2());

```

### 4.0.2. stable_sort()

stable_sort() 和 sort() 具有相同的使用场景，就连语法格式也是相同的  

但是 stable_sort() 是基于归并排序实现的  

当可用空间足够的情况下，该函数的时间复杂度可达到`O(N*log2(N))`；反之，时间复杂度为`O(N*log2(N^2))`

### 4.0.3. partial_sort()  partial_sort_copy()

假设这样一种情境，有一个存有 100 万个元素的容器，但我们只想从中提取出值最小的 10 个元素  
使用 sort() 或者 stable_sort() 排序函数, 仅仅为了提取 10 个元素，却要先对 100 万个元素进行排序，可想而知这种实现方式的效率是非常低的。  

平均时间复杂度为N*log(M)，其中 N 指的是 [first, last) 范围的长度，M 指的是 [first, middle) 范围的长度。

函数定义, 和普通sort一样, 拥有 有无自定义规则的两个版本

```cpp
// 定义
void partial_sort (RandomAccessIterator first,RandomAccessIterator middle,
                   RandomAccessIterator last);
void partial_sort (RandomAccessIterator first,RandomAccessIterator middle,
                   RandomAccessIterator last,Compare comp);
      
// 将 myvector 中最小的 4 个元素移动到开头位置并排好序
std::partial_sort(myvector.begin(), myvector.begin() + 4, myvector.end());
// 以指定的 mycomp2 作为排序规则，将 myvector 中最大的 4 个元素移动到开头位置并排好序
std::partial_sort(myvector.begin(), myvector.begin() + 4, myvector.end(), mycomp2());


// 定义
RandomAccessIterator partial_sort_copy (InputIterator first,InputIterator last,
                       RandomAccessIterator result_first,RandomAccessIterator result_last);
RandomAccessIterator partial_sort_copy (InputIterator first,InputIterator last,
                       RandomAccessIterator result_first,RandomAccessIterator result_last,
                       Compare comp);

// 值得一提的是，[first, last] 中的这 2 个迭代器类型仅限定为输入迭代器
// 这意味着相比 partial_sort() 函数，partial_sort_copy() 函数放宽了对存储原有数据的容器类型的限制。
// 即，partial_sort_copy() 函数还支持对 list 容器或者 forward_list 容器中存储的元素进行“部分排序”
// 但是，介于 result_first 和 result_last 仍为随机访问迭代器


```

### 4.0.4. nth_element() 

在有序序列中，我们可以称第 n 个元素为整个序列中“第 n 大”的元素  

nth_element() 函数的功能，当采用默认的升序排序规则（`std::less<T>`）时  
该函数可以从某个序列中找到第 n 小的元素 K，并将 K 移动到序列中第 n 的位置处。  
不仅如此，整个序列经过 nth_element() 函数处理后，所有位于 K 之前的元素都比 K 小，所有位于 K 之后的元素都比 K 大。  

应用场景: 如果只需要找到最大或最小的 n 个元素，但不要求对这 n 个元素进行排序，则优先选择 nth_element() 函数  

```cpp
//排序规则采用默认的升序排序
void nth_element (RandomAccessIterator first,
                  RandomAccessIterator nth,
                  RandomAccessIterator last);
//排序规则为自定义的 comp 排序规则
void nth_element (RandomAccessIterator first,
                  RandomAccessIterator nth,
                  RandomAccessIterator last,
                  Compare comp);

```

* first 和 last：都是随机访问迭代器，[first, last) 用于指定该函数的作用范围（即要处理哪些数据）；
* nth：也是随机访问迭代器，其功能是令函数查找“第 nth 大”的元素，并将其移动到 nth 指向的位置；
* comp：用于自定义排序规则。

### 4.0.5. is_sorted() 和 is_sorted_until()

本就是一组有序的数据，如果我们恰巧需要这样的升序序列，就没有必要再执行排序操作。  

因此，当程序中涉及排序操作时，我们应该为其包裹一层判断语句  

```cpp
if (!is_sorted(mylist.begin(), mylist.end())) {
  // 需要排序
}

if (is_sorted_until(myvector.begin(), myvector.end(),mycomp2()) != myvector.end()){
  // 需要排序
}

// 定义
//判断 [first, last) 区域内的数据是否符合 std::less<T> 排序规则，即是否为升序序列
bool is_sorted (ForwardIterator first, ForwardIterator last);

//判断 [first, last) 区域内的数据是否符合 comp 排序规则  
bool is_sorted (ForwardIterator first, ForwardIterator last, Compare comp);


//排序规则为默认的升序排序
ForwardIterator is_sorted_until (ForwardIterator first, ForwardIterator last);
//排序规则是自定义的 comp 规则
ForwardIterator is_sorted_until (ForwardIterator first,
                                 ForwardIterator last,
                                 Compare comp);
```

* first 和 last 都为正向迭代器（这意味着该函数适用于大部分容器）
* [first, last) 用于指定要检测的序列；
* comp 用于指定自定义的排序规则。 

## 4.1. 自定义排序规则的优化 

数对象可以理解为伪装成函数的对象，根据以往的认知，函数对象的执行效率应该不如普通函数。但事实恰恰相反;   
将普通函数定义为更高效的内联函数，其执行效率也无法和函数对象相比。  

```cpp
//以普通函数的方式实现自定义排序规则
inline bool mycomp(int i, int j) {
    return (i < j);
}
//以函数对象的方式实现自定义排序规则
class mycomp2 {
public:
    bool operator() (int i, int j) {
        return (i < j);
    }
};
```
以 mycomp2() 函数对象为例，其 mycomp2::operator() 也是一个内联函数，  
编译器在对 sort() 函数进行实例化时会将该函数直接展开，这也就意味着，展开后的 sort() 函数内部不包含任何函数调用。  

而如果使用 mycomp 作为参数来调用 sort() 函数，情形则大不相同。要知道，C++ 并不能真正地将一个函数作为参数传递给另一个函数，  
换句话说，如果我们试图将一个函数作为参数进行传递，编译器会隐式地将它转换成一个指向该函数的指针，并将该指针传递过去。  


# 5. 二分查找 Binary Search Operations (On sorted ranges)


* 注意自定义 comp 的使用规则
* 每次执行的是 comp(element,value), 即搜索的value 在后面
* 搜索保证的是 comp 为 true 的一定在 comp 为false 的前面

## 5.1. *_bound

* 和容器的成员 *_bound 的返回值类似
* 返回一个前向迭代器代表
  * lower_bound   : not less than value
  * upper_bound   : greater than value
  * equal_bound   : 返回前两个函数的组合pair, 代表与 value 相等的区间



```cpp
template< class ForwardIt, class T >
constexpr ForwardIt lower_bound( ForwardIt first, ForwardIt last, const T& value );

// 自定义比较规则
// Compare must meet the requirements of BinaryPredicate. 
template< class ForwardIt, class T, class Compare >
constexpr ForwardIt lower_bound( ForwardIt first, ForwardIt last, const T& value, Compare comp );

template< class ForwardIt, class T >
constexpr ForwardIt upper_bound( ForwardIt first, ForwardIt last, const T& value );

// 同样的
template< class ForwardIt, class T, class Compare >
constexpr ForwardIt upper_bound( ForwardIt first, ForwardIt last, const T& value, Compare comp );


template< class ForwardIt, class T >
constexpr std::pair<ForwardIt,ForwardIt>
              equal_range( ForwardIt first, ForwardIt last,
                           const T& value );

template< class ForwardIt, class T, class Compare >
constexpr std::pair<ForwardIt,ForwardIt>
              equal_range( ForwardIt first, ForwardIt last,
                           const T& value, Compare comp );
```

## 5.2. binary_search

* 该函数返回布尔值
* 找到是否有一个元素与 value 相, true if an element equal to value is found, false otherwise. 

函数参数要求同 *_bound 相同
```cpp
template< class ForwardIt, class T >
constexpr bool binary_search( ForwardIt first, ForwardIt last, const T& value );

template< class ForwardIt, class T, class Compare >
constexpr bool binary_search( ForwardIt first, ForwardIt last, const T& value, Compare comp );

```