# Thread support library

* C++11 内置了多线程编程的库  包括进程互斥 情况变量等内容   
* C++11 新加入了和多线程相关的多个头文件
  * `<thread>`
  * `<mutex>`
  * `<future>`
  * `<condition_variable>`
  * `<atomic>`

# thread

* 从C++11 起加入标准
* 包括了线程的类及其基本操作函数
* 该头文件包括了 `<compare>` 头文件

## classes

总共定义了三个类
1. thread C++11 主要的线程类
2. jthread C++20 新加入的支持自动joining和中止的线程类
3. `std::hash<std::thread::id>`  specializes std::hash


### thread

线程类的特征:
* 要执行的函数通过构造函数传入, 在对象建立的时候立即开始执行
* 函数返回值会被忽略
* 如果函数发生了异常, 会调用 `std::terminate`
* 函数的返回值或者异常可以通过 `std::promise` 传递给进程调用者
* 不会有两个对象指向同一个执行, 线程对象可以移动, 但是没有拷贝构造函数, 不可以赋值

#### 构造方法

```cpp
// 默认/空 构造函数
thread() noexcept;
// 移动构造函数
thread( thread&& other ) noexcept;
// 拷贝构造函数=delete 代表不可拷贝, 直接删掉
thread( const thread& ) = delete;

// 真正有用的构造函数
// Creates new std::thread object and associates it with a thread of execution
template< class Function, class... Args > explicit thread( Function&& f, Args&&... args );
// 1. Function &&f  : 一个可以被调用的对象
// 2. args          : 要传递给 f 的参数


```
#### member classes id

是thread类中的一个子类 `class thread::id;`  

* 轻量化的, 可拷贝的类, 作为一个线程唯一的识别器 只有一个默认构造函数
* 该类的值有可能不代表任何一个线程
* 该类的值有可能已经过期, 即原本的线程已结束, 而该值被分配给了新的线程

实现了比较运算和流输出运算  

#### get_id()

`std::thread::id get_id() const noexcept;`

返回一个和当前线程关联的 std::thread::id 

## function

主要功能都作为类的方法定义了, 因此函数部分相对简单, 都是一些辅助功能   

## 
