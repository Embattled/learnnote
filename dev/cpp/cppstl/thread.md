# 1. Thread support library

* C++11 内置了多线程编程的库  包括进程互斥 情况变量等内容   
* C++11 新加入了和多线程相关的多个头文件
  * `<thread>`
  * `<mutex>`
  * `<future>`
  * `<condition_variable>`
  * `<atomic>`
* 与C++11 相匹配的, C11 也加入了C源生的多线程相关的内容  `<threads.h>`
  * C11 的多线程功能都集中在了这一个头文件, 包括所有锁的实现

* 对于C11的多线程头文件 `<thread.h>`
* 如果宏常量 `__STDC_NO_THREADS__`被定义, 则说明 `<thread.h>` 在该系统中不被提供, 不能够使用



# 2. thread

* 从C++11 起加入标准
* 包括了线程的类及其基本操作函数
* 该头文件包括了 `<compare>` 头文件

C++ 线程的特征:
* 不再像 POSIX 标准头文件中那样需要 线程所执行函数的参数和返回值都必须为 `void*` 类型
* C++ thread 类创建的线程可以执行任意函数


## 2.1. classes

总共定义了三个类
1. thread C++11 主要的线程类
2. jthread C++20 新加入的支持自动joining和中止的线程类
3. `std::hash<std::thread::id>`  specializes std::hash, 用来对线程的 id 进行 hash


### 2.1.1. thread

线程类的特征:
* 要执行的函数通过构造函数传入, 在对象建立的时候立即开始执行
* 函数返回值会被忽略
* 如果函数发生了异常, 会调用 `std::terminate`
* 函数的返回值或者异常可以通过 `std::promise` 传递给进程调用者
* 不会有两个对象指向同一个执行, 线程对象可以移动, 但是没有拷贝构造函数, 不可以赋值



#### 2.1.1.1. 构造方法

```cpp
// Creates new std::thread object and associates it with a thread of execution
template< class Function, class... Args > 
explicit thread( Function&& f, Args&&... args );
// 1. Function &&f  : 一个可以被调用的对象
// 2. args          : 要传递给 f 的参数


// 默认/空 构造函数
thread() noexcept;
// 拷贝构造函数=delete 代表不可拷贝, 直接删掉
thread( const thread& ) = delete;
// 移动构造函数. 代表可以将临时的(匿名的) 线程对象赋值给另一个
thread( thread&& other ) noexcept;
```


#### 2.1.1.2. 线程操作方法

成员方法|功能
-|-
`get_id()`| 获取线程 id
`joinable()`| 判断当前线程是否支持 join 函数
`join()`| 阻塞当前线程, 直到对象线程执行完毕
`detach()`| 将对象线程从当前线程中分离出去, 不再能够 join
`swap()`| 交换两个线程的状态

* 所有线程对象必须使用以下一种方法来安全析构销毁
  * 使用 `detach()` 分离
  * 使用 `join()` 阻塞等待对象线程安全析构
* 否则会导致内存相关的异常
  * 线程占用的内存不能正常释放, 内存泄漏
  * 子线程未执行完, 但主线程直接结束, 导致程序引发异常


#### 2.1.1.3. member classes id

是thread类中的一个子类 `class thread::id;`  

* 轻量化的, 可拷贝的类, 作为一个线程唯一的识别器 只有一个默认构造函数
* 该类的值有可能不代表任何一个线程
* 该类的值有可能已经过期, 即原本的线程已结束, 而该值被分配给了新的线程

实现了比较运算和流输出运算  

#### 2.1.1.4. get_id()

`std::thread::id get_id() const noexcept;`

返回一个和当前线程关联的 std::thread::id 

## 2.2. function

主要功能都作为类的方法定义了, 因此函数部分相对简单, 都是一些辅助功能   


# 3. mutex

* 同 POSIX 标准一样, C++11 也有多种锁的封装实现
* 但是将多种锁分布在了不同的头文件中

```cpp
std::mutex mtx;

void threadFun() {
    while(n<10){
        //对互斥锁进行“加锁”
        mtx.lock();
        n++;
        cout << "ID" << std::this_thread::get_id() << " n = "<< n << endl;
        //对互斥锁进行“解锁”
        mtx.unlock();
        //暂停 1 秒
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

```

# 4. condition_variable

* 条件变量的头文件名称就是 `condition_variable` (超长)
  

