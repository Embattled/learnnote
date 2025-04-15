# 1. Concurrent Execution

Python STL 专门用于并行处理的章节 

The appropriate choice of tool will depend on the:
* task to be executed(CPU bound vs IO bound)
* preferred style of development (event driven cooperative multitasking vs preemptive multitasking). 


同时, 关于嵌套调用的新模组也放在该章节 (subprocess)


编程思路: 通过 多线程/多进程 实现高速化处理的可能性取决于该任务是 IO-bound / CPU-bound (IO密集型还是CPU密集型)
* 多线程适合 IO-bound 任务, 更加容易扩展, 多线程的上线取决于 CPU (各个线程轮询, 分别等待 IO, 某一个线程等待 IO 的适合其他线程执行计算) 需要线程锁
* 多进程 适合 CPU-oound, 多个进程在单独的和核心上运行, 有自己的内存空间, 因此不需要线程锁, 但是需要手动设置进程之间的必要数据复制以及对应的锁

# 2. threading — Thread-based parallelism 基于线程的并行

基于多线程的程序并行执行, 对于计算密集型的任务可能不会有本质上的提速  




# 3. multiprocessing — Process-based parallelism 基于进程的并行

此模块在 WebAssembly 平台 wasm32-emscripten 和 wasm32-wasi 上不适用或不可用

API的构造上和 threading 相似
* 支持  both local and remote concurrency
* Effectively side-stepping the Global Interpreter Lock by using subprocesses instead of threads.
  * 通过使用子进程而非线程有效地绕过了 全局解释器锁
* Fully leverage multiple processors on a given machine.

属于性能强大的多进程, **一个缺点就是启动时间很长**, 因此需要比较重的任务才能体现出优点
(Python 3.12 以后提出的一个新的 sub-interpreters 相当于轻量化的 multiprocessing, 启动时间较短)


## 3.1. User Guide

### 3.1.1. The Process class

通过创建一个 Process 对象 并执行其 start 方法来生成一个并行计算的 **进程**


```py
from multiprocessing import Process

def f(name):
    print('hello', name)

if __name__ == '__main__':
    p = Process(target=f, args=('bob',))
    p.start()
    p.join()

```

### 3.1.2. Contexts and start methods - 上下文和启动方法

关于具体的进程启动方法  
根据不同的平台, 有不同的实现方法  

spawn:  
* 父进程会启动一个新的 Python 解释器进程, 子进程会继承所需要的计算数据
* 来自夫进程的 非必须文件描述符和 句柄 不会被继承  
* 最慢的方法
* 在 POSIX 和 Windows 平台上可用, 在 Windows 和 MacOS 上为默认方法

fork:  
* 使用 `os.fork` 来产生 python 解释器分叉  
* 子进程在开始的时候实际上与父进程相同, 父进程的**所有资源**都继承给子进程, 这种方法的缺点是很难实现多线程安全  
* POSIX 系统上可用, 在除了 MacOS 之外的所有 POSIX 上为默认
  * 在 Python 3.14 上将不再为默认
  * 在多线程里面使用这种多进程方法 会报 DeprecationWarning

forkserver:  
* 产生一个单独的 服务器进程, 且该进程是单线程的(除非因为系统库或者其他预加载导入的特性导致改变)  
  * 使用 os.fork() 是安全的
* 每需要一个新进程的时候, 父进程就会连接到该服务器并请求它 分叉一个新进程  
* 在**支持通过 Unix 管道传递文件描述符 的 POSIX 平台**上可用, 例如 Linux

在程序中, 应当手动选择要启动进程的具体方法  
```py

import multiprocessing as mp

def foo(q):
    q.put('hello')

if __name__ == '__main__':
    # 手动为全局设置默认子进程启动方式, 该函数不应该被调用多次
    mp.set_start_method('spawn')
    q = mp.Queue()
    p = mp.Process(target=foo, args=(q,))
    p.start()
    print(q.get())
    p.join()


# 或者通过 get_context 获取上下文对象, 来在同一个程序中使用多种启动方式  
def foo(q):
    q.put('hello')

if __name__ == '__main__':
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    p = ctx.Process(target=foo, args=(q,))
    p.start()
    print(q.get())
    p.join()

```

### 3.1.3. Exchanging objects between processes - 在进程之间交换对象

所谓的交换对象, 应就是两个进程, 一个发送一个接受的形式  

multiprocessing 支持两种数据格式的通信通道
* Queue :单向传递数据?
* Pipe  : 支持同一个对象数据交换

```py
from multiprocessing import Process, Queue

def f(q):
    q.put([42, None, 'hello'])

if __name__ == '__main__':
    q = Queue()
    p = Process(target=f, args=(q,))
    p.start()
    print(q.get())    # 打印 "[42, None, 'hello']"
    p.join()


from multiprocessing import Process, Pipe

def f(conn):
    conn.send([42, None, 'hello'])
    conn.close()

if __name__ == '__main__':
    # 方法返回一个 Pipe 的两端, 双方都有 send() 和 recv() 方法
    # 如果两个进程尝试同时读取或者写入管道的同一端  数据会损坏  
    parent_conn, child_conn = Pipe()
    p = Process(target=f, args=(child_conn,))
    p.start()
    print(parent_conn.recv())   # 打印 "[42, None, 'hello']"
    p.join()

```

### 3.1.4. Synchronization between processes 进程间同步

再次声明 multiprocessing 包含来自 threading 的所有同步原语的等价物  

因此锁机制也是相同的可以使用的  

不使用锁的情况下, 来自于多进程中如果有输出的话很容易产生混淆

```py
from multiprocessing import Process, Lock

def f(l, i):
    l.acquire()
    try:
        print('hello world', i)
    finally:
        l.release()

if __name__ == '__main__':
    lock = Lock()

    for num in range(10):
        Process(target=f, args=(lock, num)).start()
```


### 3.1.5. Sharing state between processes 进程间共享状态

在 并行/并发 编程的时候, 最好避免使用共享状态, 多进程的时候尤其如此  

但是如果确实需要使用一些共享数据的话, multiprocessing 提供了两种方法  
* 共享内存
  * 将数据存储在共享内存映射中
  * 仅支持预定义的数据种类? 
  * `multiprocessing.sharedctypes` 支持共享 更广泛的 任意 ctypes 对象  

```py
from multiprocessing import Process, Value, Array

def f(n, a):
    n.value = 3.1415927
    for i in range(len(a)):
        a[i] = -a[i]

if __name__ == '__main__':
    # 两种预定义的共享数据对象
    # 字符表示 array 模块的 typecode, 可能因为是源于C 的接口所以需要定义数据类型  
    num = Value('d', 0.0)
    arr = Array('i', range(10))

    p = Process(target=f, args=(num, arr))
    p.start()
    p.join()

    print(num.value)
    print(arr[:])

```



* 服务进程
  * 创建一个服务器进程, 该进程保存一些对象, 其他进程可以通过代理的方式来操作这些数据
  * 支持任意对象类型
    * 支持 ` list, dict, Namespace, Lock, RLock, Semaphore, BoundedSemaphore, Condition, Event, Barrier, Queue, Value and Array`
  * 甚至支持通过网络来在不同计算机上的不同进程来共享
  * 速度较慢


```py

from multiprocessing import Process, Manager

def f(d, l):
    d[1] = '1'
    d['2'] = 2
    d[0.25] = None
    l.reverse()

if __name__ == '__main__':
    with Manager() as manager:
        d = manager.dict()
        l = manager.list(range(10))

        p = Process(target=f, args=(d, l))
        p.start()
        p.join()

        print(d)
        print(l)
```



### 3.1.6. Using a pool of workers 使用工作池


Pool 类表示一个工作进程池  
它具有允许以几种不同方式将任务分配到工作进程的方法  

看起来是最实用的方法

```py

from multiprocessing import Pool, TimeoutError
import time
import os

def f(x):
    return x*x

if __name__ == '__main__':
    # 启动 4 个工作进程
    with Pool(processes=4) as pool:

        # 打印 "[0, 1, 4,..., 81]"
        print(pool.map(f, range(10)))

        # 以任意顺序打印同样的数字
        for i in pool.imap_unordered(f, range(10)):
            print(i)

        # 异步地对 "f(20)" 求值
        res = pool.apply_async(f, (20,))      # runs in *only* one process
        print(res.get(timeout=1))             # prints "400"

        # 异步地对 "os.getpid()" 求值
        res = pool.apply_async(os.getpid, ()) # *仅在* 一个进程中运行
        print(res.get(timeout=1))             # 打印进程的 PID

        # 异步地进行多次求值 *可能* 会使用更多进程
        multiple_results = [pool.apply_async(os.getpid, ()) for i in range(4)]
        print([res.get(timeout=1) for res in multiple_results])

        # 让一个工作进程休眠 10 秒
        res = pool.apply_async(time.sleep, (10,))
        try:
            print(res.get(timeout=1))
        except TimeoutError:
            print("We lacked patience and got a multiprocessing.TimeoutError")

        print("For the moment, the pool remains available for more work")

    # 退出 'with' 代码块将停止进程池
    print("Now the pool is closed and no longer available")

```




## 3.2. API Reference


### 3.2.1. Process Pools 进程池

One can create a pool of processes which will carry out tasks submitted to it with the Pool class.
交给进程池的任务会自动分配给不同的进程
* 进程池对象, 控制提交到进程池的工作
* 带有 能够 超时管理 异步获取结果的 回调函数
* 有一个并行的 `map` 实现  

`class multiprocessing.pool.Pool([processes[, initializer[, initargs[, maxtasksperchild[, context]]]]])`
* 参数:
  * `processes` : 进程数, 默认使用 `os.cpu_count()`, 即 CPU 吃满
  * `initializer` : 进程的初始化函数
    * 如果 部位 None, 则会在进程的工作启动之前调用 `initializer(*initargs)`
  * `maxtasksperchild` : 每一个进程在它被退出之前(或者被新的进程替代) 能够执行的任务上限, 主要是为了释放未使用的资源
    * 默认值为 None, 即所有进程都和 pool 寿命相同
    * 该参数用于提前释放一定的进程资源
  * `context`
    * 上下文管理器, 通常 pool 都会与 `with` 语法一起使用, 此时上下文对象会被自动设置
    * 直接调用库的 `multiprocessing.Pool()` 类的接口也没有问题, 也会设置好上下文对象  
  
Pools 的类方法没有那么多  
* `apply(func[, args[, kwds]])`
  * 标准的 阻塞方法的调用 func, 只会使用进程池中的一个工作进程  
  * 如果需要并行执行任务, 使用 apply_async 更好
* `apply_async(func[, args[, kwds[, callback[, error_callback]]]])`
  * 返回一个 `AsyncResult` 对象
  * 回调函数: callback error_callback
    * 两个 都是接受单个参数的函数对象
    * 分别处理调用 func 执行成功或者失败时候的情况
    * 回调函数的处理应当非常简单, 否则会阻塞主线程  
* `map(func, iterable[, chunksize])`
  * 功能上与 built-in map 相同, 但是会并行执行, 且不是 yeild 方法执行的
    * 这点 `imap` 才更加贴近 built-in `map`
    * 同 built-in map 不同, 该方法只支持单个 iterable
    * 如果要迭代多个参数, 参考 `starmap()`
    * 会阻塞的执行
  * 将 iterable 分片 chops 为对应的线程池大小
    * 参数 `chunksize` 则用于独立的指定 分片的任务数量
    * 每一个 chunksize 会作为一个任务集合分配给一个子进程
      * 子进程处理完 chunk 后一起返回处理结果
    * chunksize 越大, 负载不均匀的可能性越大
  * 对于长的 iterable, 会导致很大的内存占用
    * 因为会计算所有的结果然后一次性返回 list
    * 可以使用 `imap()` 或者 `imap_unordered()` 并且明确 chunksize 来提高效率 
* `starmap(func, iterable[, chunksize])`
  * 同 map 相同, 但是支持向函数传入多个参数 (参数预先打包为 `list[tuple]`)
* `map_async(func, iterable[, chunksize[, callback[, error_callback]]])`
  * 异步的执行, 返回的不是函数处理结果而是 `AsyncResult` 对象
  * callback 函数的定义同上面相同
* `starmap_async(func, iterable[, chunksize[, callback[, error_callback]]])`
  * `strmap()` + `map_async()` 的结合体
* `imap(func, iterable[, chunksize])`
  * A lazier version of map()
  * 按照 yeild 方法返回, 内存效率更高
  * 如果 chunksize 设置为 1
    * 那么在用 `next()` 迭代返回值的时候 可以传入超时参数
    * `next(timeout)`, 如果超时的话会引起 `multiprocessing.TimeoutError`
  * `即使某个 chunk 先完成了，imap 也会等前面的 chunk 返回后才继续吐出结果`
* `imap_unordered(func, iterable[, chunksize])`
  * 对比 imap 速度更快但是不会保证返回值的顺序
  * 除非线程池的 worker 只有 1 
* `close()`
  * 组织任何新任务提交到池中
  * 所有任务完成后线程池将会 `exit`
* `terminate()`
  * 立即停止所有  task
  * 线程池被垃圾回收的时候该方法会被自动调用  
* `join()`
  * 等待线程池 `exit`
  * 因此必须在调用 join 之前调用 `close ` 或者 `terminate`

对于 `chunksize`, 默认的自动计算方法为:  
`chunksize = max(1, len(iterable) // (4 * processes))`

推荐策略
* 任务重 / CPU 密集	保持 chunksize 小一些 1-5
* 任务轻 / I/O 密集	chunksize 大一些 10-100 
* 任务非常多	加大 chunksize, 减少调度频率
* 任务量不均 部分很慢	用小 chunksize + imap_unordered


`class multiprocessing.pool.AsyncResult`
* 异步执行时候的返回值结果类 : `Pool.apply_async() and Pool.map_async()`
* `get([timeout])`
  * 获取任务执行的结果
  * 如果 任务 唤起了异常, 则异常会传递给 get
  * timeout 用于设置超时, 并根据情况唤起 `multiprocessing.TimeoutError`
* `wait([timeout])`
  * 应该是阻塞该任务
* `ready()`
  * 验证该任务是否完成, 返回 bool
* `successful()`
  * 返回 该任务是否完成, 并且没有异常的执行完成
  * 如果任务尚未完成 if the result is not ready 同样会唤起异常 `ValueError`
  * 想象不出来单独使用的机会, 需要配合其他阻塞方法


# 4. concurrent.futures — Launching parallel tasks

截止 python 3.12, The concurrent package 里面只有一个 concurrent.futures 一个子包

里面包含了并行执行程序的最高级的接口, 看起来使用非常方便


# 5. subprocess - Subprocess management 子进程

`subprocess` 模组允许程序创建一个新的 子进程, 并同时链接其 input/output/error 流, 获取其返回代码

该模组旨在替换 os 模组中的 system 和 spawn* 命令集, 使 os 负责的功能更清晰
* `os.system`
* `os.spawn*`

全局简介
* subprocess.run : 阻塞调用子进程


## 5.1. subprocess 的使用

尽管是最关键的函数,  run 其实是 python3.5 才加入的功能

`subprocess.run(args, *, 其他参数, **other_popen_kwargs)`


* Run the command described by args. Wait for command to complete, then return a `CompletedProcess` instance.
  * args 是系统终端命令
  * 该函数默认是阻塞的
  * 返回的是一个实例

很多的参数都是传递给了 Popen 类的构造函数, 除此之外的在该章节介绍:  

捕获输出
* `capture_output=False`        : 捕获子进程的 stdout 和 stderr
  * 捕获子进程的输出和 err, 如果该参数为真, 那么会屏蔽掉 `stdout` 和 `stderr` 的值
  * 捕获到的输出可以通过该函数返回的类进行访问
  * 实际上进行的操作是 `Popen` object is automatically created with `stdout=PIPE` and `stderr=PIPE`
  * 如果要 capture and combine both streams into one, use `stdout=PIPE` and `stderr=STDOUT` instead of `capture_output`

传递给 `Popen.communicate()` 的参数
* `timeout=None`                : 用于子进程的超时中止, 超时的话会触发异常 `TimeoutExpired`
* `input=None`                  : 用于对子进程进行通信, 具体为输入的内容会传入子进程的 stdin 
  * 具体实现为 `Popen` object is automatically created with `stdin=PIPE`
  * 同时会屏蔽掉另一个 `stdin` 的输入值
  * 传输入的内容有要求: 
  * it must be a byte sequence, or a string if 这三个参数 `encoding` or `errors` is specified or `text` is true.

执行结果检查 `check=False`: 
* 如果进程的返回的值不为 0 , 则会触发异常 ` CalledProcessError`
* 此时由于函数没有正常结束, 所以 exit code 可以从异常对象获取, 同理  stdout and stderr if they were captured.

### 5.1.1. CompletedProcess

<!-- 完 -->
run 函数的返回值 `class subprocess.CompletedProcess` , 代表了一个子进程的结束, 可以从该类里获取一些信息  

* `args`  : process 的CLI参数, 可以是 list or a string
* `returncode`  : 0 代表正常结束 
  * ※ A negative value `-N` indicates that the child was terminated by signal N (POSIX only).
* stdout : 流输出捕捉
* stderr : 错误信息捕捉
* `check_returncode()` : 主动查验并报错

### 5.1.2. Other Constant
<!-- over -->
模组里的一些实用常量

* `subprocess.DEVNULL`  : can be used as the stdin, stdout or stderr argument to `Popen`. indicates that the special file `os.devnull` will be used.
* `subprocess.PIPE`     :  can be used as the stdin, stdout or stderr argument to `Popen`. indicates that a pipe to the standard stream should be opened.
* `subprocess.STDOUT`   : can be used as the stderr argument to `Popen`. indicates that standard error should go into the same handle as standard output.


## 5.2. Popen Constructor 

class subprocess.Popen

整个 subprocess 模组最终要的类, 作为其他类和函数 `subprocess.run` 的底层实现, 用于实际上的创建和管理子线程  
* 模组的其他子进程相关函数, 大部分传入该类的构造函数中
* 用于和系统的实际子函数创建接口进行交互, POSIX: `os.execvpe()`, windows : `CreateProcess()`

完整函数定义用于查找
```py
class subprocess.Popen(args, bufsize=- 1, executable=None, stdin=None, stdout=None, 
stderr=None, preexec_fn=None, close_fds=True, shell=False, cwd=None, env=None, 
universal_newlines=None, startupinfo=None, creationflags=0, restore_signals=True, 
start_new_session=False, pass_fds=(), *, group=None, extra_groups=None, user=None, 
umask=- 1, encoding=None, errors=None, text=None, pipesize=- 1, process_group=None)

```


参数说明:
* `args`
  * 必须参数, str or sequence or path-like
  * 如果是 sequence, 那么 args 的第一个元素必须是要执行的程序地址
* `bufsize=- 1`
* `executable=None`
* `stdin=None`
* `stdout=None`
* `stderr=None`
  * 三个流参数用于指定子程序的输入输出, 必须是:
    * `PIPE`
    * `DEVNULL`
    * `None`
    * an existing file descriptor (a positive integer)
    * an existing file object with a valid file descriptor
* `preexec_fn=None`
* `close_fds=True`
* `shell=False`
  * 如果 args 是一个单个字符串, 那么 `shell = True` 将会完整的将 args 作为 shell 里输入的内容传递过去
  * 对于要用到一些 shell 特征的情况, 如管线, 非常有用
* `cwd=None`
* `env=None`
* `universal_newlines=None`
* `startupinfo=None`
* `creationflags=0`
* `restore_signals=True`
* `start_new_session=False`
* `pass_fds=()`
* `*`
* `group=None`
* `extra_groups=None`
* `user=None`
* `umask=- 1`
* `encoding=None`
* `errors=None`
* `text=None`
* `pipesize=- 1`





## 5.3. Older high-level API

由于 run 是3.5 才被加入的, 所以 older api 也很重要, 用来保持与旧版本的兼容性

`subprocess.call(args, *, stdin=None, stdout=None, stderr=None, 参数省略)`
* 阻塞调用子进程
* 返回 `returncode` 属性

`subprocess.check_call(args, *, stdin=None, stdout=None, stderr=None, 参数省略)` 


