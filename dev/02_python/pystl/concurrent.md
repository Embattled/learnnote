# 1. Concurrent Execution

Python STL 专门用于并行处理的章节 

The appropriate choice of tool will depend on the:
* task to be executed(CPU bound vs IO bound)
* preferred style of development (event driven cooperative multitasking vs preemptive multitasking). 


同时, 关于嵌套调用的新模组也放在该章节 (subprocess)

# 2. subprocess - Subprocess management 子进程

`subprocess` 模组允许程序创建一个新的 子进程, 并同时链接其 input/output/error 流, 获取其返回代码

该模组旨在替换 os 模组中的 system 和 spawn* 命令集, 使 os 负责的功能更清晰
* `os.system`
* `os.spawn*`

全局简介
* subprocess.run : 阻塞调用子进程


## 2.1. subprocess.run

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

### 2.1.1. CompletedProcess

<!-- 完 -->
run 函数的返回值 `class subprocess.CompletedProcess` , 代表了一个子进程的结束, 可以从该类里获取一些信息  

* `args`  : process 的CLI参数, 可以是 list or a string
* `returncode`  : 0 代表正常结束 
  * ※ A negative value `-N` indicates that the child was terminated by signal N (POSIX only).
* stdout : 流输出捕捉
* stderr : 错误信息捕捉
* `check_returncode()` : 主动查验并报错

### 2.1.2. Other Constant
<!-- over -->
模组里的一些实用常量

* `subprocess.DEVNULL`  : can be used as the stdin, stdout or stderr argument to `Popen`. indicates that the special file `os.devnull` will be used.
* `subprocess.PIPE`     :  can be used as the stdin, stdout or stderr argument to `Popen`. indicates that a pipe to the standard stream should be opened.
* `subprocess.STDOUT`   : can be used as the stderr argument to `Popen`. indicates that standard error should go into the same handle as standard output.


## 2.2. class subprocess.Popen

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





## 2.3. Older high-level API

由于 run 是3.5 才被加入的, 所以 older api 也很重要, 用来保持与旧版本的兼容性

`subprocess.call(args, *, stdin=None, stdout=None, stderr=None, 参数省略)`
* 阻塞调用子进程
* 返回 `returncode` 属性

`subprocess.check_call(args, *, stdin=None, stdout=None, stderr=None, 参数省略)` 


# 3. threading — Thread-based parallelism 基于线程的并行


# 4. multiprocessing — Process-based parallelism 基于进程的并行

API的构造上和 threading 相似
* 支持  both local and remote concurrency
* Effectively side-stepping the Global Interpreter Lock by using subprocesses instead of threads.
* Fully leverage multiple processors on a given machine.

