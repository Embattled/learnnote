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

## 2.1. Older high-level API

由于 run 是3.5 才被加入的, 所以 older api 也很重要, 用来保持与旧版本的兼容性

`subprocess.call(args, *, stdin=None, stdout=None, stderr=None, 参数省略)`
* 阻塞调用子进程
* 返回 `returncode` 属性

`subprocess.check_call(args, *, stdin=None, stdout=None, stderr=None, 参数省略)` 



## 2.2. subprocess.run

尽管是最关键的函数,  run 其实是 python3.5 才加入的功能

具有的新功能:
* 捕获子进程的 stdout 和 stderr
* 

`subprocess.run(args, *, 其他参数, **other_popen_kwargs)`
* Run the command described by args. Wait for command to complete, then return a `CompletedProcess` instance.
  * args 是系统终端命令
  * 该函数默认是阻塞的
  * 返回的是一个实例


## 2.3. class subprocess.Popen

整个 subprocess 模组最终要的类, 作为其他类和函数的底层实现, 用于实际上的创建和管理子线程  
* 模组的其他紫禁城相关函数, 都是传入该类的构造函数中
* 用于和系统的实际子函数创建接口进行交互, POSIX: `os.execvpe()`, windows : `CreateProcess()`

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



## 2.4. class subprocess.CompletedProcess

The return value from run(), representing a process that has finished.


# 3. threading — Thread-based parallelism 基于线程的并行


# 4. multiprocessing — Process-based parallelism 基于进程的并行

API的构造上和 threading 相似
* 支持  both local and remote concurrency
* Effectively side-stepping the Global Interpreter Lock by using subprocesses instead of threads.
* Fully leverage multiple processors on a given machine.

