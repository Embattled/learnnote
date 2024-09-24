# 1. Networking and Interprocess Communication

* 这个模块组存储了所有网络设备相关的内容, 包括内部的处理函数  
* 算是比较低级的模组


# 2. socket - Low-level networking interface¶

provides access to the BSD socket interface
* 该模组在 all modern Unix systems, Windows, MacOS, and probably additional platforms. 都可用
* 在python stl 中是非常低级的模组, 但相比于 C 接口, 仍然做了很大的透明化

注意:  特殊平台  WebAssembly platforms 可能不支持 socket 接口  例如  `wasm32-emscripten` and `wasm32-wasi`

Python socket 接口是 :
* 完全的 Unix 系统调用的封装, 但是基于 Python 的面向对象方法来实现
* 基本的 `socket()` 方法用来获取一个 python 风格的 socket 实例
* 参数的类型不再是 C 语言的低级类型
* 相关的 buffer 内存分配都是自动进行的


## 2.1. Module contents

上一节主要介绍了 python 下的 socket 构成, 这一节记录模组的主要内容, 但是最核心的 socket 类在下一节单独叙述

### 2.1.1. Creating sockets

创建 socket object 的几种方法  


### 2.1.2. Other functions - 和系统网络相关的信息函数

特殊功能函数
* `socket.close(f)`     : 关闭一个 socket file descriptor. 在 Unix 下基本上当作普通文件使用 `os.close()` 即可, 然而在 windows 下有可能会产生特例, 需要使用该函数  



网络信息获取函数  `socket.get*`,  与 `auditing event` 密切配合, 即`sys.audit(event, *args)`
* `getaddrinfo(host, port, family=0, type=0, proto=0, flags=0)`     : 获取对应 host/port 时, 构成一个 socket 所需要的所有信息
  * 参数:
    * `host` : domain name, string of an IPv4/v6 address, `None`
    * `port` : service name such as `http`, numberic port number, `None`
    * 传递 None 相当于给 C 接口传递 `NULL`
    * 参数里的family, type, proto 都是用来缩减返回的 list 长度的, 默认值的 0 代表完整的检测  
    * flags 用于调控该接口的动作模式, 例如 `AI_NUMERICHOST` 将限定 host 只能是数字地址, 而不启用域名解析, 此时传入域名会 raise error
  * return `a sequence of 5-tuples`  : `list[(family, type, proto, canonname, sockaddr),`
    * len(result) : 可能代表了当前 host 所启用的网络服务  
    * 返回值里的 family, type, proto 都是直接用于传递给 `socket()` 函数的  
    * canonname : 只有 flag 里有 `AI_CANONNAME ` 的时候, 会返回 canonical name of the host
    * sockaddr : 根据 family 的种类, 返回可以访问的 `(address, port) 2-tuple`. ipv6 模式下返回 `(address, port, flowinfo, scope_id) 4-tuple`
  * Raises an auditing event socket.getaddrinfo with arguments host, port, family, type, protocol.
* `gethostname()` : 返回当前 python 运行环境的 hostname, 不需要参数
  * Raises an auditing event socket.gethostname with no arguments.

## 2.2. socket Objects - socket 对象

socket 对象的各种 methods, 几乎都是直接对接 Unix 系统调用的, 除了 `makefile()`
socket 对象支持 context manager 调用方法



## 2.3. Notes on socket timeouts 关于网络超时

