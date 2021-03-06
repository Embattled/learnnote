# 1. socket 套接字

学习计算机之间如何通讯, 并用程序实现

* 通常套接字都指的是 Internet套接字, 但是也存在其他套接字
* 创建套接字必须指定数据传输方式
  * 流格式套接字 (Stream Sockets)  TCP 协议
  * 数据报格式套接字 (Datagram Socket) UDP 协议


## 1.1. Linux

* UNIX/Linux 系统中, 为了统一对各种硬件的操作, 简化接口, 不同的硬件设备也都被看成一个文件
* 因此, 网络连接也是一个文件, 网络通信也是对一个文件进行I/O, 使用一般的文件读写命令即可
  * `read() write()`
  * `close()`


## 1.2. Windows

* Windows 会区分 socket 和文件
* 需要调用专门针对 socket 而设计的数据传输函数
  * `recv() send()`
  * `closesocket()`

* windows 下的 socket 运行需要 dll 文件 `Winsock.dll 或 ws2_32.dll`, 有两个版本
  * 早些的是 `wsock32.dll` 头文件为 `winsock1.h`  不需要考虑
  * 最新的是 `ws2_32.dll`  头文件为 `winsock2.h`  已覆盖所有常见 windows
  * 动态链接库的加载可以使用 `#pragma comment (lib, "ws2_32.lib")`
* 在正式调用 dll 函数前还需要初始化 `WSAStartup()`
  * `int WSAStartup(WORD wVersionRequested, LPWSADATA lpWSAData);`
  * 用于指定 WinSock 规范的版本 
  * 最新版本号为 2.2, 较早的有 2.1、2.0、1.1、1.0, `ws2_32.dll` 支持所有版本


```cpp
// lpWSAData 为指向 WSAData 结构体的指针
// WSAData 的定义:
// szDescription 和 szSystemStatus 包含的信息基本没有实用价值
typedef struct WSAData {
    WORD           wVersion;            //ws2_32.dll 建议我们使用的版本号
    WORD           wHighVersion;        //ws2_32.dll 支持的最高版本号

    //一个以 null 结尾的字符串, 用来说明 ws2_32.dll 的实现以及厂商信息
    char           szDescription[WSADESCRIPTION_LEN+1];

    //一个以 null 结尾的字符串, 用来说明 ws2_32.dll 的状态以及配置信息
    char           szSystemStatus[WSASYS_STATUS_LEN+1];
    unsigned short iMaxSockets;     //2.0以后不再使用
    unsigned short iMaxUdpDg;       //2.0以后不再使用
    char FAR       *lpVendorInfo;   //2.0以后不再使用
} WSADATA, *LPWSADATA;


// wVersionRequested 参数用来指明我们希望使用的版本号, 它的类型为 WORD, 等价于 unsigned short
// 需要用 MAKEWORD() 宏函数对版本号进行转换
MAKEWORD(1, 2);  //主版本号为1, 副版本号为2, 返回 0x0201
MAKEWORD(2, 2);  //主版本号为2, 副版本号为2, 返回 0x0202


// 使用方法 main:

// 创建一个空
WSADATA wsaData;
// 初始化并通过 wsaData 可以获取一些信息
WSAStartup( MAKEWORD(2, 2), &wsaData);

//  ... 正式的 socket 创建
```
## 1.3. protocol

* socket 是协议的实现
* 现在的socket 都基于 TCP和UDP协议


## 1.4. port

端口号, 计算机会为每个网络程序分配一个独一无二的端口号

* Web 80
* FTP 21
* SMTP 25

# 2. socket 编程


## 2.1. 创建 socket

* 不管 linux 还剩 win, `socket()` 函数及其参数都是相同的, 区别在于返回值不同
* Linux 中一切都是文件, 因此返回一个 文件描述符
* Windows中会区分 socket, 因此会返回一个 `SOCKET` 类型

**头文件**
* Linux : `<sys/socket.h>`
* Win   : `winsock2.h`


`returnValue socket(int af, int type, int protocol);`
1. af (Address Family) : IP地址类型
   * `AF_INET6` 表示IPv6  `PF_INET6` 等价
   * `AF_INET`  表示IPv4  `PF_INET`  等价
2. type         : 数据传输方式 和后面的协议TCP/UDP对应
   * `SOCK_STREAN`
   * `SOCK_DGRAM`
3. protocol     : 传输协议, 因为大多数情况下 type已经决定了协议, 因此可以传入 0
   * `IPPROTO_TCP`
   * `IPPROTO_UDP`

```cpp

//IPPROTO_TCP表示TCP协议
returnValue tcp_socket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);

//IPPROTO_UDP表示UDP协议
retuenValue udp_socket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);  


returnValue tcp_socket = socket(AF_INET, SOCK_STREAM, 0);  //创建TCP套接字
returnValue udp_socket = socket(AF_INET, SOCK_DGRAM, 0);  //创建UDP套接字

```

## 2.2. 端口绑定 bind

* 创建好 socket 后, 需要将其绑定到本机的 ip 以及 port 上
* 这样流经该port的数据才能被程序所接收到
* 对于流传输(TCP)
  * 服务端用 bind() 绑定
  * 客户端用 connect() 建立连接
* `connect()` 用于建立链接, 所有参数都和 `bind()` 相同


总结:
* 绑定的关键参数是 `struct sockaddr`  
* 相关的结构体还有 `struct sockaddr_in`
  * sockaddr 整合了地址和端口, 因此不好赋值
  * sockaddr_in 分离了地址和端口, 容易赋值, 而且可以强制类型转换不出错
* 可以理解为 `sockaddr` 是一种通用的结构体, `sockaddr_in` 是IPv4专用的结构体
* 相对应的 `sockaddr_in6` 是IPv6 专用结构体


```cpp
int bind(   int sock,       struct sockaddr *addr, socklen_t addrlen);  //Linux
int bind(SOCKET sock, const struct sockaddr *addr, int addrlen);  //Windows

int connect(int sock, struct sockaddr *serv_addr, socklen_t addrlen);  //Linux
int connect(SOCKET sock, const struct sockaddr *serv_addr, int addrlen);  //Windows


// sockaddr_in 结构体原型
struct sockaddr_in{
    sa_family_t     sin_family;   //地址族（Address Family）, 也就是地址类型
    uint16_t        sin_port;     //16位的端口号
    struct in_addr  sin_addr;     //32位IP地址
    char            sin_zero[8];  //不使用, 一般用0填充
};
// 1. sin_family 地址类型, 和 socket 函数的第一的参数需要保持一致
// 2. sin_port   端口号, 理论上 0~65535 取值, 传入的时候需要使用 htons() 进行转换
// 3. sin_addr   in_addr的结构体 用于指定地址的结构体
// 4. sin_zero[8]纯粹多余的8个字节, 一般填充0, 可用 memset先给整个结构体填充0再复制

// sockaddr 结构体原型
struct sockaddr{
    sa_family_t  sin_family;   //地址族（Address Family）, 也就是地址类型
    char         sa_data[14];  //IP地址和端口号
};
// sockaddr 和 sockaddr_in 的长度相同, 都是16字节
// 只是将IP地址和端口号合并到一起, 用一个成员 sa_data 表示



// in_addr 结构体
// 在头文件 <netinet/in.h> 中定义了, typedef unsigned long in_addr_t
struct in_addr{
    // 因为是一个整数表示地址, 所以需要转换函数
    in_addr_t  s_addr;  //32位的IP地址
};
in_addr_t ip = inet_addr("127.0.0.1");

```

## 开始监听 listen

* 通过 listen() 函数可以让套接字进入被动监听状态
* 被动监听即: 没有客户端请求时, 套接字处于睡眠状态, 而收到客户端请求时会唤醒
* 客户端的请求会形成请求队列 (Request Queue)
  * 前一个客户端请求正在处理的时候, 新的请求会进入缓冲区
  * 缓冲区的长度就是在 `listen()` 中指定的, 通过 `backlog` 来指定长度


```cpp
int listen(int sock, int backlog);  //Linux
int listen(SOCKET sock, int backlog);  //Windows

// 1. sock 是需要进入监听的套接字
// 2. backlog 是请求队列的最大长度

```

