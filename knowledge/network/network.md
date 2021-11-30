# 1. Computer Network




## 1.1. 计算机网络结构


OSI 7层结构: 
1. 物理层: 电信号与数字信号的互相转换
2. 数据链接层: Data Link, 定义数据在中继设备的传输方式
3. 网络层: 意识到 地址的层, 明确区分发送端和接收端, 定义相关规格, 路径选择算法等
4. transport层: 进行数据检查的层, 确认包完整性, 请求重发等任务
5. session层 : 确认与目标设备的会话 session, 建立连接 connection
6. presentation 层: 定义网络数据与应用程序接收到的程序的转换关系, 数据压缩, 加密等
7. 应用层: 定义应用相关的协议 

TCP/IP 4层结构 : TCP/IP 不仅仅是两个协议, 而是建立在这两个协议的4层网络结构的整个协议群的总称
1. Network interface 层
2. Internet 层
3. Transport 层
4. Application 层

# 2. Network interface 层

Mac 地址:
* 48 bit
* 前 24 位是 制造厂家ID (OUI)
* 后 24 位是 给设备分配的ID

## 2.1. Ethernet LAN

目前, Ethernet 是 LAN 的主流方式, Ethernet 是 LAN 的标准化产物, 基于 IEEE802.3 系列

Ethernet 的规格: 10BASE-T
* 1 `10` 代表传送速度 m/s , 10 mbit/s
* 2 `BASE` 代表 base band 传送方式, 目前ethernet 不存在BASE以外的传输方式
* 3 `-T` 代表传输线材的种类
  * 数字代表同轴电缆, 数字是电缆的最大长度
  * 文字代表线缆的种类 T 双绞线 F 光纤
* 一些对照表
  * 10BASE2 同轴电缆5mm直径  802.3a
  * 10BASE5 同轴电缆12mm直径 802.3
  * 10BASE-T 双绞线 cat3     802.3i
  * 100BASE-T 双绞线 cat5   802.3u
  * 1000BASE-T 双绞线 cat6  802.3ab

## 2.2. Wireless LAN (WLAN)

无线 LAN 的规格是 802.11, 常见标准
* 802.11a   54mbit/s  5GHz
* 802.11b   11mbit/s  2.4GHz
* 802.11g   54mbit/s  2.4GHz
* 802.11n   150mbit/s 支持双倍带宽40MHz, z支持多输入多输出 MIMO
* 802.11ac  
* 802.11g



# 3. 网络层协议

## 3.1. IPv4

32bit, 分8bit 4个部分
* A 类地址 : 0~127,  host 地址为 24 bit
* B 类地址 : 128~191 , host 地址为 16 bit
* C 类地址 : 192~223 , host 地址为 8 bit
* D 类地址 : 224~239 , 广播地址
* E 类地址 : 240~255 , 实验用地址, 

子网掩码 : 
* 在原有分类网络的基础上, 将一部分 host 地址再扩张作为网络地址, 从而增加网络数
* 通过这种方法扩张的网络, 或者说将网络地址与 host地址分离的方法称作 CIDR (Classless Inter Domain Routing)

公网与私网:
* Global address , 地址不能重复, 通过 ICANN 机构来管理
* Private address
  * A 类私有地址 , 网络地址: 10.
  * B 类私有地址 , 网络地址: `172.0001****.`
  * C 类私有地址 , 网络地址: 192.168

## WAN (Wide Area Network)

外部网的服务提供, 根据原理区分方式
1. 专用线: 和据点搭建专线连接, 费用高, 连接稳定
2. 回线交换: 通过交换机和目标连接, 占用公共电话网络, 根据通信时间来确定费用
3. package交换: 数据以 package 为单位传输, 根据通信的数据量来决定费用
4. ATM : 数据以 53byte (cell)为单位传输, 初始是为了效率的传输声音等多媒体数据而开发的

几种 WAN 服务的落地方式
* ISDN : 最早的通过电话网的传输方式
* ADSL : Asymmetric Digital Subscriber Line 通过电话网络传输, 但是和电话通信使用不同的频段, 特点是非对称速度, 下行较快
* FTTH : Fiber To The Home, 光纤入户

WAN网络的接入设备:
* MOdulator-DEModulator : 调制解调器 (変復調装備)
  * 数字信号和电话网上的模拟信号进行互相转换用的
* DSU (Digital Service Unit)
  * 对计算机直出的数字信号进行变换, 使之适合长距离传输, ISDN 方式时会用到
* TA (Terminal Adapter)
  * ISDN线路和非ISDN终端(模拟信号电话等)进行连接时的信号转换机器

## 3.2. NAT 地址转换

* 将私有地址转换成共有地址的方法
* NAT (Network Address Translation) : 私有地址和共有地址 1:1 转换
* NAPT (Network Address Port Translation) : 借助 port, 多个私有地址host可以公用一个共有地址



## 3.3. ARP (Address Resolution Protocol)

IP地址 -> MAC地址




## 3.4. ICMP (Internet Control Message Protocol)

网络控制协议, 定义了多种报文

# 4. TCP/UDP 协议

传输层的两个协议

Well Know Port:
* 20    : FTP 数据传输
* 21    : FTP 控制
* 23    : TELNET
* 25    : SMTP
* 80    : HTTP
* 110   : POP3

# 5. Application Protocol 位于应用层的协议

TCP/IP 4层网络结构的应用层定义了许多协议


## 5.1. DNS (Domain Name System/Server)

域名解析服务: 域名 -> IP 地址

## 5.2. FTP (File Transfer Protocol)

## 5.3. HTTP (Hyper Text Transfer Protocol)

Web服务器向Web浏览器传输数据的协议


* Hyper Text Transfer Protocol 超文本传输协议  
* 基于 TCP/IP 通信协议
* 默认端口为 80

### 5.3.1. CGI (Common GateWay Interface)

CGI 是 HTTP 服务器与其他机器上的程序进行交互的工具, 必须运行在网络服务器上, 可用来
* 处理来自表单的输入信息
* 在服务器产生相应的处理
* 将相应的信息反馈给浏览器
使网页具有交互功能  

### 5.3.2. 工作原理

HTTP 协议工作于 客户端-服务端架构(C/S) 上, 浏览器作为 HTTP 客户端通过 URL 向 HTTP 服务端发送所有请求  
服务端 (Web服务器):  Apache 服务器, IIS(Internet Information Services)服务器等  

HTTP工作特点:
1. 无连接的, 服务端每次处理完一个请求后立即断开连接
2. 无状态的, 对事务处理没有记忆能力, 缺少状态意味着后续处理如果需要之前的信息, 则必须重传
3. 媒体独立的, 任何数据都可以通过 HTTP 发送, 只要协商好数据处理方法 (MIME Type)


* MIME 多用途 Internet 邮件扩展
* MIME Type: 是资源的媒体类型, 由 IETF 经过组织协商来指定的  
* 一般都是广泛应用的格式会得到 MIME TYPE  
* 指定的方法类似于 `Content-Type：text/HTML`  

### 5.3.3. 消息格式

客户端请求消息:
* 一个客户端传来的请求消息包括4个组成部分
  * 请求行 request line
  * 请求头部 header
  * 空行
  * 请求数据
* 请求行的组成
  * 请求方法
  * URL
  * 协议版本
* 请求头部, 包含多个头部行, 每一行包括
  * 头部字段名 : 值
* 空行: 由回车和换行组成, 用于分隔请求数据

示例: 一个请求行和三行请求头部
```
GET /hello.txt HTTP/1.1
User-Agent: curl/7.16.3 libcurl/7.16.3 OpenSSL/0.9.7l zlib/1.2.3
Host: www.example.com
Accept-Language: en, mi
```

服务器响应消息
* HTTP响应也由4个部分组成
  * 状态行
  * 消息报头
  * 空行
  * 响应正文 : 即一个超文本页面的源代码

示例
```
HTTP/1.1 200 OK
Date: Mon, 27 Jul 2009 12:28:53 GMT
Server: Apache
Last-Modified: Wed, 22 Jul 2009 19:15:56 GMT
ETag: "34aa387-d-1568eb00"
Accept-Ranges: bytes
Content-Length: 51
Vary: Accept-Encoding
Content-Type: text/plain
```
### 5.3.4. 请求类型

* HTTP 1.0 定义了三种请求方法： GET, POST 和 HEAD方法
* HTTP1.1 新增了六种请求方法：OPTIONS、PUT、PATCH、DELETE、TRACE 和 CONNECT 方法

| 请求名  | 功能                                                                        |
| ------- | --------------------------------------------------------------------------- |
| GET     | 向特定资源发出请求                                                          |
| HEAD    | 向服务器索要与 GET 请求相一致的响应, 但是响应体不会被返回, 即只获取响应头部 |
| POST    | 想指定资源提交数据请求处理 (表单, 文件), 数据被包含在请求体中               |
| OPTIONS | 返回服务器针对特定资源所支持的HTTP形式, 测试服务器的功能                    |
| PUT     | 向服务器指定资源位置上传最新内容                                            |
| PATCH   | 对PUT的补充, 对已知资源进行局部更新                                         |
| DELETE  | 请求服务器删除 Request-URI 所标识的资源                                     |
| TRACE   | 回显服务器收到的请求, 主要用于测试或诊断                                    |
| CONNECT | HTTP/1.1 协议中预留给将连接更改为管道方式的代理服务器                       |


### 5.3.5. 响应消息报头

HTTP 的请求和响应中都有对应的报头, 报头类似于 键值对, 提供了对信息的多种描述  


### 5.3.6. Content-Type

* Content-Type 也是 HTTP 响应里消息报头的一行, 用于告诉用于网络文件的类型或网页的编码  
* 这决定了浏览器怎么样读取这个文件

```
Content-Type: text/html; charset=utf-8
Content-Type: multipart/form-data; boundary=something
```
常见的媒体格式类型如下：
    text/html ： HTML格式
    text/plain ：纯文本格式
    text/xml ： XML格式
    image/gif ：gif图片格式
    image/jpeg ：jpg图片格式
    image/png：png图片格式



### 5.3.7. HTTP 状态码

HTTP Status Code, 即在服务器响应消息的状态行中, 对应响应信息的状态  

由三个十进制数字组成, 第一个数字决定了状态码的类型  
1. 信息, 服务器已收到请求, 需要请求者继续执行操作
2. 成功, 请求被接受并且成功处理
3. 重定向, 需要进一步的操作以完成请求
4. 客户端错误, 请求包含语法错误或者无法完成请求
5. 服务器错误, 服务器在处理过程中发生了错误


### 5.3.8. 400系列 - 客户端错误

| 代码 | 正式名                          | 意思                                                                                                                                                          |
| ---- | ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 400  | Bad Request                     | 客户端请求的语法错误，服务器无法理解                                                                                                                          |
| 401  | Unauthorized                    | 请求要求用户的身份认证                                                                                                                                        |
| 402  | Payment Required                | 保留，将来使用                                                                                                                                                |
| 403  | Forbidden                       | 服务器理解请求客户端的请求，但是拒绝执行此请求                                                                                                                |
| 404  | Not Found                       | 服务器无法根据客户端的请求找到资源（网页）。通过此代码，网站设计人员可设置对应的个性页面                                                                      |
| 405  | Method Not Allowed              | 客户端请求中的方法被禁止                                                                                                                                      |
| 406  | Not Acceptable                  | 服务器无法根据客户端请求的内容特性完成请求                                                                                                                    |
| 407  | Proxy Authentication Required   | 请求要求代理的身份认证，与401类似，但请求者应当使用代理进行授权                                                                                               |
| 408  | Request Time-out                | 服务器等待客户端发送的请求时间过长，超时                                                                                                                      |
| 409  | Conflict                        | 服务器完成客户端的 PUT 请求时可能返回此代码，服务器处理请求时发生了冲突                                                                                       |
| 410  | Gone                            | 客户端请求的资源已经不存在。410不同于404，如果资源以前有现在被永久删除了可使用410代码，网站设计人员可通过301代码指定资源的新位置                              |
| 411  | Length Required                 | 服务器无法处理客户端发送的不带Content-Length的请求信息                                                                                                        |
| 412  | Precondition Failed             | 客户端请求信息的先决条件错误                                                                                                                                  |
| 413  | Request Entity Too Large        | 由于请求的实体过大，服务器无法处理，因此拒绝请求。为防止客户端的连续请求，服务器可能会关闭连接。如果只是服务器暂时无法处理，则会包含一个Retry-After的响应信息 |
| 414  | Request-URI Too Large           | 请求的URI过长（URI通常为网址），服务器无法处理                                                                                                                |
| 415  | Unsupported Media Type          | 服务器无法处理请求附带的媒体格式                                                                                                                              |
| 416  | Requested range not satisfiable | 客户端请求的范围无效                                                                                                                                          |
| 417  | Expectation Failed              | 服务器无法满足Expect的请求头信息                                                                                                                              |

### 5.3.9. 500系列 - 服务端错误

| 代码 | 正式名                     | 意思                                                                                                |
| ---- | -------------------------- | --------------------------------------------------------------------------------------------------- |
| 500  | Internal Server Error      | 服务器内部错误，无法完成请求                                                                        |
| 501  | Not Implemented            | 服务器不支持请求的功能，无法完成请求                                                                |
| 502  | Bad Gateway                | 作为网关或者代理工作的服务器尝试执行请求时，从远程服务器接收到了一个无效的响应                      |
| 503  | Service Unavailable        | 由于超载或系统维护，服务器暂时的无法处理客户端的请求。延时的长度可包含在服务器的Retry-After头信息中 |
| 504  | Gateway Time-out           | 充当网关或代理的服务器，未及时从远端服务器获取请求                                                  |
| 505  | HTTP Version not supported | 服务器不支持请求的HTTP协议的版本，无法完成处理                                                      |

### 5.3.10. 300系列 - 重定向

| 代码 | 正式名             | 意思                                                                                                                                                             |
| ---- | ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 300  | Multiple Choices   | 多种选择。请求的资源可包括多个位置，相应可返回一个资源特征与地址的列表用于用户终端（例如：浏览器）选择                                                           |
| 301  | Moved Permanently  | 永久移动。请求的资源已被永久的移动到新URI，返回信息会包括新的URI，浏览器会自动定向到新URI。今后任何新的请求都应使用新的URI代替                                   |
| 302  | Found              | 临时移动。与301类似。但资源只是临时被移动。客户端应继续使用原有URI                                                                                               |
| 303  | See Other          | 查看其它地址。与301类似。使用GET和POST请求查看                                                                                                                   |
| 304  | Not Modified       | 未修改。所请求的资源未修改，服务器返回此状态码时，不会返回任何资源。客户端通常会缓存访问过的资源，通过提供一个头信息指出客户端希望只返回在指定日期之后修改的资源 |
| 305  | Use Proxy          | 使用代理。所请求的资源必须通过代理访问                                                                                                                           |
| 306  | Unused             | 已经被废弃的HTTP状态码                                                                                                                                           |
| 307  | Temporary Redirect | 临时重定向。与302类似。使用GET请求重定向                                                                                                                         |

### 5.3.11. 200系列 - 成功

| 代码 | 正式名                        | 意思                                                                                               |
| ---- | ----------------------------- | -------------------------------------------------------------------------------------------------- |
| 200  | OK                            | 请求成功。一般用于GET与POST请求                                                                    |
| 201  | Created                       | 已创建。成功请求并创建了新的资源                                                                   |
| 202  | Accepted                      | 已接受。已经接受请求，但未处理完成                                                                 |
| 203  | Non-Authoritative Information | 非授权信息。请求成功。但返回的meta信息不在原始的服务器，而是一个副本                               |
| 204  | No Content                    | 无内容。服务器成功处理，但未返回内容。在未更新网页的情况下，可确保浏览器继续显示当前文档           |
| 205  | Reset Content                 | 重置内容。服务器处理成功，用户终端（例如：浏览器）应重置文档视图。可通过此返回码清除浏览器的表单域 |
| 206  | Partial Content               | 部分内容。服务器成功处理了部分GET请求                                                              |

## 5.4. POP (Post Office Protocol)

从邮箱服务器把电子邮件抓取的协议

## 5.5. SMTP (Simple Mail Transfer Protocol)

电子邮件的传输协议, 由用户向 Mail 服务器发送, 以及 Mail 服务器之间的邮件传送等



## 5.6. SNMP (Simple Network Management Protocol)

网络管理协议

## 5.7. TELNET

## 5.8. DHCP (Dynamic Host Configuration Protocol)

动态IP配置协议
