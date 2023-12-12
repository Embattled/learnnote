# 1. Secure Shell  OpenSSH 

SSH 为 Secure Shell 的缩写，由 IETF 的网络小组（Network Working Group）所制定的一个协议  
SSH 为建立在应用层基础上的安全协议。SSH 是较可靠，专为远程登录会话和其他网络服务提供安全性的协议。  

1. SSH是安全的加密协议，用于远程连接Linux服务器               
2. SSH的默认端口是22，安全协议版本是SSH2               
3. SSH服务器端主要包含2个服务功能SSH连接和SFTP服务器               


OpenSSH 是由 OpenBSD 最早开发的 ssh 应用实现 现在已经并入 Linux 内核  
* 安装client和server
`$ sudo apt install openssh-client`  
`$ sudo apt install openssh-server`  

The OpenSSH suite consists of the following tools:
*  Remote operations are done using `ssh`, `scp`, and `sftp`.
*  Key management with `ssh-add`, `ssh-keysign`, `ssh-keyscan`, and `ssh-keygen`.
*  The service side consists of `sshd`, `sftp-server`, and `ssh-agent`. 


ssh服务端由2部分组成： openssh(提供ssh服务)    openssl(提供加密的程序)
SSH(远程连接工具)连irtsve接原理：ssh服务是一个守护进程 (deamon) ，系统后台监听客户端的连接   
ssh服务端的进程名为sshd,负责实时监听客户端的请求(IP 22端口)，包括公共秘钥等交换等信息   
* 查看server是否启动以及启动
`$ ps -e | grep sshd`  
`$ service ssh start`  
ssh的客户端可以用 XSHELL，Securecrt, Mobaxterm等工具进行连接  

# 2. ssh-keygen

用于生成和管理本机的 ssh 密钥

## 2.1. 公钥生成

* 通过运行 `ssh-keygen` 可以生成本机的密钥
* 将公钥储存在远程主机上,直接允许登录shell，不再要求密码。

```shell
$ ssh-keygen  

# 运行后，在$HOME/.ssh/目录下，会新生成两个文件：id_rsa.pub和id_rsa。前者是你的公钥，后者是你的私钥。
# 将公钥发送给远程主机的根目录下
$ scp .ssh/idrsa.pub user_name@192.168.xxx.xxx:/home/user_name/
# 把拷贝过来的 id_rsa.pub 中的密匙写入 authorized_keys, 并给与其 600 权限
$ cat id_rsa.pub >> .ssh/authorized_keys
$ sudo chmod 600 .ssh/authorized_keys
```

通过 `-t` 可以指定密钥的算法(类型) , 默认的是 `rsa-sha2-512`


## 2.2. 管理命令

* 删除host的密钥缓存
  * `-R hostname | [hostname]:port`
  * 从本机的 `known_hosts` 文件中删除相关 host 的 key 信息



# 3. ssh 

积累的常用命令笔记  
```shell

# Verbose mode. Causes ssh to print debugging messages about its progress
# 用于对连接进行 debug 可以增加 v 的个数来提高输出的精细化程度  最高到 -vvv
-v 
-vvv

# Requests that standard input and output on the client be forwarded to _host_ on _port_ over the secure channel.
# 用于写在 ProxyCommand 中进行跳板  
-W host:port
```

大部分的 CLI 参数都可以直接写在 config 中, 具体能不能写则会在文档中具体给出

## 3.1. ssh 端口转发

端口转发的配置都可以直接定义在 Config 中

简要说明: -R 和 -L 是一个相反意思但是功能有重叠的命令, 二者之间的抉择主要基于安全考量
* -R  : 将远程的端口转发到本地  (Local Port Forwarding)
  * connections to the given TCP port or Unix socket on the remote (server) host are to be forwarded to the local side. 
  * 在远端应用已经运行了一个监听在(远端)本地端口 8080的时候, 将该服务暴露给远端
  * 会锁定远端的端口
  * 可以理解为把远端的活儿揽到自己身上

* -L  : 将本地的端口转发到远程  (Remote Port Forwarding)
  * connections to the given TCP port or Unix socket on the local (client) host are to be forwarded to the given host and port, or Unix socket, on the remote side
  * 会把本地的端口锁定
  * 可以把自己活推给远端
* 依据安全要求, 选择 -R 或者 -L
* bind_address 在不指明的情况下默认等同于 `local_host`


相关配合命令:
* `-N` : 不在远端执行 command, 这对于搭建端口转发的时候很有用.  

```sh

-L [bind_address:]port:host:hostport
ssh -L 80:localhost:80 SUPERSERVER
# 将连接到本地 localhost 80 端口的访问转发到 目标(这里是 SUPERSERVER) 的 80 端口上
# 这意味这任何访问 (包括本地作为服务器由第三方用户访问的时候) 本地计算机80端口 (即浏览器) 的时候, 会得到 SUPERSERVER 的80端口的服务回应
# 此时 localhost 不运行任何 webserver

-L [bind_address:]port:remote_socket
-L local_socket:host:hostport
-L local_socket:remote_socket

# 发往本机的80端口访问转发到192.168.1.1的8080端口
ssh -C -f -N -g -L 80:192.168.1.1:8080 user@192.168.1.1


-R [bind_address:]port:host:hostport
ssh -R 80:localhost:80 tinyserver
# 所有访问 tinyserver 80端口的连接都会转发到 本机 localhost 的 80 端口
# 此时可以理解为 tinyserver 是一个性能弱小的 server, 而 localhost 是一个性能强劲的服务器
# tinyserver 不运行 webserver, 本地运行, 但对于访问 tinyserver 的第三方用户感知到的是 tinyserver 是 webserver

# 更复杂的应用场景, 假设本地 localhost 是性能强力的服务器, 且在不同的端口同时运行着多个 webserver
# 由不同的 tinyserver 作为服务接入端
ssh -R 80:localhost:30180 tinyserver1
ssh -R 80:localhost:30280 tinyserver2
# 更甚至于, 使用 ssh 来把本地变成一个中继站, 用于连接 SUPERSERVER 和 tinyserver, 注意, 此时可能需要添加 -g 命令
ssh -R 80:SUPERSERVER:30180 tinyserver1
ssh -R 80:SUPERSERVER:30280 tinyserver2

-R [bind_address:]port:local_socket
-R remote_socket:host:hostport
-R remote_socket:local_socket
-R [bind_address:]port

# 把发往192.168.1.1的8080访问转发到本机的80端口
ssh -C -f -N -g -R 80:192.168.1.1:8080 user@192.168.1.1
```

* `-D` : 动态 application-level port 转发
  * 主要用于搭建 SOCKS4 服务器
  * Currently the SOCKS4 and SOCKS5 protocols are supported




## 3.2. 其他命令

辅佐命令 
* `-g`    : 启动全局转发
  * Allows remote hosts to connect to local forwarded ports.
  * If used on a multiplexed connection, then this option `must` be specified on the master process. 

* `-C` : 要求对所有传输的数据进行压缩化传输, 这回提高在低配置网络下的访问速度, 但会减慢在高配网络下的速度


* `-f` : 

## 3.3. 连接配置


* `-n`  : Redirects stdin from /dev/null (actually, prevents reading from stdin)
  * 将输入流重定位到 /dev/null
  * 该命令可以用于测试 ssh 连通性  `ssh -n -o ConnectTimeout=10 target` 即用很短的时间测试一个服务器的连通性
  * 该命令的主要用途是在远端启动一个 x11 window 程序 `ssh -n <如果配置没有开启x11 forward 的话需要加-X> target xclock &`, 结尾的 & 是让一整个程序从一开始就运行在后台

* `-o option` : 在 CLI 中指定用于覆盖 ssh config 的各种详细配置, 支持的详细内容可以直接参照 ssh_config





# 4. ssh_config

ssh_config — OpenSSH client configuration file

ssh 命令的执行会遵从如下顺序来获取相应的信息
1. CLI 参数
2. user's configuration file `~/.ssh/config`
3. system-wide configuration file `/etc/ssh/ssh_config`

config 文件包括关键字参数对, 每行一个, `#` 开头的行作为注释    
参数可以用 `"` 括起来, 用以表示包含空格的参数, 配置选项可以用空格或者可选空格和一个 `=` 分割





通过配置文件保存密码

```shell
#配置前
ssh username@hostname -p port
#然后输入密码

#配置以后，我们只需要输入连接账户的别名即可
ssh 别名


# 配置方法 在.ssh/config中配置，如果没有config，创建一个即可
# 这里是用户的ssh配置  
Host 别名
    Hostname 主机名
    Port 端口
    User 用户名
```

### 4.0.1. 防止自动断开

用ssh链接服务端，一段时间不操作或屏幕没输出（比如复制文件）的时候，会自动断开  

两种方法, 配置客户端或者配置服务端

```sh
# 客户端配置方法
# 在系统的ssh配置文件中添加设置
vi  /etc/ssh/ssh_config
Host *
    ServerAliveInterval 30
# 或者在ssh连接中加入选项
ssh -o ServerAliveInterval=30 hostname

# 服务端配置方法
vi /etc/ssh/sshd_config
ClientAliveInterval 60
ClientAliveCountMax 1
```

### 4.0.2. ProxyCommand 跳板

很多环境都有一台统一登录跳板机,我们需要先登录跳板机,然后再登录自己的目标机器.  
ProxyCommand是openssh的特性,如果使用putty,xshell,那么是没有这个功能的  

在windows下面,推荐使用`mobaxterm`,其基于cgywin,里面移植了一个完整版本的openssh实现  
在linux和macos里面直接就是openssh了  


`ProxyCommand ssh -q -x -W %h:%p tiaoban`  
* %h:%p : 表示要连接的目标机端口,可以直接写死固定值,但是使用%h和%p可以保证在Hostname和Port变化的情况下ProxyCommand这行不用跟着变化.


### 4.0.3. SHA1 支持

新版Openssh中认为SHA1这种hash散列算法过于薄弱，已经不再支持，所以我们需要手动去enable对于SHA1的支持

```shell
# 在全局 ssh 配置中
$ vim /etc/ssh/ssh_config

#取消注释这一行内容：   MACs hmac-md5,hmac-sha1,umac-64@openssh.com,hmac-ripemd160
#并且在文件结尾添加：
HostkeyAlgorithms ssh-dss,ssh-rsa
KexAlgorithms +diffie-hellman-group1-sha1

```


# 5. SCP(secure copy )远程拷贝文件与文件夹

scp 是 linux 系统下基于 ssh 登陆进行安全的远程文件拷贝命令。  
scp 是加密的，rcp 是不加密的，scp 是 rcp 的加强版。  

openssh 8.0 将 scp 标记为过时的不建议使用的, 推荐用 sftp 或者 rsync 来代替 scp  

`scp [可选参数] file_source file_target `  

| 参数说明             |
| -------------------- | ----------------------------------------------------------------------------------------------- |
| -1                   | 强制scp命令使用协议ssh1                                                                         |
| -2                   | 强制scp命令使用协议ssh2                                                                         |
| -4                   | 强制scp命令只使用IPv4寻址                                                                       |
| -6                   | 强制scp命令只使用IPv6寻址                                                                       |
| -B                   | 使用批处理模式（传输过程中不询问传输口令或短语）                                                |
| -C                   | 允许压缩。（将-C标志传递给ssh，从而打开压缩功能）                                               |
| -p                   | 保留原文件的修改时间，访问时间和访问权限。                                                      |
| -q                   | 不显示传输进度条。                                                                              |
| -r                   | 递归复制整个目录。                                                                              |
| -v                   | 详细方式显示输出。scp和ssh(1)会显示出整个过程的调试信息。这些信息用于调试连接，验证和配置问题。 |
| -c cipher            | 以cipher将数据传输进行加密，这个选项将直接传递给ssh。                                           |
| -F ssh_config        | 指定一个替代的ssh配置文件，此参数直接传递给ssh。                                                |
| -i identity_file     | 从指定文件中读取传输时使用的密钥文件，此参数直接传递给ssh。                                     |
| -l limit             | 限定用户所能使用的带宽，以Kbit/s为单位。                                                        |
| -o ssh_option        | 如果习惯于使用ssh_config(5)中的参数传递方式，                                                   |
| -P port注意是大写的P | port是指定数据传输用到的端口号                                                                  |
| -S program           | 指定加密传输时所使用的程序。此程序必须能够理解ssh(1)的选项。                                    |


如果远程服务器防火墙有为scp命令设置了指定的端口，我们需要使用 -P 参数来设置命令的端口号，命令格式如下：
```shell
#scp 命令使用端口号 4588

scp -P 4588 remote@www.runoob.com:/usr/local/sin.sh /home/administrator
```

**实例**

从本地复制文件到远程
```shell
#  指定了用户名，命令执行后需要再输入密码 仅指定了远程的目录 文件名字不变
scp local_file remote_username@remote_ip:remote_folder 

#  指定了用户名和文件名
scp local_file remote_username@remote_ip:remote_file 

#  没有指定用户名，命令执行后需要输入用户名和密码 
scp local_file remote_ip:remote_folder 
scp local_file remote_ip:remote_file

```

复制本地目录到远端目录
```
scp -r local_folder remote_username@remote_ip:remote_folder 
scp -r local_folder remote_ip:remote_folder 
```

远程复制到本地
```
scp root@www.runoob.com:/home/root/others/music /home/space/music/1.mp3 
scp -r www.runoob.com:/home/root/others/ /home/space/music/
```


# 6. 