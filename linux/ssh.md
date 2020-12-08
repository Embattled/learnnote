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
SSH(远程连接工具)连接原理：ssh服务是一个守护进程 (deamon) ，系统后台监听客户端的连接   
ssh服务端的进程名为sshd,负责实时监听客户端的请求(IP 22端口)，包括公共秘钥等交换等信息   
* 查看server是否启动以及启动
`$ ps -e | grep sshd`  
`$ service ssh start`  
ssh的客户端可以用 XSHELL，Securecrt, Mobaxterm等工具进行连接  

## 1.1. ssh 

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
## 1.2. ssh_config

通过配置文件保存密码

```shell
#配置前
ssh username@hostname -p port
#然后输入密码

#配置以后，我们只需要输入连接账户的别名即可
ssh 别名


# 配置方法 在.ssh/config中配置，如果没有config，创建一个即可  
Host 别名
    Hostname 主机名
    Port 端口
    User 用户名
```


### 1.2.1. ProxyCommand 跳板

很多环境都有一台统一登录跳板机,我们需要先登录跳板机,然后再登录自己的目标机器.  
ProxyCommand是openssh的特性,如果使用putty,xshell,那么是没有这个功能的  

在windows下面,推荐使用`mobaxterm`,其基于cgywin,里面移植了一个完整版本的openssh实现  
在linux和macos里面直接就是openssh了  


`ProxyCommand ssh -q -x -W %h:%p tiaoban`  
* %h:%p : 表示要连接的目标机端口,可以直接写死固定值,但是使用%h和%p可以保证在Hostname和Port变化的情况下ProxyCommand这行不用跟着变化.


### 1.2.2. SHA1 支持

新版Openssh中认为SHA1这种hash散列算法过于薄弱，已经不再支持，所以我们需要手动去enable对于SHA1的支持

```shell
# 在全局 ssh 配置中
$ vim /etc/ssh/ssh_config

#取消注释这一行内容：   MACs hmac-md5,hmac-sha1,umac-64@openssh.com,hmac-ripemd160
#并且在文件结尾添加：
HostkeyAlgorithms ssh-dss,ssh-rsa
KexAlgorithms +diffie-hellman-group1-sha1

```


## 1.3. SCP(secure copy )远程拷贝文件与文件夹

scp 是 linux 系统下基于 ssh 登陆进行安全的远程文件拷贝命令。  
scp 是加密的，rcp 是不加密的，scp 是 rcp 的加强版。  

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


# 2. 应用示例
## 2.1. 登录及配置文件

### 2.1.1. 公钥免密登录
用户将自己的公钥储存在远程主机上,直接允许登录shell，不再要求密码。
```shell
#用户必须提供公钥，如果没有公钥，可以生成一个  
$ ssh-keygen  

# 运行后，在$HOME/.ssh/目录下，会新生成两个文件：id_rsa.pub和id_rsa。前者是你的公钥，后者是你的私钥。
# 将公钥发送给远程主机的根目录下
$ scp .ssh/idrsa.pub user_name@192.168.xxx.xxx:/home/user_name/
# 把拷贝过来的 id_rsa.pub 中的密匙写入 authorized_keys，并给与其 600 权限
$ cat id_rsa.pub >> .ssh/authorized_keys
$ sudo chmod 600 .ssh/authorized_keys
```
