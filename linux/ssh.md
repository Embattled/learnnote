# SSH的基本操作

## 1. 安装与登录  

```shell
# 查看是否安装以及版本 ubuntu上默认只安装了openssh-client
ssh -V

# 安装client和server
$ sudo apt-get install openssh-client
$ sudo apt-get install openssh-server

# 查看server是否启动以及启动
$ ps -e | grep sshd
$ service ssh start

```

**公钥登录**  
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



## SCP(secure copy )远程拷贝文件与文件夹

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
| -P port注意是大写的P | port是指定数据传输用到的端口号                                                                |
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
