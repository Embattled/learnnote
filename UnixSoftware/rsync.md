# Remote Sync  (rsync)

通过网络快速同步多台主机之间的文件和目录, 名利用差分算法减少数据传输量  

rsync 有两种认证方式, 一种是 rsync-daemon, 另一种是 ssh  
* rsync-daemon 方式是主要的认证方式, 会默认监听 TCP 的 `873` 端口, 需要客户端服务器都安装 rsync   
* rsync-daemon 方式需要启动守护进程, 因此可能需要特殊权限  

rsync 的软件本体不区分客户端和服务器端  

优点:
* 完整的目录树和文件系统同步
* 容易保证原来文件的权限, 时间, 软硬连接等
* 无需特殊权限
* 快速, 压缩传输
* 使用 scp ssh 等安全传输
* 匿名传输, 用于网站镜像
* openssh 官方推荐的 scp 的替代品


# rsync CLI

```sh
Local:
    rsync [OPTION...] SRC... [DEST]

Access via remote shell:
    Pull:
        rsync [OPTION...] [USER@]HOST:SRC... [DEST]
    Push:
        rsync [OPTION...] SRC... [USER@]HOST:DEST

Access via rsync daemon:
    Pull:
        rsync [OPTION...] [USER@]HOST::SRC... [DEST]
        rsync [OPTION...] rsync://[USER@]HOST[:PORT]/SRC... [DEST]
    Push:
        rsync [OPTION...] SRC... [USER@]HOST::DEST
        rsync [OPTION...] SRC... rsync://[USER@]HOST[:PORT]/DEST)
```

## OPTIONS

### common 

* `--help` 
* `-V --version`
* `-v --verbose` : 提高信息详细程度, 默认情况下, rsync 会以最安静的方式执行  



### general

* `-a --archive`  : 递归同步, 并且保留几乎所有内容.
  * 相当于 `-rlptgoD` (WTF?) 
  * 要记住的仅有 `-a` 所不包含的内容:
    * `-A`  : ACLs
    * `-X`  : xattrs
    * `-U`  : atimes
    * `-N`  : crtimes
    * `-H`  : finding and preserving of hardlinks


* `-z --compress` : 传输的时候压缩数据,


# Blogger 笔记

服务端配置

服务端需要启动守护进程


# rsyncd.conf -⁠ configuration file for rsync in daemon mode

The rsyncd.conf file is the runtime configuration file for rsync when run as an rsync daemon.
The rsyncd.conf file controls authentication, access, logging and available modules.

## FILE FORMAT

## LAUNCHING THE RSYNC DAEMON

## GLOBAL PARAMETERS

## MODULE PARAMETERS

## AUTHENTICATION STRENGTH

