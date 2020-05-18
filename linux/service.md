# Linux 的服务管理

Linux 服务管理两种方式service和systemctl 

systemd是Linux系统**最新的初始化系统**(init),作用是提高系统的启动速度，尽可能启动较少的进程，尽可能更多进程并发启动。

systemd对应的进程管理命令就是 `systemctl`

# service  

service命令其实是去`/etc/init.d`目录下，去执行相关程序, 已经被淘汰

```shell
# service命令启动redis脚本
service redis start
# 直接启动redis脚本
/etc/init.d/redis start
# 开机自启动
update-rc.d redis defaults
```

# 1. systemctl

## 1. 概念

systemctl命令兼容了service  

即systemctl也会去`/etc/init.d`目录下，查看，执行相关程序  

```shell
# 开机自启动
systemctl enable redis
```

通过 `Unit` 作为单位管理进程

`/usr/lib/systemd/system(Centos)`  
`/etc/systemd/system(Ubuntu)`  

systemd 默认读取 `/etc/systemd/system `下的配置文件，该目录下的文件会链接/lib/systemd/system/下的文件。执行 ls /lib/systemd/system 你可以看到有很多启动脚本，其中就有最初用来定义开机脚本的 `rc.local.service`

主要有四种类型文件.mount,.service,.target,.wants  
代表四种`Unit`  

## 2. Unit操作命令

`systemctl –-version`  查看版本  

`systemctl [command] [unit]` 命令格式  

### a.command综述

| 命令      | 功能简述                                                                   |
| --------- | -------------------------------------------------------------------------- |
| start     | 立刻启动后面接的 unit。                                                    |
| stop      | 立刻关闭后面接的 unit。                                                    |
| restart   | 立刻关闭后启动后面接的 unit，亦即执行 stop 再 start 的意思。               |
| reload    | 不关闭 unit 的情况下，重新载入配置文件，让设置生效。                       |
| enable    | 设置下次开机时，后面接的 unit 会被启动。                                   |
| disable   | 设置下次开机时，后面接的 unit 不会被启动。                                 |
| show      | 列出 unit 的配置。                                                         |
| status    | 目前后面接的这个 unit 的状态，会列出有没有正在执行、开机时是否启动等信息。 |
| is-active | 目前有没有正在运行中。                                                     |
| is-enable | 开机时有没有默认要启用这个 unit。                                          |
| kill      | 不要被 kill 这个名字吓着了，它其实是向运行 unit 的进程发送信号。           |
| mask      | 注销 unit，注销后你就无法启动这个 unit 了。                                |
| unmask    | 取消对 unit 的注销。                                                       |

### b.status 

* 第一行是对 unit 的基本描述。  
* 第二行中的 Loaded 描述操作系统启动时会不会启动这个服务，enabled 表示开机时启动，disabled 表示开机时不启动。  
  * 关于 unit 的启动状态，除了 enable 和 disable 之外还有:  
  *   static:这个 unit 不可以自己启动，不过可能会被其它的 enabled 的服务来唤醒。
  * mask:这个 unit 无论如何都无法被启动！因为已经被强制注销。可通过 systemctl unmask 改回原来的状态。
* 第三行 中的 Active 描述服务当前的状态，active (running) 表示服务正在运行中。如果是 inactive (dead) 则表示服务当前没有运行。 
  * active (exited)：仅执行一次就正常结束的服务，目前并没有任何程序在系统中执行。
  * active (waiting)：正在执行当中，不过还再等待其他的事件才能继续处理。 
* 第四行的 Docs 提供了在线文档的地址。  


## 3.综合查看命令

systemctl 提供了子命令可以查看系统上的 unit，命令格式为:   
`systemctl [command] [--type=TYPE] [--all]` 

不带任何参数执行 systemctl 命令会列出所有已启动的 unit  
如果添加 -all 选项会同时列出没有启动的 unit。    

**command**  
不带任何参数执行 systemctl 命令会列出所有已启动的 unit  
systemctl list-units: 列出当前已经启动的 unit  
systemctl list-unit-files: 根据 /lib/systemd/system/ 目录内的文件列出所有的 unit, 即列出所有以安装的服务    

systemd-cgls   以树形列出正在运行的进程，它可以递归显示控制组内容

**--type=TYPE**  

可以过滤某个类型的 unit。  
` systemctl list-units --type=service `  

### 操作环境管理
通过指定 `--type=target` 就可以用 `systemctl list-units` 命令查看系统中默认有多少种 target

| 操作环境              | 功能                                                                                                                                   |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| graphical.target  | 就是文字界面再加上图形界面，这个 target 已经包含了下面的 multi-user.target。                                                           |
| multi-user.target | 纯文本模式！                                                                                                                           |
| rescue.target     | 在无法使用 root 登陆的情况下，systemd 在开机时会多加一个额外的临时系统，与你原本的系统无关。这时你可以取得 root 的权限来维护你的系统。 |
| emergency.target  | 紧急处理系统的错误，在无法使用 rescue.target 时，可以尝试使用这种模式！                                                                |
| shutdown.target   | 就是执行关机。                                                                                                                         |
| getty.target      | 可以设置 tty 的配置。                                                                                                                  |

正常的模式是` multi-user.target `和 `graphical.target `两个，救援方面的模式主要是 `rescue.target` 以及更严重的 `emergency.target`。如果要修改可提供登陆的 tty 数量，则修改 getty.target。


命令|功能
-|-
get-default|取得目前的 target。
set-default|设置后面接的 target 成为默认的操作模式。
isolate|切换到后面接的模式。

我们还可以在不重新启动的情况下切换不同的 target，比如从图形界面切换到纯文本的模式：
`systemctl isolate multi-user.target`

# 2. Unit的编写与设置

## 1. Unit的基本概念
一般都会有  
Unit小节: 描述,启动时间与条件等等  

### .service

.service文件定义了一个服务，分为[Unit]，[Service]，[Install]三个小节

`[Unit]` 段: 启动顺序与依赖关系   
`[Service] `段: 启动行为,如何启动，启动类型  
`[Install]` 段: 定义如何安装这个配置文件，即怎样做到开机启动  
### .mounnt

.mount文件定义了一个挂载点，[Mount]节点里配置了What(名称),Where(位置),Type(类型)三个数据项

```
What=hugetlbfs
Where=/dev/hugepages
Type=hugetlbfs
```
等于执行以下命令  
`mount -t hugetlbfs /dev/hugepages hugetlbfs`  

### .target

.target定义了一些基础的组件，供.service文件调用

### .wants文件

`.wants`文件定义了要执行的文件集合，每次执行，`.wants`文件夹里面的文件都会执行  

## 2.编写开机启动rc.local

查看`/lib/systemd/system/rc.local.server`,默认会缺少`Install`段,显然这样配置是无效的  

```shell
# rc.local.server的Unit段, 可以看到会执行 /etc/rc.local
[Unit]
Description=/etc/rc.local Compatibility
Documentation=man:systemd-rc-local-generator(8)
ConditionFileIsExecutable=/etc/rc.local
After=network.target


#修改文件进行配置  
[Install]  
WantedBy=multi-user.target  


#之后再在/etc/目录下面创建rc.local文件，赋予执行权限

$ touch /etc/rc.local
$ chmod +x /etc/rc.local

# 在rc.local里面写入
# 注意：'#!/bin/sh' 这一行一定要加

#!/bin/sh
exit 0

# 最后将/lib/systemd/system/rc.local.service 链接到/etc/systemd/system目录

$ ln -s /lib/systemd/system/rc.local.service /etc/systemd/system/
```


