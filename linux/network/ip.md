# Linux 的网络管理  

linux的ip命令和ifconfig类似，但前者功能更强大，并旨在取代后者。使用ip命令，只需一个命令，你就能很轻松地执行一些网络管理任务。ifconfig是net-tools中已被废弃使用的一个命令，许多年前就已经没有维护了  
# iproute2  简介
作为网络配置工具的一份子，iproute2的出现旨在从功能上取代net-tools。net-tools通过`procfs(/proc)和ioctl`系统调用去访问和改变内核网络配置，而iproute2则通过`netlink`套接字接口与内核通讯。  
抛开性能而言，iproute2的用户接口比net-tools显得更加直观。比如，各种网络资源（如link、IP地址、路由和隧道等）均使用合适的对象抽象去定义，使得用户可使用一致的语法去管理不同的对象。更重要的是，到目前为止，iproute2仍处在持续开发中。

## 1. 基础命令以及与net-tools的对照

| 功能                                             | iproute2命令                                                                   | net-tools命令                                                                    |
| ------------------------------------------------ | ------------------------------------------------------------------------------ | -------------------------------------------------------------------------------- |
| 显示出所有可用网络接口的列表（无论接口是否激活） | ip link show                                                                   | ifconfig -a                                                                      |
| 激活或停用某个指定的网络接口                     | ip link set up/down eth1                                                       | ifconfig eth1 up/down                                                            |
| 配置网络接口的IPv4地址                           | ip addr add 10.0.0.1/24 dev eth1    可以使用iproute2给同一个接口分配多个IP地址 | ifconfig eth1 10.0.0.1/24                                                        |
| 移除网络接口的IPv4地址                           | ip addr del 10.0.0.1/24 dev eth1                                               | 除了给接口分配全0地址外，net-tools没有提供任何合适的方法来移除网络接口的IPv4地址 |
| 查看某一接口的地址                               | ip addr show dev eth1  快速查看ip的命令为`$ ip a`                              | ifconfig eth1                                                                    |

## 路由相关设定

修改网卡Mac地址,修改前需要先停用  
`sudo ip link set dev eth1 address 08:00:27:75:2a:67`  


查看路由表  
`ip route show `  

添加修改移除默认路由  
```shell
 $ sudo ip route add default via 192.168.1.2 dev eth0
$ sudo ip route replace default via 192.168.1.2 dev eth0
$ sudo ip route del 172.16.32.0/24 
```

查看套接字统计信息  
`$ ss`  


# net-tools 简介

net-tools起源于BSD的TCP/IP工具箱，后来成为老版本Linux内核中配置网络功能的工具。但自2001年起，Linux社区已经对其停止维护。同时，一些Linux发行版比如Arch Linux和CentOS/RHEL 7则已经完全抛弃了net-tools，只支持iproute2。