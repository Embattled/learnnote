# 1. Netplan 设置网络

新出的Ubuntu服务器18.04版本修改了IP地址配置程序, Ubuntu和Debian的软件架构师删除了以前的ifup/ifdown命令和/etc/network/interfaces配置文件, 改为使用/etc/netplan/01-netcfg.yaml和sudo netplay apply命令管理IP地址.

完整教程:  
https://netplan.io/examples


## 1.1.  Netplan configuration files

目录
/etc/netplan  下的  
The configuration file might have a name such as` 01-network-manager-all.yaml `or `50-cloud-init.yaml`.  

通过目录可以对网卡进行初始化的配置  

## 1.2. 启用以及开启DHCP

```shell
network:
    #有线网的配置
    ethernets:
        #网卡名称
        eth0:
            dhcp4: true
            optional: true
    version: 2

    #写入新一字段开启无线网,命名为 wifis
    wifis:
        #网卡名
        wlan0:
            #同样的两个字段
            dhcp4: true
            optional: true
            #记入接入点信息
            access-points:
                # 这一行内容为 '接入点名:'
                'SSID-NAME':
                    password:'passwd'

```
写入完成后应用更改  
```
$ sudo netplan apply
$ sudo netplan --debug apply
```