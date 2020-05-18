# Linux 配置DHCP客户端  
最基础的配置dhcp客户端  

编辑文件/etc/network/interfaces:sudo vi /etc/network/interfaces  
```
auto [eth0]
iface [eth0] inet dhcp
```
# DHCP CLIENT
dhclient是一个DHCP协议客户端，它使用DHCP协议或者BOOTP协议或在这两个协议都不可用时使用静态地址来配置一个或多个网络接口  


# dhcpcd
