# Linux下的路由软件
# isc-dhcp-server

## 1.简介  

如有重启错误，先确保hostapd已启动

```
 /etc/default/isc-dhcp-server
INTERFACES="wlan0"

/etc/dhcp/dhcpd.conf

subnet 192.168.10.0 netmask 255.255.255.0 {
 range 192.168.10.10 192.168.10.20;
 option broadcast-address 192.168.10.255;
 option routers 192.168.10.1;
 default-lease-time 600;
 max-lease-time 7200;
 option domain-name "local-network";
 option domain-name-servers 8.8.8.8, 8.8.4.4;
}
```
# dnsmasq

