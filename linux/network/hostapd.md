# 热点软件hostapd

开启系统路由转发在这个文件里  
/etc/sysctl.conf

Hostapd only creates wireless Ethernet switches, it does not know about the IP protocol or routing.   

使用 iw 来查看无线网卡的支持技术,必须要有 `AP`  
`iw list | grep "Supported interface modes" -A 8`  


* What it can do

    Create an AP;
    Create multiple APs on the same card (if the card supports it, usually up to 8);
    Create one AP on one card and another AP on a second card, all within a single instance of Hostapd;
    Use 2.4GHz and 5GHz at the same time on the same card. This requires a card with two radios though, which is pretty rare (but hostapd supports it) - if the card creates two wlanX interfaces, you might be lucky;

* What it cannot do

    Create multiple APs on different channels on the same card. Multiple APs on the same card will share the same channel;  
    Create a dual-band AP, even with two cards. But it can create two APs with the same SSID;  
    Assign IPs to the devices connecting to the AP, a dhcp server is needed for that;  
    Assign an IP to the AP itself, it is not hostapd's job to do that;



# 从零开始配置一个热点

##　1. Common Options : options that you will probably want to set  

```shell
#选择要使用的无线网卡  
interface=wlan0  

#如果路由系统需要桥接,在这里指定桥接接口  
# Set to a bridge if the wireless interface in use is part of a network bridge interface. 
bridge=br0

# 设置驱动程序
# For our purposes, always nl80211
driver = nl80211

# 设置网卡工作模式为 802.11G
# "g" simply means 2.4GHz band
# epend on the hardware, but are always a subset of a, b, g.
hw_mode = g / n /ac /

# 设置网络名称
ssid=test

# 设置信道
channel = 6



#802.11n设置
# If your hardware doesn't support 802.11n, or you don't plan on using it, you can ignore these.
ieee80211n=1 # enable 802.11n 
# A list of the 802.11n features supported by your device. 
ht_capab=[HT40+][SHORT-GI-40][DSSS_CCK-40]



# ----加密模式设置------------
# 设置密码为123456789
wpa_passphrase = 123456789


#  是否开启MAC地址过滤
macaddr_acl=0

# 这是一个比特符号位
# This is a bit field where the first bit (1) is for open auth, the second bit (2) is for Shared key auth (WEP) and  (3) is both. 
auth_algs=1/3

# 设置为1的话将会不广播ssid
ignore_broadcast_ssid=0


# wpa: This is a bit field like auth_alg
# The first bit enables WPA1 (1), the second bit enables WPA2 (2), and both enables both (3) 
wpa =1/  2 / 3

# This controls what key management algorithms a client can authenticate with. 
wpa_key_mgmt = WPA-PSK

# 设置加密方式为CCMP
# wpa_pairwise: This controls WPA's data encryption. 
wpa_pairwise = CCMP / TKIP

# rsn_pairwise: This controls WPA2's data encryption
# 看不懂
rsn_pairwise = CCMP

# A good starting point for a WPA and WPA2 enabled access point is
macaddr_acl=0
auth_algs=1
ignore_broadcast_ssid=0
wpa=3
wpa_passphrase=YourPassPhrase
wpa_key_mgmt=WPA-PSK
wpa_pairwise=TKIP
rsn_pairwise=CCMP

# If, alternatively, you just want to support WPA2, you could use something like: 
macaddr_acl=0
auth_algs=1
ignore_broadcast_ssid=0
wpa=2
wpa_passphrase=YourPassPhrase
wpa_key_mgmt=WPA-PSK
wpa_pairwise=TKIP
rsn_pairwise=CCMP



beacon_int = 100

wmm_enabled = 1
wmm_enabled=1

```


## 