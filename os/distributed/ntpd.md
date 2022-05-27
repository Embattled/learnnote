# Network Time Protocol 

高精度的时间校正协议,LAN上与标准间差小于1毫秒，WAN上几十毫秒  

1. 在linux下，我们可以通过自带的NTP(Network Time Protocol)协议通过网络使自己的系统保持精确的时间。  
2. 一般自己管理的网络中建立至少一台时间服务器来同步本地时间，这样可以使得在不同的系统上处理和收集日志和管理更加容易。

## Windows 与 Linux 的时间

1. 在Windows中,系统时间会自动保存在Bios的时钟里面，启动计算机的时候，系统会自动在Bios里面取硬件时间，以保证时间的不间断。
2. 在Linux中, 系统启动时会自动从Bios中取得硬件时间，设置为系统时间。此后系统时间和硬件时间以异步的方式运行，互不干扰。硬件时间的运行，是靠Bios电池来维持，而系统时间，是用CPU tick来维持的。


## date 时间命令

在Linux中查看和设置系统时间的命令

```shell
$ date
# 2008年 12月 12日 星期五 14:44:12 CST

# data --set 设置时间 有几种不同的格式可以使用

$ date --set "1/1/09 00:01"
# 2009年 01月 01日 星期四 00:01:00 CST
$ date 012501012009.30
# 2009年 01月 25日 星期日 01:01:30 CST

```

## clock 硬件时间命令

硬件时间的设置，可以用hwclock或者clock命令,二者用法相近，只用一个就行  
只不过clock命令除了支持x86硬件体系外，还支持Alpha硬件体系  

```shell

# 查看时间
$ hwclock 
$ hwclock --show
$ hwclock -r

# 把系统时间写入硬件时间 
$ hwclock -w

# 把硬件时间读取到系统时间
$ hwclock -s
```

## 时间相关的文件

1. `/usr/share/zoneinfo/`: 在这个目录下的文件其实是规定了各主要时区的时间设定文件，例如北京地区的时区设定文件在 `/usr/share/zoneinfo/Asia/Beijing`
2. `/etc/sysconfig/clock`：linux 的主要时区设定文件。每次开机后，Linux 会自动的读取这个文件来设定自己系统所默认要显示的时间。
3. `/etc/localtime`：本地端的时间配置文件, 根据 clock 文件的时区, 从 zoneinfo 中拷贝对应的时区便车给 locoltime 
4. `/etc/timezone`：系统时区文件


# 服务配置

所需要的软件其实仅有 ntp
* ntpd: 主要提供 NTP 服务的程序
* ntpdate :  用于客户端的时间校正，如果你没有要启用 NTP 而仅想要使用 NTP Client 功能的话，那么只会用到这个指令而已


1. 使用ntpd服务，要好于ntpdate加cron的组合。因为，ntpdate同步时间，会造成时间的跳跃，对一些依赖时间的程序和服务会造成影响。比如sleep，timer等。而且，ntpd服务可以在修正时间的同时，修正cpu tick。理想的做法为，在开机的时候，使用ntpdate强制同步时间，在其他时候使用ntpd服务来同步时间。
2. 因为 ntpd 有一个自我保护设置: 如果本机与上源时间相差太大, ntpd 不运行. 所以新设置的时间服务器一定要先 ntpdate 从上源取得时间初值, 然后启动 ntpd服务。ntpd服务运行后, 先是每64秒与上源服务器同步一次, 根据每次同步时测得的误差值经复杂计算逐步调整自己的时间, 随着误差减小, 逐步增加同步的间隔. 每次跳动, 都会重复这个调整的过程。

## ntpd 服务设置

一台机器，可以同时是ntp服务器和ntp客户机。在网络中，推荐使用像DNS服务器一样分层的时间服务器来同步时间   
网络的NTP server 尽量仅提供自己内部的 Client 端联机进行网络校时就好, 同时NTP Server 上面也要找一部最靠近自己的 Time Server 来进行自我校正  


`/etc/ntp.conf`：这个是NTP Daemon 的主要设文件，也是 NTP 唯一的设定文件。

参数介绍:  
* 权限设定 restrict `restrict [IP] mask [netmask_IP] [parameter]`
  *  IP 可以是软件地址，也可以是 default ，default 就类似 0.0.0.0 
  *  paramter 表示权力
     *  ignore : 不提供 NTP 联机服务 
     *  nomodify : 只读, 不能修改
     *  noquery : 不提供读取
     *  notrust : 该 Client 除非通过认证，否则拒绝该 Client 
     *  notrap : 不提供trap这个远程事件登入
     *  空 : 该 IP (或网域)“没有任何限制”
*  上层服务器设定 server  `server [IP|HOST Name] [prefer]`
   *  perfer 表示在设定了多个服务器后指定一个作为主要
*  driftfile 记录时间差异 `driftfile [可以被 ntpd 写入的目录与档案]`
   *  NTP Server 本身的时间计算是依据 BIOS 的芯片震荡周期频率来计算的，但是这个数值与上层 Time Server 不见得会一致
   *  NTP 这个 daemon (ntpd) 会自动的去计算我们自己主机的频率与上层 Time server 的频率，并且将两个频率的误差记录下来，记录下来的档案就是在 driftfile 后面接的完整档名当中了
* ntp服务，默认只会同步系统时间
  * 如果想要让ntp同时同步硬件时间，可以设置`/etc/sysconfig/ntpd` 文件
  * 在`/etc/sysconfig/ntpd`文件中，添加 `SYNC_HWCLOCK=yes`

参数设置
```shell
# restrict [IP] mask [netmask_IP] [parameter] 

restrict default nomodify notrap noquery　# 关闭所有的 NTP 要求封包 
restrict 127.0.0.1　　　 #这是允许本机查询
restrict 192.168.0.1 mask 255.255.255.0 nomodify 
#在192.168.0.1/24网段内的服务器就可以通过这台NTP Server进行时间同步了 


# server [IP|HOST Name] [prefer]
server cn.pool.ntp.org prefer
server  127.127.1.0     # local clock


#在启动NTP服务前，先对提供服务的这台主机手动的校正一次时间
$ ntpdate cn.pool.ntp.org
# 然后，启动ntpd服务
$  service ntpd start


```