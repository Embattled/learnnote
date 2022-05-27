# iptables 详解

iptables的主要功能是实现对网络数据包进出设备及转发的控制。当数据包需要进入设备、从设备中流出或者经该设备转发、路由时，都可以使用iptables进行控制。

iptables/netfilter 组成了linux平台下的包过滤防火墙  
有如下功能:  
网络地址转换NAT (Network Address Translate)  
数据包内容修改  
数据包过滤  


## 1. “四表五链”及“堵通策略”

**四表**
| 四表名称 | 功能及何可以控制的规则链                                                                      |
| -------- | --------------------------------------------------------------------------------------------- |
| raw      | 控制nat表中连接追踪机制的启用状况，可以控制的链路有prerouting, output                         |
| mangle   | 修改数据包中的原数据，可以控制的链路有所有五链                                                |
| nat      | 控制数据包中地址转换，可以控制的链路有prerouting,postrouting  ,Input,output                   |
| filter   | 控制数据包是否允许进出及转发（INPUT、OUTPUT、FORWARD）,可以控制的链路有input, forward, output |

4个表的优先级由高到低：raw-->mangle-->nat-->filter


**五链**  
五链是指内核中控制网络的`NetFilter`定义的五个规则链，分别为
| 名称                | 拥有的表              |
| ------------------- | --------------------- |
| PREROUTING, 路由前  | raw mangle nat        |
| INPUT, 数据包流入口 | mangle nat filter     |
| FORWARD, 转发管卡   | mangle filter         |
| OUTPUT, 数据包出口  | raw mangle nat filter |
| POSTROUTING, 路由后 | mangle nat            |

**堵通策略**  
是指对数据包所做的操作，一般有两种操作——“通（`ACCEPT`）”、“堵（`DROP`）”，还有一种操作很常见`REJECT`.

`iptables [-t table] COMMAND [chain] CRETERIA -j ACTION`  
-t table,是指操作的表,filter、nat、mangle或  raw, **默认使用filter**  
COMMAND,子命令,定义对规则的管理  
chain, 指明链路  
CRETERIA, 匹配的条件或标准  
ACTION,操作动作  

例如，不允许10.8.0.0/16网络对80/tcp端口进行访问

`iptables -A INPUT -s 10.8.0.0/16 -d 172.16.55.7 -p tcp --dport 80 -j DROP`

## 2. 命令COMMAND 

### a.查看命令  

| 查看规则                 | 细则                                                 |
| ------------------------ | ---------------------------------------------------- |
| -L, --list [chain]       | 列出规则,最主要的命令                                |
| -v, --verbose            | 详细信息；                                           |
| -vv， -vvv               | 更加详细的信息                                       |
| -n, --numeric            | 数字格式显示主机地址和端口号,不解析地址,执行速度更快 |
| -x, --exact              | 显示计数器的精确值；                                 |
| --line-numbers           | 列出规则时，显示其在链上的相应的编号,没有短命令      |
| -S, --list-rules [chain] | 显示指定链的所有规则                                 |

例:查看 filter表的规则列表  
`iptables -nL`

pkts    :对应规则匹配到的报文的个数  
bytes   :对应规则匹配到的报文包的大小总和  
target  :规则对应的动作,即匹配成功后采取的措施  
prot    :表示规则只对应的协议  
opt     :表示规则对应的选项  
in      :表示数据包由哪个网卡流入  
out     :数据包从哪个网卡流出  
source  :规则对应的源头地址,可以是一个ip  
destination:规则对应的目标地址  

### b.管理命令

| 规则管理 | 功能                                                                                    |
| -------- | --------------------------------------------------------------------------------------- |
| -A       | --append chain rule-specification：追加新规则于指定链的尾部；                           |
| -I       | --insert chain `rulenum` rule-specification：插入新规则于指定链的指定位置，默认为首部； |
| -R       | --replace chain `rulenum` rule-specification：替换指定的规则为新的规则；                |
| -D       | --delete chain `rulenum` :根据规则编号删除规则；                                        |

例: 路由转发的规则  
`iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE`  


| 链管理 | 功能                                                        |
| ------ | ----------------------------------------------------------- |
| -N     | --new-chain chain：新建一个自定义的规则链；                 |
| -X     | --delete-chain [chain]：删除用户自定义的引用计数为0的空链； |
| -F     | --flush [chain]：清空指定的规则链上的规则；                 |
| -E     | --rename-chain old-chain new-chain：重命名链；              |
| -Z     | --zero [chain [rulenum]]：置零计数器                        |
| -P     | --policy chain target， 设置链路的默认策略                  |


## 3. 匹配条件CRETERIA

### a.基本匹配条件

### b.扩展匹配条件


## 4. 处理动作Action

| 动作       | 功能                                                                            |
| ---------- | ------------------------------------------------------------------------------- |
| ACCEPT     | 允许数据通过                                                                    |
| DROP       | 扔掉数据,不作回应                                                               |
| REJECT     | 拒绝数据,回应一个拒绝信息包                                                     |
| SNAT       | 源地址转换,解决内网用户用同一个公网地址上网的问题                               |
| MSAQUERADE | SNAT的一种特定形式,用于动态IP                                                   |
| DNAT       | 目标地址转换                                                                    |
| REDIRECT   | 再本机做端口映射                                                                |
| LOG        | 在`/var/log/nessages`文件中记录日志信息,然后将数据包传给下一条规则,只是单纯记录 |