# 1. WSL的使用

## 1.1. 安装与配置WSL2

1. 启用“适用于 Linux 的 Windows 子系统”可选功能
   `dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart`  
   重启后即可安装发行版linux到WSL1
2. 启用“虚拟机平台”可选组件
   `dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart`  
   重新启动计算机，以完成 WSL 安装并更新到 WSL 2。  
3. 将 WSL 2 设置为默认版本
   `wsl --set-default-version 2`  
   这会将安装的任何新分发版的版本设置为 WSL 2  
4. WSL 内核查看与更新
   `wsl --status`
   `wsl --update`


## 1.2. WSL版本管理与启动

1. 检查分配给每个已安装的 Linux 分发版的 WSL 版本  
   `wsl --list --verbose`  

   `wsl --list --all` 列出所有分发，包括当前不可用的分发  
   `wsl --list --running`   列出当前正在运行的所有分发版  
2. 更改linux的启动wsl版本
   `wsl --set-version <distribution name> <versionNumber>`  
3. 更改使用`wsl`命令时默认启动的分发版  
   `wsl -s <DistributionName>`  例如 `wsl -s Ubuntu`
   `wsl -d <DistributionName>` 运行特定的分发版  

4. 若要查看特定于分发版的命令
    `[distro.exe] /?`  例如 `ubuntu /?`  
5. 删除与重新安装  
   wsl分发版不能通过商店卸载, 可以通过 WSL Config 来取消注册/卸载分发版  
   `wsl --unregister <DistributionName>`  从 WSL 中取消注册分发版，以便能够重新安装或清理它。 若要重新安装，请在 Microsoft Store 中找到该分发版，然后选择“启动”。


从命令行运行 WSL 的最佳方式是使用 ` wsl.exe` , 这会保留当前的工作命令并切换到linux中  

## 1.3. CLI 参数

`wsl -u <Username>`  以特定用户的身份运行   
* 可以用于在 windows 层面来运行一些需要 sudo 的指令
* `wsl -u root`  就可以在不输入 sudo 密码的情况下进行系统层面的功能, 例如服务启动等

## 1.4. 系统相关
在Linux中的项目尽量保存到Linux的文件系统中,才能更快的访问  

在Linux的根目录 输入 ` explorer.exe . `(不要忘记最后的点) 使用Windows文件资源管理器打开WSL文件系统  

从 Windows (localhost) 访问 Linux 网络应用:  
若要查找为 Linux 分发版提供支持的虚拟机的 IP 地址，请执行以下操作:  
* 在 WSL 分发版（即 Ubuntu）中运行以下命令：ip addr
* 查找并复制 eth0 接口的 inet 值下的地址。
* 如果已安装 grep 工具，请通过使用以下命令筛选输出来更轻松地查找此地址：* ip addr | grep eth0
* 使用此 IP 地址连接到 Linux 服务器。
  


## 1.5. 扩展 WSL 2 虚拟硬件磁盘的大小  
WSL 2 使用虚拟硬件磁盘 (VHD) 来存储 Linux 文件。 如果达到其最大大小，则可能需要对其进行扩展。

WSL 2 VHD 使用 ext4 文件系统。 此 VHD 会自动调整大小以满足你的存储需求，并且其最大大小为 256GB。 如果你的分发版大小增长到大于 256GB，则会显示错误，指出磁盘空间不足。 可以通过扩展 VHD 大小来纠正此错误。

若要将最大 VHD 大小扩展到超过 256GB，请执行以下操作：

    使用 wsl --shutdown 命令终止所有 WSL 实例

    查找你的分发版安装包名称（“PackageFamilyName”）
        使用 PowerShell（其中，“distro”是分发版名称）输入以下命令：
        Get-AppxPackage -Name "*<distro>*" | Select PackageFamilyName

    找到 WSL 2 安装使用的 VHD 文件 fullpath，这将是你的 pathToVHD：
        %LOCALAPPDATA%\Packages\<PackageFamilyName>\LocalState\<disk>.vhdx

    通过完成以下命令调整 WSL 2 VHD 的大小：
        以管理员权限打开 Windows 命令提示，然后输入：
            diskpart
            Select vdisk file="<pathToVHD>"
            expand vdisk maximum="<sizeInMegaBytes>"

    启动 WSL 分发版（例如 Ubuntu）。

    通过从 Linux 分发版命令行运行以下命令，让 WSL 知道它可以扩展其文件系统的大小：
        sudo mount -t devtmpfs none /dev
        mount | grep ext4
        复制此项的名称，该名称类似于：/dev/sdXX（X 表示任何其他字符）
        sudo resize2fs /dev/sdXX
        使用前面复制的值。 可能还需要安装 resize2fs：apt install resize2fs

## 1.6. x11

export DISPLAY=$(awk '/nameserver / {print $2; exit}' /etc/resolv.conf 2>/dev/null):0 # in WSL 2
export LIBGL_ALWAYS_INDIRECT=1

# 2. usbipd

桥接物理 usb 设备与 wsl

usbipd list
usbipd wsl list
usbipd wsl attach
usbipd wsl detach --busid 4-4


# 3. Concepts

8秒规则 : 所有对 wsl 配置的修改都需要经过 `wsl --list --running` 确认容器已经完全停止的情况下, 重启才会刷新配置  


 
## 3.1. Advanced settings configuration: in WSL

两个文件 `wsl.conf` `.wslconfig` 用于为每个 wsl 系统和 所有 wsl 系统进行个性化配置  


    .wslconfig to configure settings globally across all installed distributions running on WSL 2.
    wsl.conf to configure settings per-distribution for Linux distros running on WSL 1 or WSL 2.


### 3.1.1. wsl.conf

可以同时用于 wsl1 wsl2 , 为单个 wsl distribution 进行配置  

全局拥有 5 个模块
* automount
* network
* interop
* user
* boot 新追加的

**Automount settings** : `[automount]`


**Network settings** : `[network]`

| key                | 类型    | 默认值           | 功能                                                              |
| ------------------ | ------- | ---------------- | ----------------------------------------------------------------- |
| generateHosts      | boolean | true             | 设置 wsl 自动生成 `/etc/hosts`, 包括主机名以及对应的 ip 静态地址  |
| generateResolvConf | boolean | true             | 设置 wsl 自动生成 `/etc/resolv.conf` , 包括了 wsl 使用的 DNS 列表 |
| hostname           | string  | Windows hostname | Sets hostname to be used for WSL distribution.                    |

**Interop settings** : `[interop]`   available in Insider Build 17713 and later.
| key               | 类型    | 默认值 | 功能                                                       |
| ----------------- | ------- | ------ |
| enabled           | boolean | true   | 决定 wsl 中是否可以调用 windows 的进程                     |
| appendWindowsPath | boolean | true   | 决定是否把 windows 的环境变量 PATH 追加到 wsl 的环境变量中 |


**User settings** : `[user]` available in Build 18980 and later.
| key     | 类型   | 默认值                                    | 功能                                    |
| ------- | ------ | ----------------------------------------- |
| default | string | The initial username created on first run | 指定 wsl 终端启动的时候要用的默认的用户 |

**Boot setting** : `[boot]` available on Windows 11 and Server 2022.

许多 linux 发行版默认运行在 systemd 上, 但是 WSL 默认并非如此, 如果要启动 systemd 需要 wsl 模块作为单独的 package 安装在系统上 (需要windows 11) 并在 0.67.6 版本以上
在重启后可通过 `systemctl list-unit-files --type=service`  来确认 systemd 内核是否启动  
| key     | 类型   | 默认值                                    | 功能                                    |
| ------- | ------ | ----------------------------------------- |
command  | string | `""` | 用于在 wsl 启动的时候自动以 root 运行的命令, 主要用于服务启动
systemd | boolean | false | systemd 管理命令

```sh
[boot]
systemd=true
command = service docker start
```

### 3.1.2. .wslconfig


## 3.2. File access and permissions: for WSL

在该篇章中说明 如何通过 Windows 来解释 WSL 的Linux 文件权限  

从 wsl 访问windows 的时候, 文件权限是根据 Windows 的当前用户确定的
* 另一方面 WSL 可以向文件添加元数据, 用于确定权限. 元数据默认不启用

## 3.3. Networking considerations: Accessing network applications with WSL - WSL 的网络配置 


WSL 同外界通信的规则
* NAT based architecture (默认)
* Mirrored Networking mode (最新)


确定 IP 地址 : 在使用 wsl 的时候, IP 地址的时候需要考虑 2 种场景
1. 从 Windows host, 你可能需要获取 WSL 的ip 地址, 用来从 windows 端直接访问在 wsl 运行的服务器程序
    wsl 端运行 `hostname -I` 即, 从windows 端运行 `wsl [-d <DistributionName>] hostname -I`
    典型的 ip `172.30.98.229`
2. 从 wsl 端 获取 windows 的 ip 地址, 用来访问运行在 windows 的服务器程序
    wsl 运行 `ip route show | grep -i default | awk '{ print $3}'`
    同理 典型的 ip 为 `172.30.96.1`

上述操作发生在 NAT based 网络模式中, 需要执行地址查询. 而镜像模式下, 则不需要查询, 直接通过 `127.0.0.1` 来访问


### 3.3.1. NAT mode (default)

默认, wsl 通过 NAT(Network Address Translation) 来访问网络.


* 从 Windows 访问 Linux: e.g. 在 WSL 上建立 SQL 服务器, 则可以直接用 local
* 从 Linux 访问 Windows: e.g. 在 Windows 上建立的 SQL 服务器, 从 Linux 访问, 这不是一个常见的场景, 但仍然是可实现的 , 参考上述实现方法  


**从外部网络访问** 应用程序, 此时程序需要绑定监听 `0.0.0.0`, 即允许任意地址的访问.  

**从 LAN (Local area network) 访问 WSL**
* 尽管 WSL1 允许直接通过 LAN 访问 WSL, 但 WSL2 默认的 NAT 不允许
* WSL2 下需要按照标准虚拟机的方式配置转发规则(wsl团队正在尝试进行改善)
```powershell
netsh interface portproxy add v4tov4 listenport=<yourPortToForward> listenaddress=0.0.0.0 connectport=<yourPortToConnectToInWSL> connectaddress=(wsl hostname -I)
```
  * 获取 wsl2 的 ip 地址
  * `wsl hostname -I` wsl 的 ip 地址
    * (ps. 小写的 i 会返回本地计算机看自己的地址, 通常是 127.0.0.1)
  * `cat /etc/resolv.conf` 从 wsl 看 windows 的地址


### 3.3.2. IPv6 访问





### 3.3.3. Mirrored mode networking

Windows11 所支持的全新的网络结构, 可以通过在 `.wslconfig` 中指定
* `networkingMode=mirrored` 来启用, 其目的是提高网络的兼容性
* 完全将 Windows 的网络接口镜像到 WSL 上


根据官方说明, 该模式下的优点有以下
* IPv6 support                  
* Connect to Windows servers from within Linux using the localhost address 127.0.0.1. IPv6 localhost address ::1 is not supported
* 提高网络和 VPN 的兼容性, Improved networking compatibility for VPNs
* Multicast support 
* Connect to WSL directly from your local area network (LAN)

针对 LAN 链接到 WSL, 需要使用管理员权限在 PowerShell 中开启防火墙端口来允许 入境 链接
(配置 Hyper-V firewall)

```powershell
Set-NetFirewallHyperVVMSetting -Name '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' -DefaultInboundAction Allow
# OR
New-NetFirewallHyperVRule -Name "MyWebServer" -DisplayName "My Web Server" -Direction Inbound -VMCreatorId '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' -Protocol TCP -LocalPorts 80
```

### 3.3.4. DNS Tunneling
* 在 Windows11 以后, 通过在 wslconfig 里设置 `dnsTunneling=true` 默认值
* 使得  WSL 使用虚拟化的功能来应答 wsl内部的DNS请求, 可以提高网络和 VPN 的兼容性

**Auto Proxy**
* 设置 `autoProxy=true (默认值) ` 可以使得 WSL 强制使用 Windows 的 HTTP proxy 代理
* 开启该功能可以使得 wsl 无需进行 代理配置


