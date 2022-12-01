# Network Shell (Netsh)

这是 windows 的命令行网络通信管理软件  

[官方中文文档](https://learn.microsoft.com/zh-cn/windows-server/networking/technologies/netsh/netsh) 

    适用于：Windows Server 2022、Windows Server 2019、Windows Server 2016、Azure Stack HCI、版本 21H2 和 20H2


作为一个命令行工具, Netsh 可以用来管理 Windows 的
* Windows Server 上配置和显示各种网络通信服务器角色和组件的状态
* Windows 10 客户端计算机上, 可以配置 DHCP 客户端
* 很多功能与 Microsoft 管理控制台 (MMC) 里的相同

## Netsh contexts

Netsh 类似于一个平台, 通过 DLL 动态链接库来实现在不同的平台上有不同的功能选项(extensive set of features called a contexts)

Netsh context : a group of commands specific to a networking server role or feature.
* contexts 使得 Netsh 作为一个平台可以管理很多的服务, 工具, 以及网络协议


通过在命令行键入  `netsh /?` 来获取在当前平台可以使用的所有命令(部件)  
* 基础的使用方法是在命令行中输入 `netsh` 来进入 NetSh 平台
* 再输入对应的 context 来进入对应功能的交互界面, 例如 `routing`
* 在各个 context 内部也可以使用 `/?` 来查看对应子 context 的使用方法
若要在命令行通过 one command 来实现对应的配置, 参考下文

## Netsh commands

`netsh[ -a AliasFile] [ -c Context ] [-r RemoteComputer] [ -u [ DomainName\ ] UserName ] [ -p Password | *] [{NetshCommand | -f ScriptFile}]`