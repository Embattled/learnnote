# PowerShell Overview


* [PowerShell Document](https://learn.microsoft.com/en-us/powershell/scripting/overview)



根据文档的构成形式, Windows PowerShell 的功能大概为:
* 像 linux 一样通过命令行来 管理, 维护, 配置 windows 电脑或者服务器
* 由于 Windows 目前只由微软来维护, 所以整体上文档很统一, 较为规整
* 要注意, 大部分功能配置都是能够在 GUI 图形界面找到对应设置的, 因此 命令行配置基本上只服务于专业人士


## What is Windows PowerShell?

有趣的来了: Windows PowerShell and PowerShell are two separate products.
* `Windows PowerShell` 是 Windows 操作系统中附带的 PowerShell 版本, 该版本完全使用 `.NET Framework`来实现, 使得该版本只能用于 Windows. Windows Powershell 的最新版本是 5.1. 且 微软已经停止向 Windows PowerShell 更新新功能. Windows PowerShell 的支持服务与 Windows 操作系统绑定
* `PowerShell` 是基于新版本的 `.NET` 为不是 `.NET Framework`, 可以在 Windows, Linux, macOS 三者上运行. 



## What is a PowerShell command (cmdlet)?  cmdlet

cmdlet 是微软为 PowerShell 中原始的功能模组定义的名称 (pronounced command-lets)

具体表现为
* Cmdlets are native PowerShell commands, not stand-alone executables. 
* Cmdlets are collected into PowerShell modules that can be loaded on demand. 
* Cmdlets can be written in any compiled `.NET` language or in the `PowerShell scripting language` itself.


## 


# What's New in PowerShell
