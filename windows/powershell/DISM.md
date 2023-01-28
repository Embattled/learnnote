# Deployment Image Servicing and Management (DISM)

开发镜像以及管理平台是用于在部署 Windows 之前进行 Windows 的装载和服务配置, 即对 Windows 系统安装镜像 `.wim` 进行各种配置, 也有一些命令可以用于对当前正在运行的系统进行配置


通过 DISM 的各种命令, 可以进行
* mount, and get information about, Windows image (.wim) files or virtual hard disks (.vhd or .vhdx).
* install, uninstall, configure, and update Windows features, packages, and drivers in a Windows image or to change the edition of a Windows image.
有点类似于对系统镜像进行自定义的感觉  



作为一个具体的平台, DISM 提供了
* 命令行工具 DISM.exe
* DISM API

# WindowsCapability 管理

Windows capabilities for an image or a running operating system.

## Get-WindowsCapability 情况确认

Gets Windows capabilities for an image or a running operating system.

## Add-WindowsCapability 添加功能



Installs a Windows capability package on the specified operating system image.
