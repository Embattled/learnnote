# WSL的使用

## 1. 安装与配置WSL2

1. 启用“适用于 Linux 的 Windows 子系统”可选功能
   `dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart`  
   重启后即可安装发行版linux到WSL1
2. 启用“虚拟机平台”可选组件
   `dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart`  
   重新启动计算机，以完成 WSL 安装并更新到 WSL 2。  
3. 将 WSL 2 设置为默认版本
   `wsl --set-default-version 2`  
   这会将安装的任何新分发版的版本设置为 WSL 2  

## 2.WSL版本管理与启动

1. 检查分配给每个已安装的 Linux 分发版的 WSL 版本  
   `wsl --list --verbose`  

   `wsl --list --all` 列出所有分发，包括当前不可用的分发  
   `wsl --list --running`   列出当前正在运行的所有分发版  
2. 更改linux的启动wsl版本
   `wsl --set-version <distribution name> <versionNumber>`  
3. 更改使用`wsl`命令时默认启动的分发版  
   `wsl -s <DistributionName>`  例如 `wsl -s Ubuntu`
    `wsl -d <DistributionName>` 运行特定的分发版  
    `wsl -u <Username>`  以特定用户的身份运行   
4. 若要查看特定于分发版的命令
    `[distro.exe] /?`  例如 `ubuntu /?`  
5. 删除与重新安装  
   wsl分发版不能通过商店卸载, 可以通过 WSL Config 来取消注册/卸载分发版  
   `wsl --unregister <DistributionName>`  从 WSL 中取消注册分发版，以便能够重新安装或清理它。 若要重新安装，请在 Microsoft Store 中找到该分发版，然后选择“启动”。


从命令行运行 WSL 的最佳方式是使用 ` wsl.exe` , 这会保留当前的工作命令并切换到linux中  


## 3. 系统相关
在Linux中的项目尽量保存到Linux的文件系统中,才能更快的访问  

在Linux的根目录 输入 ` explorer.exe . `(不要忘记最后的点) 使用Windows文件资源管理器打开WSL文件系统  

从 Windows (localhost) 访问 Linux 网络应用:  
若要查找为 Linux 分发版提供支持的虚拟机的 IP 地址，请执行以下操作:  
* 在 WSL 分发版（即 Ubuntu）中运行以下命令：ip addr
* 查找并复制 eth0 接口的 inet 值下的地址。
* 如果已安装 grep 工具，请通过使用以下命令筛选输出来更轻松地查找此地址：* ip addr | grep eth0
* 使用此 IP 地址连接到 Linux 服务器。
  


## 扩展 WSL 2 虚拟硬件磁盘的大小  
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
