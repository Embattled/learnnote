# syslinux

syslinux是一个轻量级的启动装载器, 所谓轻量级就是与lilo和grub相比的,尤其是grub2

syslinux有很多变种（都是官方的）适用于各种媒质
1. syslinux用于从微软的文件系统fat 16/32引导
2. isolinux用于从光盘引导
3. pexlinux用于从网络引导
4. extlinux用于从ext2/3文件系统引导
因为兼容各种介质,主要用来建立修护或其它特殊用途的启动盘。  
它的安装很简单，一旦安装syslinux好之后，sysLinux启动盘就可以引导各种基于DOS的工具，以及MS-DOS/Windows或者任何其它操作系统。  
不仅支持采用BIOS结构的主板，而且从6.0版也开始支持采用EFI结构的新型主板。  



syslinux对于各种设备的兼容做的很细致，据`grub4dos`的人说，用于兼容性方面的代码甚至超过了程序主要功能的实现代码  


