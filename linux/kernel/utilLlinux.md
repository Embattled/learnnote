# util-linux

https://en.wikipedia.org/wiki/Util-linux

Linux 官方维护的一个工具包, 是作为 linux 内核的一部分的, 非常基础.  
util-linux is a standard package distributed by the Linux Kernel Organization for use as part of the Linux operating system.

大部分 linux 基础命令属于该软件包


# mount umount - 挂载

* mount - mount a filesystem
* umount - unmount file systems

## mount 命令


```sh
mount [-l|-h|-V]
mount -a [-fFnrsvw] [-t fstype] [-O optlist]
mount [-fnrsvw] [-o options] device|dir
mount [-fnrsvw] [-t fstype] [-o options] device dir
```

所有的 linux 文件和设备都作为 根目录`/` 的枝叶 来管理,  mount 用来把 device attach 到文件系统中.  
Linux 下所有硬件设备都必须挂载后才能使用, 区别是硬盘分区挂载被写入了系统启动脚本, 而其他设备(例如U盘等)需要手动挂载.  

* `mount [-l|-h|-V]`  基础命令 
  * 打印所有已经 mount 的设备
    * `mount | column -t` : 用于清晰的输出
  * ` -l, --show-labels`  打印的同时输出通过 `e2label` 设置的label, 挂载设备的卷标, 实测下来 wsl 下输出没有区别.
  * `-h` 打印帮助命令


* `mount -a [-fFnrsvw] [-t fstype] [-O optlist]`  自动挂载
  * 重新读取 `/etc/fstab` 检查文件中有无疏漏被挂载的设备文件


* `mount [-fnrsvw] [-t fstype] [-o options] device dir`   标准挂载
  * `-t`    : 想要挂载的硬件的系统类型, 可以自动检测





### options

更加细节的选项参数, 需要参照文档中的  
*  FILESYSTEM-INDEPENDENT MOUNT OPTIONS
*  FILESYSTEM-SPECIFIC MOUNT OPTIONS

`-o, --options opts`
具体的使用为  a comma-separated list,  e.g. `mount LABEL=mydisk -o noatime,nodev,nosuid`

