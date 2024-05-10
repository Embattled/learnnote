# 1. Linux 的应用安装

Linux下的软件包众多，且几乎都是经 GPL 授权、免费开源（无偿公开源代码）

二进制包是 Linux 下默认的软件安装包，因此二进制包又被称为默认安装软件包。目前主要有以下 2 大主流的二进制包管理系统：

    1. RPM 包管理系统：功能强大，安装、升级、査询和卸载非常简单方便，因此很多 Linux 发行版都默认使用此机制作为软件安装的管理方式，例如 Fedora、CentOS、SuSE 等。
    2. DPKG 包管理系统：由 Debian Linux 所开发的包管理机制，通过 DPKG 包，Debian Linux 就可以进行软件包管理，主要应用在Debian和Ubuntu中。

RPM 包管理系统和 DPKG 管理系统的原理和形式大同小异，可以触类旁通。

## 1.1. 源码包安装

源码包一般包含多个文件，为了方便发布，通常会将源码包做打包压缩处理，Linux 中最常用的打包压缩格式为“tar.gz”，因此源码包又被称为 Tarball。  

Tarball 是 Linux 系统的一款打包工具，可以对源码包进行打包压缩处理，人们习惯上将最终得到的打包压缩文件称为 Tarball 文件。  

由于源码包的安装需要把源代码编译为二进制代码，因此安装时间较长。  

总的来说，使用源码包安装软件具有以下几点好处：

    开源。如果你有足够的能力，则可以修改源代码。
    可以自由选择所需的功能。
    因为软件是编译安装的，所以更加适合自己的系统，更加稳定，效率也更高。
    卸载方便。


## 1.2. 二进制安装  

### 1.2.1. 二进制软件包的命名规则

RPM 二进制包的命名需遵守统一的命名规则，用户通过名称就可以直接获取这类包的版本、适用平台等信息。

RPM 二进制包命名的一般格式如下：

包名-版本号-发布次数-发行商-Linux平台-适合的硬件平台-包扩展名

例如，RPM 包的名称是httpd-2.2.15-15.el6.centos.1.i686.rpm，其中：

| 字段   | 意义                                                                                                                                                                                                                                                |
| ------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| httped | 软件包名。这里需要注意，httped 是包名，而 httpd-2.2.15-15.el6.centos.1.i686.rpm 通常称为包全名，包名和包全名是不同的，在某些 Linux 命令中，有些命令（如包的安装和升级）使用的是包全名，而有些命令（包的查询和卸载）使用的是包名，一不小心就会弄错。 |
| 2.2.15 | 包的版本号，版本号的格式通常为主版本号.次版本号.修正号。                                                                                                                                                                                            |
| 15     | 二进制包发布的次数，表示此 RPM 包是第几次编程生成的。                                                                                                                                                                                               |
| el*    | 软件发行商，el6 表示此包是由 Red Hat 公司发布，适合在 RHEL 6.x (Red Hat Enterprise Unux) 和 CentOS 6.x 上使用。                                                                                                                                     |
| centos | 表示此包适用于 CentOS 系统。                                                                                                                                                                                                                        |
| i686   | 表示此包使用的硬件平台                                                                                                                                                                                                                              |



## 1.3. APT

APT由几个名字以“apt-”打头的程序组成。apt-get、apt-cache 和apt-cdrom是处理软件包的命令行工具。  
Debian 使用一套名为 Advanced Packaging Tool（APT）的工具来管理这种包系统,就是最常用的 Linux 包管理命令都被分散在了 apt-get、apt-cache 和 apt-config 这三条命令当中


而 apt 命令的引入就是为了解决命令过于分散的问题，它包括了 apt-get 命令出现以来使用最广泛的功能选项，以及 apt-cache 和 apt-config 命令中很少用到的功能。  

    简单来说就是：apt = apt-get、apt-cache 和 apt-config 中最常用命令选项的集合。

在 apt  中, 软件的各个文件的安装位置在 `.deb` 文件中都写死了, 很难修改安装位置

### 1.3.1. 基础被替换的命令

| apt 命令         | 取代的命令           | 命令的功能                     |
| ---------------- | -------------------- | ------------------------------ |
| apt install      | apt-get install      | 安装软件包                     |
| apt remove       | apt-get remove       | 移除软件包及其依赖包           |
| apt purge        | apt-get purge        | 移除软件包及配置文件           |
| apt update       | apt-get update       | 刷新存储库索引                 |
| apt upgrade      | apt-get upgrade      | 升级所有可升级的软件包         |
| apt autoremove   | apt-get autoremove   | 自动删除不需要的包             |
| apt full-upgrade | apt-get dist-upgrade | 在升级软件包时自动处理依赖关系 |
| apt search       | apt-cache search     | 搜索应用程序                   |
| apt show         | apt-cache show       | 显示安装细节                   |

### 1.3.2. 软件列表

| 新的apt命令      | 命令的功能                        |
| ---------------- | --------------------------------- |
| apt list         | 列出包含条件的包 已安装，可升级等 |
| apt edit-sources | 编辑源列表                        |

```shell

apt list --upgradeable

# 显示已安装的包
# [installed]指的是用户主动安装的包
# [installed, automatic] 指的是用户安装别的包时候的依赖
# apt remove 某个包，那么所有他依赖的包也会被移除
apt list --installed

```
### 1.3.3. 软件降级

```shell
# 本质上都是通过指定版本号来指定版本安装实现降级
apt install packagname=version

# 通过apt-cache在cache中查找旧的版本号
apt-cache show <name-of-program>  

#  通过policy 将显示特定包的所有可用版本以及安装位置
apt-cache policy <packagename>


```


# 2. update-alternatives 版本控制

用于处理linux系统中软件版本的切换，在各个linux发行版中均提供了该命令，命令参数略有区别，但大致是一样的  

```shell

# 注册位置
update-alternatives --install link name path priority
# link : 注册最终地址，成功后将会把命令在这个固定的目的地址做真实命令的软链，以后管理就是管理这个软链
#        link为系统中功能相同软件的公共链接目录，比如/usr/bin/java(需绝对目录)
# name : 服务名 以后管理时以它为关联依据
# path : 命令绝对路径
# priority : 优先级, 在自动模式下默认选择优先级最高的

# 示例
update-alternatives --install /usr/bin/java java /opt/jdk1.8.0_91/bin/java 200

# 删除一个注册
update-alternatives --remove java /opt/jdk1.8.0_91/bin/java
# 删除全部注册
update-alternatives --remove-all java

# 注册成功后可通过 name 查看已注册的所有版本
update-alternatives --display java

# 设置自动模式
update-alternatives --auto java

# 手动设置版本 交互
update-alternatives --config java
# 手动设置版本 命令
update-alternatives --set java /opt/jdk1.8.0_91/bin/java

```


以java为例，作为jre运行环境可以，但如果你作为开发测试环境，你会发现javac找不到  

原因是我们只对java命令做了版本管理。   
事实上，update-alternatives的原理是软链管理，可以处理目录。那么我们就可以把整个软件包目录都纳入管理。  

```shell


```