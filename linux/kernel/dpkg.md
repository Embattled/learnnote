# dpkg

dpkg is the software at the base of the package management system in the free operating system Debian and its numerous derivatives. 

是 debian 及其派生的 linux 发布所使用的最基础的 软件包管理工具及其组件 

官网: 	wiki.debian.org/Teams/Dpkg
wiki: https://en.wikipedia.org/wiki/Dpkg

dpkg 是一个软件管理格式, 是 apt 的 low-rank 

dpkg 本身同时是一个工具包, 提供了包括 `dpkg` 等一系列实用命令, 还包括
* dpkg-deb, dpkg-split, dpkg-query, dpkg-statoverride, dpkg-divert and dpkg-trigger.
* `update-alternatives` and `start-stop-daemon`
* The `install-info` program used to be included as well
  * but was later removed as it is now developed and distributed separately.
* The Debian package `dpkg-dev` includes the numerous build tools described below.


完整的 软件包似乎可以在这里 查找: https://packages.debian.org/sid/amd64/dpkg/filelist



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