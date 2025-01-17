# OpenWRT

开源的路由系统

https://openwrt.org/

固件很大程度上需要自己编译





# Documentation - Developer guide 开发教程


Overview: 
* OpenWrt 是一个 GNU/Linux distribution, 主要面向嵌入式设备
* OpenWrt 的包管理工具是 `opkg`
* 系统的所有的包都需要单独编译, 编译完成后会被安装在一个临时目录中, 并被压缩为 只读文件, 该文件最终会被写入设备的 `SquashFS` 分区作为设备固件
* kernel 也是作为一个 包, 但是他会议 bootloader 所期望的特殊方式被加遭到固件镜像中

构建过程 实际上是创建一个 firmware 文件, 该文件用于 安装 OpenWrt 或者升级  

firmware 的格式为 memory image, 通常开发者会直接称呼其为 image

How a package is compiled:
* package's Makefile 包含了源代码的下载链接
  * 对于 upstream project, 会记录 version number
  * 其他的记录方法还有 git commit, timestamp 等
* 每一个 包的文件夹中还有 `patches` 目录, 包含了 代码下载后, 但是在编译之前会应用的代码
  * 除此之外还有 configuration files
* kernel 的 Makefile 也是大差不差的
* 所有的 package 会使用 OpenWrt 独有的 toolchain
* 在运行固件构建脚本 `make` 的时候, 会先用系统的基础编译设施编译 OpenWrt 的 toolchain, 然后用工具链来编译固件, 这保证了编译的方便性以及避免了 cross-compiling

Package feeds:
* 大部分固件里的 包 并不是来源于 OpenWrt 项目本身
* 固件核心的包以外的包 通过 `package feeds` 方式来由社区的维护者提供

固件的地址
https://github.com/openwrt/openwrt

官方的 package feeds 的地址: 会由官方进行编译  
    https://github.com/openwrt/luci
    https://github.com/openwrt/telephony
    https://github.com/openwrt-routing/packages
    https://github.com/openwrt/packages

用户也可以定义自己的包源

## The OpenWrt source code

## Build system


### Build system setup - 配置 Build System
<!-- 完 -->
https://openwrt.org/docs/guide-developer/toolchain/install-buildsystem

参照官网给出的命令行, 配置 OpenWrt 的编译环境


## Build system usage - 如何正式开启编译  

https://openwrt.org/docs/guide-developer/toolchain/use-buildsystem

简要的步骤有
1. git下载代码
2. 选择版本并更新代码
3. 添加插件 修改 `feeds.conf.default` 文件
   1. https://openwrt.org/docs/guide-developer/toolchain/use-buildsystem#creating_a_local_feed
4. update feeds
5. make 配置固件
6. (可选) make 配置内核
7. make 编译


```sh
# 下载
git clone https://git.openwrt.org/openwrt/openwrt.git [<buildroot>]

git pull

# 选择一个稳定 tag
git tag
git checkout v23.05.5

# feeds 的更新和绑定
./scripts/feeds update -a
./scripts/feeds install -a

# 构建 配置文件
make menuconfig
# 或者直接下载官方配置文件
wget https://downloads.openwrt.org/releases/21.02.1/targets/ath79/generic/config.buildinfo -O .config

# download world 在编译前预选下载所有源代码, 确保多线程编译不出错
make -j5 download world
```

### Image configuration - 配置镜像

Menuconfig: 固件配置

`make menuconfig`
* 根据默认的 config 更新依赖项, 并进入 配置界面
* `/` 进入项目搜索功能
* 对于各种包, 有三种可选项
  * `y` 包被编译在 固件里
  * `m` 包单独编译, 不在固件里, 需要在安装了固件以后在 opkg 里单独安装
  * `n` 不编译的包
* 配置完成后, `.config` 会被创建


需要配置的重要项目  
* Target system (general category of similar devices)
* Subtarget (subcategory of Target system, grouping similar devices)
* Target profile (each specific device)
* Package selection
* Build system settings
* Kernel modules



