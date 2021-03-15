# 1. Docker 

* Docker 是一个开源的应用容器引擎，基于 `Go` 语言 并遵从 Apache2.0 协议开源
* 适合运维工程师及后端开发人员, 用于开发，交付和运行应用程序的开放平台。
* Docker 可以让开发者打包他们的应用以及依赖包到一个轻量级、可移植的容器中，然后发布到任何流行的 Linux 机器上
* 容器是完全使用沙箱机制，相互之间不会有任何接口（类似 iPhone 的 app）,更重要的是容器性能开销极低


## 1.1. 基本介绍
应用场景:
1. Web 应用的自动化打包和发布。
2. 自动化测试和持续集成、发布。
3. 在服务型环境中部署和调整数据库或其他的后台应用。
4. 从头编译或者扩展现有的 OpenShift 或 Cloud Foundry 平台来搭建自己的 PaaS 环境。

使用目标:
1. 将应用程序与基础架构分开，从而可以快速交付软件。
2. 与管理应用程序相同的方式来管理基础架构

开发流程:
1. 开发人员在本地编写代码，并使用 Docker 容器与同事共享他们的工作。
2. 使用 Docker 将其应用程序推送到测试环境中，并执行自动或手动测试。
3. 开发人员发现错误时，他们可以在开发环境中对其进行修复，然后将其重新部署到测试环境中，以进行测试和验证。
4. 测试完成后，将修补程序推送给生产环境，就像将更新的镜像推送到生产环境一样简单


## 1.2. Docker 架构基本概念

* 镜像（Image）：Docker 本身的镜像（Image），就相当于是一个 root 文件系统。
* 容器（Container）：镜像是静态的定义，容器是镜像运行时的实体。容器可以被创建、启动、停止、删除、暂停等。
* 仓库（Repository）：仓库可看成一个代码控制中心，用来保存镜像。
  * 每个仓库可以包含多个标签（Tag）；每个标签对应一个镜像。
  * 通常，一个仓库会包含同一个软件不同版本的镜像，而标签就常用于对应该软件的各个版本。
  * 通过 <仓库名>:<标签> 的格式来指定具体是这个软件哪个版本的镜像。如果不给出标签，将以 `latest` 作为默认标签。
* Docker Registry: 一个 Docker Registry 中可以包含多个仓库（Repository）
* Docker 主机(Host): 一个物理或者虚拟的机器用于执行 `Docker 守护进程`和`容器`。
* Docker 客户端(Client): 客户端通过命令行或者其他工具使用 Docker SDK与 Docker 的`守护进程`通信。
* Docker Machine : 是一个简化Docker安装的命令行工具，通过一个简单的命令行即可在相应的平台上安装Docker，比如VirtualBox、 Digital Ocean、Microsoft Azure。


1. 使用客户端-服务器 (C/S) 架构模式
2. 使用远程API来管理和创建Docker容器
3. Docker 容器通过 Docker 镜像来创建

## 1.3. 安装 Ubuntu

* apt 里能搜到的 `docker.io` 是旧版本, 不要安装
  * 卸载已有的版本 `$ sudo apt remove docker docker-engine docker.io containerd runc` 
  * docker 的文件库是 `/var/lib/docker/`  如果想清洁安装的话可以删除掉该文件夹

### 通过仓库安装 docker

0. 添加docker 的 apt 公钥
   * `sudo apt-key adv --recv-keys --keyserver keyserver.Ubuntu.com F273FCD8 `
1. 设置仓库
   1. 安装 apt 的相关依赖, 使得允许 `apt 通过 http 来使用一个仓库`
      * sudo apt-get update
      * sudo apt-get install 
        * apt-transport-https
        * ca-certificates
        * curl
        * gnupg
        * lsb-release
   2. Add Docker’s official GPG key
      * `curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg`
   3. 设置 stable(稳定版) 仓库
      * 在 `stable` 后面添加 `nightly` or `test` 可以获取测试版
      * `命令不复制了`
2. 安装 docker 引擎
   1. 先更新 apt index
      * `sudo apt-get update`
   2. 安装最新版的 docker engine
      * `sudo apt-get install docker-ce docker-ce-cli containerd.io`



### 通过 docker repository
* 在一台新主机上先设置 docker repository
* 在仓库中可以进行 Docker 的安装和更新

1. 

## 1.4. storage driver

storage driver 和 file system 不是同一个东西. 存储驱动是在文件系统之上创建的, 可以使用的存储驱动和文件系统有关:  
    例: `btrfs` 只能在文件系统为 `btrfs` 上的主机上使用

docker支持许多存储驱动, 在 `Ubuntu`下支持 : `overlay2, aufs, btrfs`  
使用`docker info` 可以查看当前所使用的存储驱动   

**修改存储驱动的方法:**
1. `docker daemon` 命令中添加 `--storage-driver=<name>`标识来指定
2. 在 `/etc/default/docker` 文件中通过 `DOCKER_OPTS`

不同的存储驱动对容器中的应用是有影响的,  选择原则:
1. 选择你及你的团队最熟悉的；
2. 如果你的设施由别人提供技术支持，那么选择它们擅长的；
3. 选择有比较完备的社区支持的。

不同存储驱动的简略说明:
* aufs: Docker最先使用的 storage driver 技术成熟 社区支持也很好
  * 有一些Linux发行版不支持AUFS，主要是它没有被并入Linux内核
  * Docker在Debian，Ubuntu系的系统中默认使用aufs
* device mapper : 很稳定，也有很好的社区支持
  * 在Linux2.6内核中被并入内核
  * docker 在 `RedHat` 系中默认使用 `device mapper`。
* overlayfs : 与AUFS相似，也是一种联合文件系统(union filesystem)
  * 当前默认的
  * 设计更简单, 被加入Linux3.18版本内核, 可能更快
  * 在Docker社区中获得了很高的人气，被认为比AUFS具有很多优势。
  * 但它还很年轻，在成产环境中使用要谨慎。

