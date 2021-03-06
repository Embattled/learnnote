- [1. Start Docker](#1-start-docker)
  - [1.1. 基本介绍](#11-基本介绍)
  - [1.2. Docker 架构基本概念](#12-docker-架构基本概念)
  - [1.3. 安装 Ubuntu](#13-安装-ubuntu)
    - [1.3.1. 通过仓库安装 docker](#131-通过仓库安装-docker)
    - [1.3.2. 运行权限](#132-运行权限)
  - [1.4. storage driver](#14-storage-driver)
- [2. Docker App](#2-docker-app)
  - [2.1. build](#21-build)
  - [2.2. image  管理](#22-image--管理)
  - [2.3. container 管理](#23-container-管理)
  - [2.4. tag](#24-tag)
  - [2.5. Docker repo](#25-docker-repo)
  - [2.6. Docker 运行命令](#26-docker-运行命令)
    - [2.6.1. 运行镜像](#261-运行镜像)
    - [2.6.2. 运行容器](#262-运行容器)
    - [2.6.3. 进入容器](#263-进入容器)
    - [2.6.4. 进程管理](#264-进程管理)
- [3. Dockerfile](#3-dockerfile)
# 1. Start Docker 

* Docker 是一个开源的应用容器引擎, 基于 `Go` 语言 并遵从 Apache2.0 协议开源
* 适合运维工程师及后端开发人员, 用于开发, 交付和运行应用程序的开放平台。
* Docker 可以让开发者打包他们的应用以及依赖包到一个轻量级、可移植的容器中, 然后发布到任何流行的 Linux 机器上
* 容器是完全使用沙箱机制, 相互之间不会有任何接口（类似 iPhone 的 app）,更重要的是容器性能开销极低


## 1.1. 基本介绍
应用场景:
1. Web 应用的自动化打包和发布。
2. 自动化测试和持续集成、发布。
3. 在服务型环境中部署和调整数据库或其他的后台应用。
4. 从头编译或者扩展现有的 OpenShift 或 Cloud Foundry 平台来搭建自己的 PaaS 环境。

使用目标:
1. 将应用程序与基础架构分开, 从而可以快速交付软件。
2. 与管理应用程序相同的方式来管理基础架构

开发流程:
1. 开发人员在本地编写代码, 并使用 Docker 容器与同事共享他们的工作。
2. 使用 Docker 将其应用程序推送到测试环境中, 并执行自动或手动测试。
3. 开发人员发现错误时, 他们可以在开发环境中对其进行修复, 然后将其重新部署到测试环境中, 以进行测试和验证。
4. 测试完成后, 将修补程序推送给生产环境, 就像将更新的镜像推送到生产环境一样简单


## 1.2. Docker 架构基本概念

* 镜像（Image）：Docker 本身的镜像（Image）, 就相当于是一个 root 文件系统。
* 容器（Container）：镜像是静态的定义, 容器是镜像运行时的实体。容器可以被创建、启动、停止、删除、暂停等。
* 仓库（Repository）：仓库可看成一个代码控制中心, 用来保存镜像。
  * 每个仓库可以包含多个标签（Tag）；每个标签对应一个镜像。
  * 通常, 一个仓库会包含同一个软件不同版本的镜像, 而标签就常用于对应该软件的各个版本。
  * 通过 <仓库名>:<标签> 的格式来指定具体是这个软件哪个版本的镜像。如果不给出标签, 将以 `latest` 作为默认标签。
* Docker Registry: 一个 Docker Registry 中可以包含多个仓库（Repository）
* Docker 主机(Host): 一个物理或者虚拟的机器用于执行 `Docker 守护进程`和`容器`。
* Docker 客户端(Client): 客户端通过命令行或者其他工具使用 Docker SDK与 Docker 的`守护进程`通信。
* Docker Machine : 是一个简化Docker安装的命令行工具, 通过一个简单的命令行即可在相应的平台上安装Docker, 比如VirtualBox、 Digital Ocean、Microsoft Azure。


1. 使用客户端-服务器 (C/S) 架构模式
2. 使用远程API来管理和创建Docker容器
3. Docker 容器通过 Docker 镜像来创建

## 1.3. 安装 Ubuntu

* apt 里能搜到的 `docker.io` 是旧版本, 不要安装
  * 卸载已有的版本 `$ sudo apt remove docker docker-engine docker.io containerd runc` 
  * docker 的文件库是 `/var/lib/docker/`  如果想清洁安装的话可以删除掉该文件夹

### 1.3.1. 通过仓库安装 docker

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
   3. 这样安装的话使用 apt upgrade 即可进行升级

### 1.3.2. 运行权限

* docker's daemon 是通过 Unix socket 来实现的, 而非 TCP port
* 而 Unix socket 是属于 root, 因此一般的用户执行 docker 命令需要 sudo
* 如果不想加 `sudo`, 可以创建一个名为 `docker`的用户组, docker会自动将权限赋予该用户组的用户
  * `sudo groupadd docker`
  * `sudo usermod -aG docker $USER`
  * `newgrp docker` 该命令用来刷新用户组的信息

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
2. 如果你的设施由别人提供技术支持, 那么选择它们擅长的；
3. 选择有比较完备的社区支持的。

不同存储驱动的简略说明:
* aufs: Docker最先使用的 storage driver 技术成熟 社区支持也很好
  * 有一些Linux发行版不支持AUFS, 主要是它没有被并入Linux内核
  * Docker在Debian, Ubuntu系的系统中默认使用aufs
* device mapper : 很稳定, 也有很好的社区支持
  * 在Linux2.6内核中被并入内核
  * docker 在 `RedHat` 系中默认使用 `device mapper`。
* overlayfs : 与AUFS相似, 也是一种联合文件系统(union filesystem)
  * 当前默认的
  * 设计更简单, 被加入Linux3.18版本内核, 可能更快
  * 在Docker社区中获得了很高的人气, 被认为比AUFS具有很多优势。
  * 但它还很年轻, 在成产环境中使用要谨慎。


# 2. Docker App

记录docker的相关基础命令


## 2.1. build

* 使用 `docker build [flags] dockerfile路径` 来创建 docker container
  * `-t 项目名` 用于指定该镜像的名称

## 2.2. image  管理

image 和 container 相关的命令比较类似 `docker image ls` 可以用来查看本机当前拥有的 image 以及对应的 tag
* `docker images` 和 `image ls` 功能相同
* `docker image rm [imageName]` 和 `docker rmi [imageName]` 用来删除一个 image

## 2.3. container 管理

* `docker container ls -l` 列出本机正在运行的容器
* `docker container ls -l --all` 列出本机所有容器, 包括终止运行的容器：
* `docker container rm [containerID]` 删除一个容器
* `docker ps -a` 列出所有容器
* 

## 2.4. tag

* `docker tag 旧名字 新名字` 用来给一个 image 赋予新的名字
* 新旧名字会同时存在, 但是都指向同一个 image, ID 是相同的

## 2.5. Docker repo

* `Docker Hub` 是 docker image 的标准 registry
* 通过命令行使用推送必须在本机进行登录
* `docker login -u 用户名` 然后输入密码
  * 登录信息会储存, 不需要重复输入
  
* 在 Docker Hub 中先创建好仓库后, 可以得到推送命令
  * `docker push 用户名/镜像名:tagname`
  * 注意 `用户名/镜像名` 是完整的镜像名, 必须确保 image 有这个名字
  * 否则需要用 `tag` 命令来添加新的命名
  * tagname 默认是 `latest`
* 使用 `docker pull` 命令来获取一个 prebuild image


## 2.6. Docker 运行命令

### 2.6.1. 运行镜像

* 使用 `docker run [OPTIONS] IMAGE [COMMAND] [ARG...]` 在本机的 docker engine 上运行一个镜像, 此时会创建一个容器
  * `docker run -dp 3000:3000 getting-started`
  * 使用 flags 来进行特殊设置
    * `-d` `detached mode` 运行程序, 即后台运行
    * `-p xx:xx ` 端口映射  `host:container`
    * `--name` 如果容器有名字的话, 可以使用名字调用
    * `--runtime==nvidia` 指定 docker 使用GPU
      * `-e NVIDIA_VISIBLE_DEVICES=1` 指定容器只能使用 GPU1

### 2.6.2. 运行容器

* 容器可以关闭, 再次启动时不能用`run`
  * 使用命令 `docker start [OPTIONS] CONTAINER [CONTAINER...]`
  * `docker start -i <name>-cuda-10.2`

### 2.6.3. 进入容器

在使用Docker创建了容器之后, 比较关心的就是如何进入该容器了, 其实进入Docker容器有好几多种方式

1. 使用 docker 提供的 attach命令 `docker attach 44fc0f0582d9`, 容器是单线程的, 当多个窗口同时用该命令进入容器, 所有的窗口都会同步显示, 有一个窗口阻塞了所有的窗口都不能进行操作
2. 使用 docker 的 exec 命令, 该命令会在容器中执行一个命令, 可以通过调用容器中的bash 来进入容器  `sudo docker exec -it 775c7c9ee1e1 /bin/bash`
3. 使用 nsenter 进入容器, 需要通过 docker 相关命令获得容器的 PID
    

### 2.6.4. 进程管理

* `docker ps`         类似于系统的同名命令, 显示所有正在运行的 docker 容器
* `docker stop [id]`  停止一个 docker 容器
* `docker rm <id>`    永久删除一个 docker 容器
  * 可以通过 `rm -f` 来直接停止并删除正在运行的容器


# 3. Dockerfile

* `Dockerfile` 是一个文本文件脚本, 用于创建一个容器镜像(注意没有文件后缀), 放在项目的根目录

```
FROM node:12-alpine
WORKDIR /app
COPY . .
RUN yarn install --production
CMD ["node", "src/index.js"]
```