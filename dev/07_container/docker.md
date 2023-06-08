- [1. Start Docker](#1-start-docker)
  - [1.1. 基本介绍](#11-基本介绍)
  - [1.2. Docker 架构基本概念](#12-docker-架构基本概念)
  - [1.3. 安装 Docker](#13-安装-docker)
  - [1.4. Ubuntu 安装](#14-ubuntu-安装)
    - [1.4.1. 通过仓库安装 docker](#141-通过仓库安装-docker)
    - [1.4.2. 运行权限](#142-运行权限)
  - [1.5. storage driver](#15-storage-driver)
- [2. Docker Engine](#2-docker-engine)
  - [2.1. docker build 镜像编译 - Build an image from a Dockerfile](#21-docker-build-镜像编译---build-an-image-from-a-dockerfile)
    - [2.1.1. build options](#211-build-options)
  - [2.2. docker compose 镜像统合](#22-docker-compose-镜像统合)
  - [2.3. docker image 镜像管理](#23-docker-image-镜像管理)
  - [2.4. docker container 容器管理](#24-docker-container-容器管理)
  - [2.5. docker tag  -  Create a tag TARGET\_IMAGE that refers to SOURCE\_IMAGE](#25-docker-tag-----create-a-tag-target_image-that-refers-to-source_image)
  - [2.6. docker run 启动镜像 核心命令](#26-docker-run-启动镜像-核心命令)
  - [docker stop  - Stop one or more running containers 停止容器](#docker-stop----stop-one-or-more-running-containers-停止容器)
  - [2.7. docker start - 容器启动(重启) Start one or more stopped containers](#27-docker-start---容器启动重启-start-one-or-more-stopped-containers)
  - [2.8. docker attach - 接入容器](#28-docker-attach---接入容器)
  - [2.9. 进程管理](#29-进程管理)
- [3. Docker Hub](#3-docker-hub)
- [4. Dockerfile](#4-dockerfile)


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

基础概念
* `Dockerfile` : 纯文本, 用于定义一个 Image 的各种参数, 环境, 库 等
* 镜像 (Image): 由 Dockerfile 编译(Build) 得来, 是一个 root 文件系统, 但是 Image 本身是一个静态的概念
* 容器 (Container) : 镜像是静态的定义, 容器是镜像运行时的实体. 镜像是容器的 template, 容器可以被创建、启动、停止、删除、暂停等.
* 应用 (application) : 完整的一个应用程序, 可以包括多个容器, 每个容器运行各自的 process or service, 容器之间通过一定的协议进行通信, 最终实现整个应用程序


* 仓库（Repository）: 仓库可看成一个代码控制中心, 用来保存镜像。
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


## 1.3. 安装 Docker

Docker 目前分为两个版本
* Docker Desktop    : GUI 版本的, 在 WSL 环境下并不适合安装该版本
* Docker Engine     : 原生 CLI 版本的
  * 一个运行在后台的 daemon : `dockerd`
  * 一套 API 接口, 可以使程序直接对接 Docker daemon
  * 一个 CLI 程序接口 : `docker`, 基于 Docker API 来管理 Docker 容器

## 1.4. Ubuntu 安装

* apt 里能搜到的 `docker.io` 是旧版本, 不要安装
  * 卸载已有的版本 `$ sudo apt remove docker docker-engine docker.io containerd runc` 
  * docker 的文件库是 `/var/lib/docker/`  如果想清洁安装的话可以删除掉该文件夹



### 1.4.1. 通过仓库安装 docker


1. 设置仓库
   * 安装 apt 的相关依赖, 使得允许 `apt 通过 http 来使用一个仓库`
```bash
sudo apt-get update
sudo apt-get install \
    ca-certificates \
    curl \
    gnupg \
    lsb-release
```

2. Add Docker’s official GPG key
```bash
sudo mkdir -m 0755 -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
```

3. 设置 stable(稳定版) 仓库
```sh
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```

4. 开始正式安装
```sh
# 先更新 apt index
`sudo apt-get update`

# 安装
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# 验证
sudo docker run hello-world
```

### 1.4.2. 运行权限

* docker's daemon 是通过 Unix socket 来实现的, 而非 TCP port
* 而 Unix socket 是属于 root, 因此一般的用户执行 docker 命令需要 sudo
* 如果不想加 `sudo`, 可以创建一个名为 `docker`的用户组, docker会自动将权限赋予该用户组的用户
  * `sudo groupadd docker`
  * `sudo usermod -aG docker $USER`
  * `newgrp docker` 该命令用来刷新用户组的信息

## 1.5. storage driver

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


# 2. Docker Engine

记录docker engine 的相关基础CLI运行命令

docker 是一个基础接口, 其下的各种子命令都是独立的 binary, 分别有其各自自己的 --help 和其他各种子命令


容器的创建, 停止, 再运行, 进入都分别有各自的命令, 具体的生命线可以是
* docker build  : 从 dockerfile 编译出一个镜像
* docker run    : 一开始的创建, 同时运行. Create and run a new container from an image
* docker stop   : 停止一个容器的运行
* docker start  : 容器的再启动
* docker rm     : 删除一个容器实例


## 2.1. docker build 镜像编译 - Build an image from a Dockerfile

docker build 是 docker engine 的基础命令, 通过 build 来创建一个 image

目前该命令有两个组成部分
* 从 18.09 版本开始, 以 `Moby BuildKit` 作为默认的 build 工具
  * creating scoped builder instances
  * building against multiple nodes concurrently
  * outputs configuration, inline build caching
  * specifying target platform
* 而的 CLI `Docker Buildx`, 则是在 `BuildKit` 的基础上提供了更多的新特性. 使用 `docker buildx build` 来调用更高级的 build 工具
  * building manifest lists
  * distributed caching
  * exporting build results to OCI image tarballs.


` docker build [OPTIONS] PATH | URL | -`
从一个静态的 dockerfile 和 `context` 来编译一个 docker images, 具体的一个 Context 为一个 Path 或者 URL  
* 具体的 URL 则支持三种类型:
  *  Git repositories
  *  pre-packaged tarball contexts
  *  plain text files




### 2.1.1. build options

* `--file , -f`  		  : Name of the Dockerfile (Default is PATH/Dockerfile), 即指定 Dockerfile 的名称, 这里要和 Path 区分开来

* `--build-arg` 		  : Set build-time variables, 用于根据 dockerfile 的设定动态的调整编译的结果  
* `-t --tag 项目名`   : 用于指定该镜像的名称, 最终镜像的名称为 `name:tag` format



Image 的性能 spec. 配置: 
* `--cpuset-cpus` 		CPUs in which to allow execution (0-3, 0,1)
* `--cpuset-mems` 		MEMs in which to allow execution (0-3, 0,1)
* `--memory , -m` 		Memory limit
* `--memory-swap` 		Swap limit equal to memory plus swap: -1 to enable unlimited swap
* 

## 2.2. docker compose 镜像统合

Define and run multi-container applications with Docker.

`docker compose` 即对容器的统一管理, 属于一个整体性的操作.   
`docker compose` is a tool for managing multi-container Docker applications. It allows you to define and run multiple containers as a single application using a YAML file called `docker-compose.yml`.

如何去定义 `docker-compose.yml` 是使用 compose 命令的关键, 根据定义的具体细则, 有时候可以直接省略掉手动调用 `docker build`

帮助命令 `docker compose --help`


`docker compose [-f <arg>...] [--profile <name>...] [options] [COMMAND] [ARGS...]`
* `-f <arg>...` : 手动指定别的 compose 配置文件.
  * 默认会读取当前目录下的 `docker-compose.yml` 和 `docker-compose.override.yml`
    * 因此若不通过 -f 手动指定配置文件, 则需要保证目录下必须要有 `docker-compose.yml`
    * `docker-compose.override.yml` 默认也会被读取, 会更新替换掉 `docker-compose.yml` 的各种值

`options` compose 全局选项:
* 

`COMMAND` 子命令:
* up      : Create and start containers. 创建并运行 app
* build   : Build or rebuild services



## 2.3. docker image 镜像管理

一个镜像通过 dockerfile 被编译后, 会存储在当前本机中, 即 Host machine.  默认的存储位置是:
* linux : `/var/lib/docker` 
* Windows: `C:\ProgramData\Docker`

`docker image` 和 `docker container` 的使用方法比较类似, 一个是管理系统硬盘上的静态Images, 一个是管理内存中的 动态Container


命令简要:
* `docker image ls`     : 查看本机当前拥有的 image 以及对应的 tag
  * `docker images` 和 `image ls` 功能相同
* `docker image rm [imageName]` 和 `docker rmi [imageName]` 用来删除一个 image

## 2.4. docker container 容器管理

* `docker container ls -l` 列出本机正在运行的容器
* `docker container ls -l --all` 列出本机所有容器, 包括终止运行的容器：
* `docker container rm [containerID]` 删除一个容器
* `docker ps -a` 列出所有容器
* 

## 2.5. docker tag  -  Create a tag TARGET_IMAGE that refers to SOURCE_IMAGE

* `docker tag 旧名字 新名字` 用来给一个 image 赋予新的名字
* 新旧名字会同时存在, 但是都指向同一个 image, ID 是相同的




## 2.6. docker run 启动镜像 核心命令

`docker run [OPTIONS] IMAGE [COMMAND] [ARG...]`   在本机的 docker engine 上运行一个镜像, 此时会创建一个容器
该命令会被用来进行各种各样的容器配置


* `docker run -dp 3000:3000 getting-started`
* 使用 flags 来进行特殊设置
  * `-d` `detached mode` 运行程序, 即后台运行
  * `-p xx:xx ` 端口映射  `host:container`
  * `--name` 如果容器有名字的话, 可以使用名字调用
  * `--runtime==nvidia` 指定 docker 使用GPU
    * `-e NVIDIA_VISIBLE_DEVICES=1` 指定容器只能使用 GPU1


容器行为:
* `--rm`         		      Automatically remove the container when it exits`, 容器退出的时候自动删除, 清洁命令
* `--tty , -t` 		        Allocate a pseudo-TTY, 开启虚拟的远程访问接口 TTY
* `--interactive , -i` 		Keep `STDIN` open even if not attached, 保持 STDIN 的始终开启
* `-it`                   上面两条命令的结合, 会让整个容器变为可交互式的状态
* 

容器路径管理:
* `--volume , -v    <local_path:container_path>` 		Bind mount a volume, 把本地路径挂载到容器的某个路径

spec. 管理:
* `--gpus <all>` 	      GPU devices to add to the container (‘all’ to pass all GPUs)
* `--shm-size=??gb` 		Size of /dev/shm, 共有的内存大小管理

网络管理:
* `--publish , -p  <local_port:container_port>` 		Publish a container’s port(s) to the host


docker 环境
* `--stop-signal` 		    : Signal to stop the container, 更改容器的停止信号


## docker stop  - Stop one or more running containers 停止容器

` docker stop [OPTIONS] CONTAINER [CONTAINER...]`

最为简单的命令, 从外部直接停止一个容器

OPTIONS:
* `--signal , -s`		      : Signal to send to the container, 传送给容器的信息
* `--time , -t`        		: Seconds to wait before killing the container. 在多少秒后停止容器

这里涉及到一个信号的概念:
* 默认下, 执行 docker stop 后, container 会受到一个信号  `SIGTERM`, 然后在特定的宽限期后, 收到 `SIGKILL`, docker 应该就是依据这种信号模式来停止容器的
* 在高级情况下, 可以通过 Dockerfile 的`STOPSIGNAL` 字段 来更改容器的停止信号 
* 同理, 在 `docker run` 部分也有更改容器停止信号的 OPTION `--stop-signal`


## 2.7. docker start - 容器启动(重启) Start one or more stopped containers

容器可以关闭, 再次启动时不能用`run`

使用命令 `docker start [OPTIONS] CONTAINER [CONTAINER...]`

example: `docker start -i <name>-cuda-10.2`

OPTIONS:
* `--attach , -a` 		  : Attach STDOUT/STDERR and forward signals, 启动的同时执行 attach
* `--interactive , -i` 	: Attach container’s STDIN, 交互式的 attach
* `--detach-keys`    		: Override the key sequence for detaching a container, 没太懂, 更改容器的 detach key?, 可能是配合 -a 一起用的

## 2.8. docker attach - 接入容器

Attach local standard input, output, and error streams to a running container

`docker attach [OPTIONS] CONTAINER`

1. 使用 docker 提供的 attach命令 `docker attach 44fc0f0582d9`, 容器是单线程的, 当多个窗口同时用该命令进入容器, 所有的窗口都会同步显示, 有一个窗口阻塞了所有的窗口都不能进行操作
2. 使用 docker 的 exec 命令, 该命令会在容器中执行一个命令, 可以通过调用容器中的bash 来进入容器  `sudo docker exec -it 775c7c9ee1e1 /bin/bash`
    

## 2.9. 进程管理

* `docker ps`         类似于系统的同名命令, 显示所有正在运行的 docker 容器
* `docker stop [id]`  停止一个 docker 容器
* `docker rm <id>`    永久删除一个 docker 容器
  * 可以通过 `rm -f` 来直接停止并删除正在运行的容器




# 3. Docker Hub

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

# 4. Dockerfile

* `Dockerfile` 是一个文本文件脚本, 用于创建一个容器镜像(注意没有文件后缀), 放在项目的根目录

```
FROM node:12-alpine
WORKDIR /app
COPY . .
RUN yarn install --production
CMD ["node", "src/index.js"]
```