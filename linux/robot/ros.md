# 1. Robot Operating System

ROS (Robot Operating System) 是一个开源工具集, 用于 robotics applications.  

帮助各种机器人行业的开发人员进行标准平台下的研究和原型设计以及生产部署.  

ROS 是一款非常复杂的软件, 与计算机操作系统以及系统库非常紧密的嵌合, 因此在官网上更多的是提供了配置好ROS的Linux发行版安装包  


ROS 2 的两个发行版 Humble Hawksbil, Iron Irwini 都只能运行在 Ubuntu 22 上  



# 2. ROS Noetic Ninjemys

因为 ROS2 需要高版本的 Ubuntu, 安装了这个  




# 3. ROS2 Humble Hawksbill 


## 3.1. install

检查 locale 是否支持 UTF-8

```sh
# 添加 ROS 2 GPG key
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
```

## 3.2. Tutorials

ROS 的官方教程

### 3.2.1. CLI tools

ROS 的环境配置, 有些类似于 Python, 通常来说, 一个机器人应用会需要多个子应用来互相配合的工作, 而不同的模组可能会基于不同的 ROS 版本, ROS 本身是基于 shell 来工作的, 可以同时安装不同版本的 ROS 在相同的计算机上, 再通过 sourcing 不同的 setup file, 可以指定不同的 ROS2 版本.

workspace 在 ROS 中是一个专有的概念, 代表了你的系统开发所给予的 ROS 版本, 对于系统中的 core ROS2 版本, 被称为 underlay. 而对应的后续 local workspaces 则被称为 overlays.

在 ROS 系统中, 对应的命令索引并不会自动被添加到 shell 中, 每次启动新的 shell 的时候都必须手动 source 对应的 ROS setup files.  



