# 1. Node.js

* Node.js 就是运行在服务端的 JavaScript
* 基于Chrome JavaScript 运行时建立的一个平台
* 事件驱动I/O服务端JavaScript环境，基于Google的V8引擎


## 1.1. Node.js 的应用构成

1. 使用 require 指令来载入 Node.js 模块
2. 创建服务器：服务器可以监听客户端的请求，类似于 Apache 、Nginx 等 HTTP 服务器。
3. 接收请求与响应请求 服务器很容易创建，客户端可以使用浏览器或终端发送 HTTP 请求，服务器接收请求后返回响应数据

# 2. 包管理器

## 2.1. npm

NPM是随同NodeJS一起安装的包管理工具，能解决NodeJS代码部署上的很多问题，常见的使用场景有以下几种：

* 允许用户从NPM服务器下载别人编写的第三方包到本地使用。
* 允许用户从NPM服务器下载并安装别人编写的命令行程序到本地使用。
* 允许用户将自己编写的包或命令行程序上传到NPM服务器供别人使用。

### 2.1.1. 安装 卸载

安装模块   `npm install <Module Name>`   
搜索模块   `npm search  <Module Name>`  


* 安装使用 `install` 指令
* 包的安装可以分为 本地安装(local) 和 全局安装 (global)
  * 默认为本地安装              
    * 会把包安装在 npm 执行目录下, 也就是当前的 workspace 目录
    * 可以通过 `require()` 来引入本地安装的包
  * 全局安装使用参数命令 `-g`
    * 将安装包放在 `/usr/local` 下或者你 node 的安装目录
    * 可以直接在命令行里使用
  * 二者的调用方法不同, 如果想要都使用, 则需要在两个地方都安装或者使用 `npm link`


* 卸载使用 `uninstall` 命令


### 2.1.2. 升级

npm 可以通过自己的命令来升级自己, 但是算是安装命令  
`sudo npm install npm -g`  

一般的包使用 `update` 命令来升级  
`npm update 模块名称`

### 2.1.3. 列表

* 使用 `list` 来打印已安装的模块
  * 默认打印本地安装
  * 使用 `-g` 打印全局模块
* 使用 `list 模块名` 查看模块的版本


## 2.2. yarn