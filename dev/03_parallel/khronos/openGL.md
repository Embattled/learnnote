# 1. Open Graphics Library OpenGL

wiki : https://www.khronos.org/opengl/wiki/Main_Page

开放图形库: 用于渲染2D, 3D矢量图形的跨语言, 跨平台的 API
* 由近350个不同的函数调用组成, 绘制从简单的图形比特到复杂的三维景象
* 另一种与 OpenGL 平行的API库 是仅用于Microsoft Windows上的Direct3D
* OpenGL规范描述了绘制2D和3D图形的抽象API, 尽管这些API可以`完全通过软件实现`, 但它是为大部分或者全部使用硬件加速而设计的


OpenGL 是一种协议库, 定义了函数接口, 具体的实现是由 `显示设备厂商` 提供, 一般被包括在驱动程序里
* 语言无关, 平台无关, 规范只字未提获得和管理OpenGL上下文相关的内容, 全部交给底层的窗口系统
* 由 1992年成立的OpenGL架构评审委员会（ARB）维护
* ARB由一些对创建一个统一的, 普遍可用的API特别感兴趣的公司组成
* 由于具体实现是由各家厂商提供, 因此不开源, 而名为 `Mesa` 的项目则是开源的 OpenGL 协议实现

函数的定义表面上类似于C编程语言, 但它们是语言独立的
* 因此基于 OpenGL 标准的协议下, 有各种语言的绑定版本
* JavaScript绑定的WebGL (基于OpenGL ES 2.0)
* C绑定的WGL GLX和CGL
* iOS提供的C绑定
* Android提供的Java和C绑定

大版本日期:
* OpenGL 4.6 API Specification (May 5, 2022) 
* Version 4.5 (Compatibility Profile) - June 29,
2017
* Version 4.0 (Core Profile) - March 11, 2010
* Version 3.1 - May 28, 2009
* Version 3.0 - September 23, 2008
* Version 2.0 - October 22, 2004
 
## 1.1. 程序库

OpenGL 上下文 (OpenGL context) 的创建过程相当复杂, 在不同的操作系统上也需要不同的做法  
* OpenGL纯粹专注于渲染, 而不提供输入、音频以及窗口相关的API
* 所谓 OpenGL 的程序库, 是独立于 OpenGL 标准函数以外, 用以服务开发者创建应用程序的库  
* 因此很多游戏开发和用户界面库都提供了自动创建 OpenGL 上下文的功能
* 也有一些库是专门用来创建 OpenGL 窗口的


* GLUT  : OpenGL Utility Toolkit, 于 1998 年停止更新, 最后的版本是 3.7
  * 早期由OpenGL官方发布的与 OpenGL 配套的库, 目前对已弃用的 OpenGL 特性有依赖
* freeglut : 基于 GLUT 的更新维护版本, 目前仍然在更新
* GLFW  : 目前仍然在更新
* GLEW  : OpenGL Extension Wrangler Library
* GLEE  :
* OpenGL Performer  : 

## 1.2. OpenGL ES OpenGL for Embedded Systems

OpenGL 三维图形 API 的子集, 针对手机, PDA和游戏主机等嵌入式设备而设计, 由 Khronos 集团定义推广
* 从 OpenGL 裁剪的定制而来的, 去除了一些非绝对必要的特性
* 现在主要有两个版本:
  * OpenGL ES 1.x 针对固定管线硬件
  * OpenGL ES 2.x 针对可编程管线硬件
* OpenGL ES Shading Language 3.00 Specification (January 29, 2016)
* OpenGL ES Shading Language 3.20 Specification (July 10, 2019)



## 1.3. Mesa 3D

Mesa 3D是一个在MIT许可证下开放源代码的三维计算机图形库, 以开源形式实现了OpenGL的应用程序接口
* 由于标准的 OpenGL 是由厂商提供的基于各家硬件的, 所以可以说是有硬件加速
* Mesa 是纯软件实现 OpenGL 的各种接口, 因此速度理论上较慢

## Getting Started - 环境配置

要使用基于OpenGL开发的软件或者游戏, 需要 按章正确的设备驱动.   
而如果要进行基于 OpenGL 的开发, 需要
* 安装正确的设备驱动
* 开发套件  development package (depends on platform and programming language).

