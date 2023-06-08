# Program Frameworks

用于编写程序的框架库, 会很大程度上直接决定整个程序的构造, 尽管如此, 目前 3.11 版本里记载的三个库都是用于开发 CLI 程序的

GUI 的开发记录在 Tcl/Tk 分类中

* turtle — Turtle graphics
* cmd — Support for line-oriented command interpreters
* shlex — Simple lexical analysis


# turtle - Turtle graphics

Turtle graphics is a popular way for introducing programming to kids. ??

一种源于 1967 年的编程语言, developed by Wally Feurzeig, Seymour Papert and Cynthia Solomon
* 想象一个 robotic turtle 在 x-y 平面的起点 (0,0)
* 类似于游戏一样, `turtle.forward(15)` 可以操纵海龟往前走
* `turtle.right(25)` 可以令海龟 顺时针 转向 25度
* 通过一些基础的简单指令, 可以实现复杂的图像


python 的 turtle 模组是 turtle 语言的重实现, 完整兼容
* 整个 模组是建立在面向对象的基础上, 由 2+2 个对象管理
* 