# Tmux

* 一个终端复用器 terminal multiplexer  
* 类似的还有 GNU Screen, Tmux与之功能类似, 但是普遍认为 Tmux 更强大

## 会话与进程

* 命令行的典型使用方式:
  * 打开一个终端窗口(window), 输入命令进行交互. 这种临时的交互被称为 会话(session)
  * 会话的特点: 窗口与其中启动的进程是连接在一起的, 窗口关闭会话也会结束, 会话内部的进程也会随之中止
* 会话与窗口解绑
  * 窗口关闭时会话并不中止
  * 等再需要的时候, 让会话重新绑定其他的窗口
  * Tmux 就是用来解绑会话与窗口的工具

## 基本用法

* 基本操作
  * 使用 `tmux` 即可进入 Tmux 窗口 `Ctrl+d` 或者输入 `exit` 推出 Tmux
  * 前缀键 : tmux 窗口的快捷键需要使用前缀键激活 `Ctrl+b`, 激活后才能使用
* 会话管理
  * 输入 tmux 后启动的窗口是编号 0, 第二个窗口是编号 1
  * 这些窗口对应的会话就是 0, 1 会话
    * 使用 `tmux new` 新建会话
    * `tmux new -s <name>` 用来给新建的会话起名字, 有名字会更直观
    * `tmux rename-session <name>` 给当前会话重命名
    * `Ctrl+b $` 快捷键重命名
    * `tmux rename-session -t 0 <name>` -t 命令用于指定重命名的会话
  * 会话分离
    * 使用 `Ctrl+b d` 或者直接输入 `tmux detach` 就可以将当前会话与窗口分离
    * 分离后就会退出当前 tmux 窗口, 但是会话仍在保持
  * 会话接入
    * `tmux ls` 和 `Ctrl+b s`可以查看当前所有的 tmux 会话
    * `tmux attach` 可以重新接入某个已存在的会话
    * `tmux switch -t <id or name>` 直接切换到另一个会话
  * 会话结束
    * `tmux kill-session -t <id or name>` 结束一个会话 

## 窗格操作

使用窗格(pane)功能即可在终端中同时运行多个命令  

* 窗格划分
  * `tmux split-window` 默认上下划分
  * `tmux split-window -h` 划分左右两个窗格
* 光标移动
  * `tmux select-pane -UDLR`
  * UDLR 代表上下左右
  * 光标移动更多的使用快捷键
* 窗格位置移动
  * `tmux swap-pane -UDLR`
  
## 窗口操作

tmux 允许新建多个窗口  