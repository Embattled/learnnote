# 1. tmux

* 一个终端复用器 terminal multiplexer  
* 类似的还有 GNU Screen, Tmux与之功能类似, 但是普遍认为 Tmux 更强大

* 命令行的典型使用方式:
  * 打开一个终端窗口(window), 输入命令进行交互. 这种临时的交互被称为 会话(session)
  * 会话的特点: 窗口与其中启动的进程是连接在一起的, 窗口关闭会话也会结束, 会话内部的进程也会随之中止
* 会话与窗口解绑
  * 窗口关闭时会话并不中止
  * 等再需要的时候, 让会话重新绑定其他的窗口
  * Tmux 就是用来解绑会话与窗口的工具

* session     : 与 tmux 服务器接入的链接单位
* windows     : attach 到一个 session, 产生一个 节目
* pane        : 由一个 windows 分割而来


```sh
tmux 	[-2CDlNuVv] [-c shell-command] [-f file] [-L socket-name] [-S socket-path] [-T features] [command [flags]]
```

基本概念, tmux 是基于 pseudo terminals 的
* tmux 的 session 是持久的, 无论是 ssh 意外断开, 还是 有意的 detach 都可以在之后重连
* 启动 tmux 的时候会自动建立一个 session, 并且建立一个 window 显示在 screen 上
* 每一个 session 都可以拥有多个 window
* 每一个 window 可以单独显示在整面屏幕, 也可以分割成矩形的多个 panes



## 1.1. 基本用法

* 基本操作
  * 使用 `tmux` 即可进入 Tmux 窗口 `Ctrl+d` 或者输入 `exit` 推出 Tmux
  * 前缀键 : tmux 窗口的快捷键需要使用前缀键激活 `Ctrl+b`, 激活后才能使用
* 查看当前所有会话
  * `tmux ls` 和 `Ctrl+b s`可以查看当前所有的 tmux 会话




# Clients and Sessions - 客户端和连接



* 输入 tmux 后启动的窗口是编号 0, 第二个窗口是编号 1
* 这些窗口对应的会话就是 0, 1 会话

0. `ls` 命令 (list-sessions) 用于显示当前已经建立的所有会话
   * `lsc` (list-clients) 显示连接到 tmux server 的所有 clients

1. `new` 命令 新建会话 (new-session)
   * `tmux new -s <name>` 用来给新建的会话起名字, 有名字会更直观
   * `tmux new -n <name>` 给新建会话的默认窗口起名字

2. `rename` 命令, 重命名一个已有的 session
   * `tmux rename-session <name>` 给当前会话重命名
   * `tmux rename-session -t 0 <name>` -t 命令用于指定重命名的会话

3. `kill-session` 命令, 结束一个会话
   * `kill-session [-aC] [-t target-session]`
   * `tmux kill-session -t <id or name>` 结束一个指定的会话会话
   * `tmux kill-session -a -t <id or name>` 结束除了指定会话以外的所有会话  
    

### 1.2.2. 会话分离与接入

1. `detach` 用于分离当前会话 (detach-client)
   * 输入后会立刻退出当前 tmux 窗口, 但是会话仍会在后台保持
   * 使用 `Ctrl+b d` 就可以快捷键的将当前会话与窗口分离

2. `attach` 接入一个会话 (attach-session)
   * `attach-session [-dErx] [-c working-directory] [-t target-session]`
   * `-d` 用于排他的接入, 接入的同时会切断其他客户端与该 session 的链接

3. `switch` 切换 session 
   * `switch-client [-Elnpr] [-c target-client] [-t target-session] [-T key-table]`
   * `tmux switch -t <id or name>` 直接切换到另一个会话


# Windows and Panes

每一个 tmux 的 window 都可以细分成多个 panes   

pane : 一个矩形的区域, 显示对应 terminal 的内容

使用窗格(pane)功能即可在同一个 window 中同时运行多个命令  

### 1.4.1. pane 管理

1. `splitw` 窗格划分 (split-windows)
   * `split-window [-bdfhIvP] [-c start-directory] [-e environment] [-l size | -p percentage] [-t target-pane] [shell-command] [-F format]`
   * `tmux split-window`  `[-bdfhIvP]` 用于指定划分的方法, 默认上下划分(-v)
     * `-h` 划分左右两个窗格
     * `-f` 用于和 v/h 配合, 使得新分割出来的 pane 填充整个屏幕的宽或高, 而不是沿用当前光标所在的 pane
2. `selectp` 光标在 pane 间移动 (select-pane)
  * `select-pane [-DdeLlMmRU] [-T title] [-t target-pane]`
    * UDLR 代表移动的上下左右
  * 光标移动更多的使用快捷键

3. `swapp` 窗格位置移动 (swap-pane)
  * `select-pane [-DdeLlMmRU] [-T title] [-t target-pane]`
  * `tmux swap-pane -UDLR`