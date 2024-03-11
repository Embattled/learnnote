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

## 1.1. Basic concepts - 基本概念


基本概念, tmux 是基于 pseudo terminals 的
* tmux 的 session 是持久的, 无论是 ssh 意外断开, 还是 有意的 detach 都可以在之后重连
* 启动 tmux 的时候会自动建立一个 session, 并且建立一个 window 显示在 screen 上
* 每一个 session 都可以拥有多个 window
* 每一个 window 可以单独显示在整面屏幕, 也可以分割成矩形的多个 panes

## 1.2. Summary of terms - 名词总结




# 2. Default Key bindings - 默认的快捷键绑定

tmux 除了通过命令以外, 还可以直接通过 快捷键绑定来操作, 默认下 C-b (Ctrl-b) 是执行快捷键的前缀, 然后再按以下的按键实现功能  

Alt 在文档中称为 Meta  (M-)


The default command key bindings are:

```
C-b
    Send the prefix key (C-b) through to the application.
C-z
    Suspend the tmux client.
#
    List all paste buffers.
'
    Prompt for a window index to select.
-
    Delete the most recently copied buffer of text.
.
    Prompt for an index to move the current window.
:
    Enter the tmux command prompt.
=
    Choose which buffer to paste interactively from a list.
D
    Choose a client to detach.
L
    Switch the attached client back to the last session.
]
    Paste the most recently copied buffer of text.
f
    Prompt to search for text in open windows.
r
    Force redraw of the attached client.
M-1 to M-5
    Arrange panes in one of the five preset layouts: even-horizontal, even-vertical, main-horizontal, main-vertical, or tiled.
Space
    Arrange the current window in the next preset layout.
M-n
    Move to the next window with a bell or activity marker.
M-p
    Move to the previous window with a bell or activity marker.
```


| session 操作 | 效果                                              |
| ------------ | ------------------------------------------------- |
| `$`          | 重命名当前 session                                |
| `(`  `)`     | 切换到 前/后 一个 session                         |
| `d`          | Detach the current client.                        |
| `s`          | 进入到交互式选择界面, 只不过直接就是 session 界面 |

| window 操作 | 效果                                             |
| ----------- | ------------------------------------------------ |
| 0 to 9      | 切换 window                                      |
| `w`         | 进入到交互式选择界面, 只不过直接就是 window 界面 |
| `i`         | 在底部状态条显示当前 window 的详细信息           |
| `n`         | 切换到 next     下一个 window                    |
| `p`         | 切换到 previous 上一个 window                    |
| `l`         | 切换到 last 刚才的 window                        |
| `c`         | 创建一个新的 window                              |
| `&`         | 停止当前 window                                  |
| `,`         | 重命名当前 window                                |


| pane 操作       | 效果                                                                          |
| --------------- | ----------------------------------------------------------------------------- |
| 方向键上下左右  | 调整光标到 对应方向的 Pane 上                                                 |
| `;`             | 移动回到上一个活动的 pane                                                     |
| `o`             | 移动到下一个 index 的 pane                                                    |
| `q`             | 简单的展示 pane indexes, Briefly display pane indexes. 在每个 pane 上展示数字 |
| `"`             | 将当前 Pane 上下切分                                                          |
| `%`             | 将当前 Pane 左右切分                                                          |
| `!`             | 将当前 Pane 从当前 window 中切出去, 作为一个新的 window                       |
| `z`             | 暂时性的聚焦到当前 Pane , 执行任何 pane 切分, 切换都会退出 toggle zoom 模式   |
| `C-o`           | rotate the panes in the current window forwards, 前向轮转 pane index          |
| `M-o`           | rotate the panes in the current window backwards, 向后轮转 pane index         |
| `{`  `}`        | 将当前的 Pane 与 index 的 前一个 / 后一个进行交换                             |
| `m`             | mark 当前 Pane                                                                |
| `M`             | 清除所有 pane 的 mark 标记                                                    |
| `x`             | 停止当前 Pane, 同 bash exit 一样                                              |
| `Ctrl + 方向键` | 按照方向键调整当前 Pane 的大小, 以 1 cell 为单位                              |
| `Alt + 方向键`  | 按照方向键调整当前 Pane 的大小, 以 5 cell 为单位                              |


| tmux 系统 | 效果                                              |
| --------- | ------------------------------------------------- |
| `?`       | List all key bindings, 进入快捷键绑定界面.        |
| `d`       | Detach the current client, 断联当前的 tmux 客户端 |

| 辅助功能  | 效果                                                               |
| --------- | ------------------------------------------------------------------ |
| `t`       | 在当前 pane 显示时间                                               |
| `[`       | 进入 copy mode                                                     |
| `Page Up` | 进入 copy mode 的同时 向上翻一页                                   |
| `~`       | Show previous messages from tmux, if any. 显示 tmux 的历史工作消息 |


# 3. Clients and Sessions - 客户端和连接



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
    

### 3.0.1. 会话分离与接入

1. `detach` 用于分离当前会话 (detach-client)
   * 输入后会立刻退出当前 tmux 窗口, 但是会话仍会在后台保持
   * 使用 `Ctrl+b d` 就可以快捷键的将当前会话与窗口分离

2. `attach` 接入一个会话 (attach-session)
   * `attach-session [-dErx] [-c working-directory] [-t target-session]`
   * `-d` 用于排他的接入, 接入的同时会切断其他客户端与该 session 的链接

3. `switch` 切换 session 
   * `switch-client [-Elnpr] [-c target-client] [-t target-session] [-T key-table]`
   * `tmux switch -t <id or name>` 直接切换到另一个会话


# 4. Windows and Panes

每一个 tmux 的 window 都可以细分成多个 panes   

pane : 一个矩形的区域, 显示对应 terminal 的内容

使用窗格(pane)功能即可在同一个 window 中同时运行多个命令  

### 4.0.1. pane 管理

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