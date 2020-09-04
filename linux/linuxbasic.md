# 1. 快速入门

## 1.1. 开始使用  

* 当需要root权限时,需要使用`sudo`执行命令  
    使用 `sudo passwd root ` 给root账户设置密码  

    解锁Root账户 `sudo passwd --unlook root  `

    切换到root管理员 `su root 要输入密码     sudo su直接登录`

    进入高级设置 `sudo raspi-config` (树莓派)


## 1.2. 关机重启命令

shutdown 最为推荐
    命令名称：shutdown。
    英文原意：bring the system down。
    所在路径：/sbin/shutdown。
    执行权限：超级用户。
    功能描述：关机和重启
`# shutdown [选项] 时间 [警告信息]`  
* -c：取消已经执行的 shutdown 命令；
* -h：关机；
* -r：重启；
```shell
shutdown -r now  # 立即重启
shutdown -r +10  # 10分钟后重启
shutdown -r 05:30 # 指定时间重启 (该命令会占用前台终端)
shutdown -r 05:30 & # 指定后把命令放入后端, & 是后台的意思
shutdown -h now #立即关机

shutdown -c # 取消定时任务
```

其他命令
```shell
reboot # 重启

halt #关机
poweroff #关机

# init 是修改 Linux 运行级别的命令，也可以用于关机和重启。 
init 0 # 关机
init 6 # 重启
```

## 1.3. 最基础的命令

| 序号          | 命令                 | 对应英文                | 作用 |
| ------------- | -------------------- | ----------------------- | ---- |
| ls            | list                 | 查看当前文件夹下的内容  |
| pwd           | print work directory | 查看当前所在文件夹      |
| cd[目录名]    | changge directory    | 切换文件夹              |
| touch[文件名] | touch                | 如果文件不存在,新建文件 |
| mkdir[目录名] | make directory       | 创建目录                |
| rm[文件名]    | remove               | 删除指定文件            |
| clear         | clear                | 清屏                    |


    输入 man [内容] 来打开指定内容的手册, f 键前进一页, b 键后退一页, q 键退出
    ctrl + shift + = 放大终端窗口的字体显示
    ctrl + - 缩小终端窗口的字体显示
    如果想要退出选择,并且不想执行当前选中的命令,可以按 ctrl + c


## 1.4. 终端的快捷键

终端有很多快捷键，不太好记  

| 快捷键 | 功能                                                   |
| ------ | ------------------------------------------------------ |
| Ctrl+r | 实现快速检索使用过的历史命令。Ctrl+r中r是retrieve中r。 |
| Ctrl+a | 光标回到命令行首。 （a：ahead）                        |
| Ctrl+e | 光标回到命令行尾。 （e：end）                          |
| ctrl+w | 移除光标前的一个单词                                   |
| Ctrl+k | 删除光标处到行尾的字符。                               |
| Ctrl+u | 删除整个命令行文本字符。                               |
| Ctrl+y | 粘贴Ctrl+u，Ctrl+k，Ctrl+w删除的文本。                 |
| Ctrl+d | 删除提示符后一个字符或exit或logout                     |
| ctrl+t | 交换光标位置前后的两个字符                             |
| ctrl+y | 粘贴或者恢复上次的删                                   |
| Esc+b  | 移动到当前单词的开头                                   |
| Esc+f  | 移动到当前单词的结尾                                   |
| Esc+t  | 颠倒光标所在处及其相邻单词的位置                       |
| esc+.  | 上一个命令的后面的参数                                 |

ESC+c: 使下一个单词首  字母变大写, 同时光标前进一个单词, 如光标停留在单词的某个字母上, 如word中的o字母上, 则o字母变大写. 而不是w   
ESC+u: 使下一个单词所有字母变大写, 同时光标前进一个单词, 同上, 如光标在o字母上, 则ord变大写, w不变.  
ESC+l:同ESC-U, 但使之全变为小写   
## 1.5. 系统查看

### 1.5.1. 查看系统版本

```shell

# 列出所有版本信息
$ lsb_release -a


# 打印发行版信息
$ cat /etc/issue

# 打印linux内核信息
$ cat /proc/version 
$ uname -a
```

# 2. linux文件目录

Linux 基金会的 FHS标准 制定了文件目录的标准  
`（Filesystem Hierarchy Standard）`  

规定了 Linux 系统中**所有**一级目录以及部分二级目录（/usr 和 /var）的用途  

## 2.1. 根目录 系统级文件

```shell
# 存放系统命令,所有用户都可以执行,包括单用户模式  
/bin/

# 系统启动目录, 包括内核文件和启动引导程序 grub
/boot/

# 设备文件  
/dev/ 
# ** 注意 /dev 下的文件是真实的设备 由UDEV在运行时创建
# udev 是Linux kernel 2.6系列的设备管理器。它主要的功能是管理/dev目录底下的设备节点
# /sys/class 是由kernel在运行时导出的，目的是通过文件系统暴露出硬件的层级关系
/sys/class
# 使用示例 查看网络接口的名称
$ ls /sys/class/net


# 配置文件 服务的启动脚本  采用默认安装方式的服务配置文件在此之中 
/etc/

# 主目录
/home/

# *系统* 调用的函数库保存位置 
/lib/

# 虽然系统准备了三个默认挂载目录 /media/、/mnt/、/misc/，但是到底在哪个目录中挂载什么设备可以由管理员自己决定
# 挂载媒体设备, 如软盘和光盘
/media/
# 挂载目录, 如U盘,移动硬盘,*其他操作系统的分区*
/mnt/
# 挂载NFS服务的共享目录  
/misc/

# root用户的主目录,和用户的/home/123/类似
/root/

# 保存与系统环境设置相关的命令
/sbin/

# 服务数据目录,保存服务启动后的数据
/srv/



```
## 2.2. 软件目录 /usr

注意不是 user, 全称为 `Unix Software Resource`  
FHS 建议所有开发者，应把软件产品的数据`合理的放置在 /usr 目录下的各子目录中`  
而不是为他们的产品创建单独的目录  
类似 Windows 系统中` C:\Windows\ + C:\Program files\` 两个目录的综合体

```shell
# 存放系统命令, 除了单用户以外的所有用户可以执行
/usr/bin/

# 同样是根文件系统不需要的系统管理命令,只有root
/usr/sbin/

# 应用程序调用的函数库位置
/usr/lib/

# 图形界面系统保存位置
/usr/XllR6/ 

#  手工安装的软件保存位置。我们一般建议源码包软件安装在这个位置
/usr/local/

# 应用程序的资源文件保存位置，如帮助文档、说明文档和字体目录
/usr/share/

# 我们手工下载的源码包和内核源码包都可以保存到这里
# 不过笔者更习惯把手工下载的源码包保存到 /usr/local/src/ 目录中
# 把内核源码保存到 /usr/src/linux/ 目录中
/usr/src/

# C/C++ 等编程语言头文件的放置目录
/usr/include/

```

## 2.3. /var 目录 

目录用于存储动态数据，例如缓存、日志文件、软件运行过程中产生的文件等  


# 3. 环境配置  

## 3.1. 环境变量

可以使用env和echo命令来查看linux中的所有环境变量  
一个相同的环境变会因为用户身份的不同而具有不同的值  
`env`  
`echo $<环境变量名称>`  


| 环境变量名称 | 作用                                   |
| ------------ | -------------------------------------- |
| HOME         | 用户的主目录（也称家目录）             |
| SHELL        | 用户使用的 Shell 解释器名称            |
| PATH         | 定义命令行解释器搜索用户执行命令的路径 |
| EDITOR       | 用户默认的文本解释器                   |
| RANDOM       | 生成一个随机数字                       |
| LANG         | 系统语言、语系名称                     |
| HISTSIZE     | 输出的历史命令记录条数                 |
| HISTFILESIZE | 保存的历史命令记录条数                 |
| PS1          | Bash解释器的提示符                     |
| MAIL         | 邮件保存路径                           |

**创建一个环境变量**  
直接使用 `<名称>=<数值>`  
`WORKDIR=/home/work1`  

这种环境变量不具有全局性，作用范围也有限，可以使用`export` 将其提升到全局变量  
`export WORKDIR`  
`export WORKDIR=/home/work1`  
这种同样也只适用于当前shell , 关闭后即消失  

## 3.2. 环境信息配置文件

### 3.2.1. 交互式shell和非交互式shell的区别
首先要弄明白什么是交互式shell和非交互式shell , 即 `login shell` 和`non-login shell`

* 交互式模式就是shell等待你的输入，并且执行你提交的命令。这种模式被称作交互式是因为shell与用户进行交互。
* 非交互式模式下，shell不与你进行交互，而是读取存放在文件中的命令,并且执行它们。当它读到文件的结尾，shell也就终止了。
  
系统中存在许多bashrc和profile文件  
* bashrc与profile都用于保存用户的环境信息  
* bashrc用于交互式non-loginshell
* 而profile用于交互式login shell。

### 3.2.2. 文件分布

* `/etc/profile` 
  * 此文件为系统的**每个用户**设置环境信息 当第一个用户登录时,该文件被执行  
  * 并从`/etc/profile.d` 目录的配置文件中搜集shell的设置 
* `/etc/bashrc`
  * 为每一个运行bash shell的用户执行此文件
  * 当bash shell被打开时,该文件被读取
  * 有些linux版本中的/etc目录下已经没有了bashrc文件。
* `~/.profile`
  * 每个用户都可使用该文件输入专用于自己使用的shell信息,当用户登录时,该文件仅仅执行一次!
* `~/.bashrc`
  * 该文件包含专用于某个用户的bash shell的bash信息,当该用户登录时以及每次打开新的shell时,该文件被读取.

**总结**  
当登入系统时候获得一个shell进程时，其读取环境设定档有三步
1. 首先读入的是全局环境变量设定档`/etc/profile`，然后根据其内容读取额外的设定的文档，如`/etc/profile.d`和`/etc/inputrc`
2. 然后根据不同使用者帐号，去其家目录读取`~/.bash_profile`，如果这读取不了就读取`~/.bash_login`，这个也读取不了才会读取`~/.profile` ，这三个文档设定基本上是一样的，读取有优先关系
3. 然后在根据用户帐号读取`~/.bashrc`
   
至于~/.profile与~/.bashrc的不区别
1. 都具有个性化定制功能
2. `~/.profile`可以设定本用户专有的路径，环境变量，等，它只能登入的时候执行一次
3. `~/.bashrc` 也是某用户专有设定文档，可以设定路径，命令别名，每次shell script的执行都会使用它一次

# 4. 查找字符文件

## 4.1. 通配符
要注意通配符与正则表达式的区别  
简单的理解为通配符只有 `*,?,[],{}` 这4种, 而正则表达式复杂多了

| 通配符                | 含义                                        | 实例                                                                               |
| --------------------- | ------------------------------------------- | ---------------------------------------------------------------------------------- |
| *                     | 匹配 0 或多个字符                           | a*b a与b之间可以有任意长度的任意字符, 也可以一个也没有, 如aabcb, axyzb, a012b, ab. |
| ?                     | 匹配任意一个字符                            | a?b a与b之间必须也只能有一个字符, 可以是任意字符, 如aab, abb, acb, a0b.            |
| [list]                | 匹配 list 中的任意单一字符                  | a[xyz]b  a与b之间必须也只能有一个字符, 但只能是 x 或 y 或 z, 如: axb, ayb, azb.    |
| [c1-c2]               | 匹配 c1-c2 中的任意单一字符 如：[0-9] [a-z] | a[0-9]b 0与9之间必须也只能有一个字符 如a0b, a1b... a9b.                            |
| [!list]或[^list]      | 匹配 除list 中的任意单一字符                | a[!0-9]b a与b之间必须也只能有一个字符, 但不能是阿拉伯数字, 如axb, aab, a-b.        |
| [!c1-c2]或[^c1-c2]    | 匹配不在c1-c2的任意字符                     | a[!0-9]b 如acb adb                                                                 |
| {string1,string2,...} | 匹配 sring1 或 string2 (或更多)其一字符串   | a{abc,xyz,123}b 列出aabcb,axyzb,a123b                                              |

## 4.2. 查找文件

    find 命令功能非常强大,通常用来在 特定的目录下 搜索 符合条件的文件  
`find [路径] -name "*.py"`  查找指定路径下扩展名是 .py 的文件,包括子目录

    
    如果省略路径,表示在当前文件夹下查找
    之前学习的通配符,在使用 find 命令时同时可用

## 4.3. 正则表达式

针对文件内容的文本过滤工具里,大都用到正则表达式,如vi,grep,awk,sed等  
其他的一些编程语言,如C++（c regex,c++ regex,boost regex）,java,python等都***有自己的正则表达式库***.

正则表达式的保留字符及意义, 若想使用他们需要在前面加上转义字符 `'\'`
| 字符 | 含义                          | 例                                                                  |
| ---- | ----------------------------- | ------------------------------------------------------------------- |
| ^    | 指向锚定行的开头              | '^grep'匹配所有以grep开头的行                                       |
| $    | 指向锚定行的结尾              | 'grep$'匹配所有以grep结尾的行                                       |
| .    | 任意非换行符的单个字符        | 'gr.p'匹配gr后接一个任意字符，然后是p                               |
| \*   | 匹配零个或多个先前字符        | '\*grep'匹配所有一个或多个空格后紧跟grep的行 `.*`一起用代表任意字符 |
| []   | 字符范围或者特殊匹配。如[a-z] | '[Gg]rep'匹配Grep和grep                                             |
| `\<` | 锚定单词的开始                | '\<grep'匹配包含以grep开头的单词的行                                |
| `\>` | 锚定单词的结束                | 'grep\>'匹配包含以grep结尾的单词的行。                              |

**特殊匹配表**
| 匹配模式   | 含义                                       | 匹配模式  | 含义         |
| ---------- | ------------------------------------------ | --------- | ------------ |
| [:alnum:]  | 字母与数字字符,如grep[[:alnum:]] words.txt | [:alpha:] | 字母         |
| [:ascii:]  | ASCII字符                                  | [:blank:] | 空格或制表符 |
| [:cntrl:]  | ASCII控制字符                              | [:digit:] | 数字         |
| [:graph:]  | 非控制、非空格字符                         | [:lower:] | 小写字母     |
| [:print:]  | 可打印字符                                 | [:punct:] | 标点符号字符 |
| [:space:]  | 空白字符，包括垂直制表符                   | [:upper:] | 大写字母     |
| [:xdigit:] | 十六进制数字                               |

**拓展匹配**

使用 `grep -E` 开启了拓展模式后可以使用更多的控制字符  
控制匹配完成的其他字符可能会遵循正则表达式的规则，对于grep命令，我们还需要在这些字符前面加上\,下表是扩展部分一览
| 选项  | 含义                         |
| ----- | ---------------------------- |
| ?     | 最多一次                     |
| *     | 必须匹配0次或多次            |
| +     | 必须匹配1次或多次            |
| {n}   | 必须匹配n次                  |
| {n,}  | 必须匹配n次或以上            |
| {n,m} | 匹配次数在n到m之间，包括边界 |


### 4.3.1. 字符查找 grep

 `grep `(global search regular expression(RE) and print out the line,全面搜索正则表达式并把行打印出来)  

`grep [OPTION]... PATTERNS [FILE]...`

**例子**   

    查找文件test中出现单词hi，并且若干字符后出现单词Jerry的行
    grep -E "\<hi\>.+\<Jerry\>" test

# 5. Linux 的进程管理
在 Linux 系统中，每个进程都有一个唯一的进程号（PID）  
启动一个进程主要有 2 种途径
* 通过手工启动
* 通过调度启动(事先进行设置，根据用户要求，进程可以自行启动)

## 5.1. 手工启动
指的是由用户输入命令直接启动一个进程, 可以细分为前台启动和后台启动 2 种方式
* 当用户输入一个命令并运行，就已经启动了一个进程，而且是一个前台的进程
  * 假如启动一个比较耗时的进程，可以把该进程挂起(放入后台并暂停运行)
* 后台启动进程，其实就是在命令结尾处添加一个 `&`符号（注意，`&` 前面有空格）
  * 该进程非常耗时，且用户也不急着需要其运行结果的时候
  * 输入命令并运行之后，Shell 会提供给我们一个数字，此数字就是该进程的进程号

## 5.2. ps 命令打印全部进程

在不同的 Linux 发行版上，ps 命令的语法各不相同  
为此，Linux 采取了一个折中的方法，即融合各种不同的风格，兼顾那些已经习惯了其它系统上使用 ps  命令的用户。  

过于复杂 , 几个基础命令如下
```shell
# a  显示一个终端的所有进程
# u  显示进程的归属用户及内存的使用情况
# x  显示没有控制终端的进程
ps aux # 可以查看系统中的所有进程

# -l 长格式显示进程的详细信息
# -e 显示所有进程
ps -le #查看系统中的所有进程, 能看到进程的父进程PID和进程优先级
ps -l  #只能看到当前SHELL产生的进程

```
ps aux 命令的输出含义:  
| 表头    | 含义                      |
| ------- | ------------------------- |
| USER    | 进程是哪个用户常见的      |
| %CPU    | 该进程占用CPU资源的百分比 |
| %MEM    | 同理, 占内存的百分比      |
| VSZ     | 占用虚拟内存的大小(KB)    |
| RSS     | 占用实际内存的大小(KB)    |
| TTY     | 表示该进程运行在哪个终端  |
| STAT    | 进程状态                  |
| START   | 进程的启动时间            |
| TIME    | 进程占用CPU的运算时间     |
| COMMAND | 产生该进程的命令名        |
STAT的状态:
  * -D: 不可被唤醒的睡眠状态, 通常用于 I/O 情况
  * -R: 正在运行
  * -S: 处于睡眠状态, 可以被唤醒
  * -s: 该进程包含紫禁城
  * -T: 停止状态, 后台挂起或者处于除错状态
  * -Z: 僵尸进程, 进程已经终止, 但是部分程序还留在内存中
  * -<: 高优先级
  * -N: 低优先级
  * -L: 被缩入内存
  * -l: 多线程(小写L)
  * -+: 表示该进程正位于后台


ps -le 命令输出信息
| 表头  | 含义                                                        |
| ----- | ----------------------------------------------------------- |
| F     | 进程的权限 1:进程可以被复制,到那时不能被执行 4:超级用户权限 |
| S     | 进程的状态, 同 aux 的 STAT                                  |
| UID   | 运行该进程的用户的IF                                        |
| PPID  | 父进程的ID                                                  |
| C     | CPU使用率(%)                                                |
| PRI   | 优先级,数值越小优先级越高                                   |
| NI    | 优先级, 数值越小优先级越高                                  |
| ADDR  | 进程在内存的哪个位置                                        |
| SZ    | 占用内存的大小                                              |
| WCHAN | 该进程是否正在运行 , `-` 表示正在运行                       |
| TTY   | 指明产生终端                                                |
| TIME  | 占用CPU运算时间                                             |
| CMD   | 同 aux 的 COMMAND                                           |

Linux的终端控制
* 本地终端
  * tty1 ~ tty7 代表本地控制台终端
  * 可以通过 Alt+ F1 ~ F7 快捷键切换不同的终端
  * tty1~tty6 是本地的字符界面终端，tty7 是图形终端
* 虚拟终端 , 一般是远程链接的终端 , 第一个链接占用 pts/0 第二个用 pts/1

## 5.3. top 动态持续监听进程

```shell
# 命令格式
$ top [选项]
```
选项:
* -d:             执行刷新秒数 默认是3秒
* -p 进程的pid:   只监听指定ID的进程
* -u 用户名:      之间听某个用户的进程
* -b:             使用批处理模式输出, 和 -n 选项合用将输出重定向到文件
* -n:             指定执行次数
* -s:             安全模式,避免在交互中出现错误

如果在操作终端执行 top 命令，则并不能看到系统中所有的进程，默认看到的只是 CPU 占比靠前的进程。  
如果我们想要看到所有的进程，则可以把 top 命令的执行结果重定向到文件中。  
不过 top 命令是持续运行的，这时就需要使用 "-b" 和 "-n" 选项了。  
` top -b -n 1 > /root/top.log`  

进入top命令后, 会进入交互式界面, 交互操作:  
| 操作   | 功能                                                             |
| ------ | ---------------------------------------------------------------- |
| ?或者h | 显示帮助                                                         |
| P      | 按照CPU使用率排序                                                |
| M      | 按照内存使用率排序                                               |
| N      | 按照PID排序                                                      |
| T      | 按照 TIME 即CPU累计运算时间排序                                  |
| k      | 按照pid给予某一个进程一个信号, 一般用于中止进程, 信号9是强制中止 |
| r      | 按照pid给与某个进程重设优先级 (NIce)值                           |
| q      | 退出                                                             |

重要的输出信息:
* load average : 系统在之前 1 分钟、5 分钟、15 分钟的平均负载, 一般认为不应该超过服务器 CPU 的核数
* id : 空闲 CPU 占用的 CPU 百分比
* 证明系统处于高负债的情况
  * 如果 1 分钟、5 分钟、15 分钟的平均负载高于 1
  * 如果 CPU 的使用率过高或空闲率过低
  * 物理内存的空闲内存过小
缓存（cache）是用来加速数据从硬盘中"读取"的，而缓冲（buffer）是用来加速数据"写入"硬盘的。  

同理 , top的输出表如下
| 表头    | 信息                                    |
| ------- | --------------------------------------- |
| PID     | 进程的 ID。                             |
| USER    | 该进程所属的用户。                      |
| PR      | 优先级，数值越小优先级越高。            |
| NI      | 优先级，数值越小、优先级越高。          |
| VIRT    | 该进程使用的虚拟内存的大小，单位为 KB。 |
| `RES`   | 该进程使用的物理内存的大小，单位为 KB。 |
| `SHR`   | 共享内存大小，单位为 KB。               |
| S       | 进程状态。                              |
| %CPU    | 该进程占用 CPU 的百分比。               |
| %MEM    | 该进程占用内存的百分比。                |
| TIME+   | 该进程共占用的 CPU 时间。               |
| COMMAND | 进程的命令名。                          |

## 5.4. pstree  查看进程树

pstree 命令是以树形结构显示程序和进程之间的关系  

`# pstree [选项] [PID或用户名]`  
| 选项 | 含义                                                         |
| ---- | ------------------------------------------------------------ |
| -a   | 显示启动每个进程对应的完整指令，包括启动进程的路径、参数等。 |
| -c   | 不使用精简法显示进程信息，即显示的进程中包含子进程和父进程。 |
| -n   | 根据进程 PID 号来排序输出，默认是以程序名排序输出的。        |
| -p   | 显示进程的 PID。                                             |
| -u   | 显示进程对应的用户名称。                                     |


如果不指定进程的 PID 号，也不指定用户名称，则会以 init 进程为根进程，显示系统中所有程序和进程的信息  
`init` 进程是系统启动的第一个进程，进程的 PID 是 1，也是系统中所有进程的父进程。  

如果想知道某个用户都启动了哪些进程，使用 pstree 命令可以很容易实现，以 mysql 用户为例  
`# pstree mysql`  

##  lsof 列出进程正在调用或者打开的文件
`list opened files`的缩写   




# 6. Linux 的服务管理

Linux 服务管理两种方式service和systemctl 

systemd是Linux系统**最新的初始化系统**(init),作用是提高系统的启动速度，尽可能启动较少的进程，尽可能更多进程并发启动。

systemd对应的进程管理命令就是 `systemctl`

## 6.1. service  

service命令其实是去`/etc/init.d`目录下，去执行相关程序, 已经被淘汰

```shell
# service命令启动redis脚本
service redis start
# 直接启动redis脚本
/etc/init.d/redis start
# 开机自启动
update-rc.d redis defaults
```

## 6.2. systemctl

### 6.2.1. 概念

systemctl命令兼容了service  

即systemctl也会去`/etc/init.d`目录下，查看，执行相关程序  

```shell
# 开机自启动
systemctl enable redis
```

通过 `Unit` 作为单位管理进程

`/usr/lib/systemd/system(Centos)`  
`/etc/systemd/system(Ubuntu)`  

systemd 默认读取 `/etc/systemd/system `下的配置文件，该目录下的文件会链接/lib/systemd/system/下的文件。执行 ls /lib/systemd/system 你可以看到有很多启动脚本，其中就有最初用来定义开机脚本的 `rc.local.service`

主要有四种类型文件.mount,.service,.target,.wants  
代表四种`Unit`  

### 6.2.2. Unit操作命令

`systemctl –-version`  查看版本  

`systemctl [command] [unit]` 命令格式  

### 6.2.3. a.command综述

| 命令      | 功能简述                                                                   |
| --------- | -------------------------------------------------------------------------- |
| start     | 立刻启动后面接的 unit。                                                    |
| stop      | 立刻关闭后面接的 unit。                                                    |
| restart   | 立刻关闭后启动后面接的 unit，亦即执行 stop 再 start 的意思。               |
| reload    | 不关闭 unit 的情况下，重新载入配置文件，让设置生效。                       |
| enable    | 设置下次开机时，后面接的 unit 会被启动。                                   |
| disable   | 设置下次开机时，后面接的 unit 不会被启动。                                 |
| show      | 列出 unit 的配置。                                                         |
| status    | 目前后面接的这个 unit 的状态，会列出有没有正在执行、开机时是否启动等信息。 |
| is-active | 目前有没有正在运行中。                                                     |
| is-enable | 开机时有没有默认要启用这个 unit。                                          |
| kill      | 不要被 kill 这个名字吓着了，它其实是向运行 unit 的进程发送信号。           |
| mask      | 注销 unit，注销后你就无法启动这个 unit 了。                                |
| unmask    | 取消对 unit 的注销。                                                       |

### 6.2.4. b.status 

* 第一行是对 unit 的基本描述。  
* 第二行中的 Loaded 描述操作系统启动时会不会启动这个服务，enabled 表示开机时启动，disabled 表示开机时不启动。  
  * 关于 unit 的启动状态，除了 enable 和 disable 之外还有:  
  *   static:这个 unit 不可以自己启动，不过可能会被其它的 enabled 的服务来唤醒。
  * mask:这个 unit 无论如何都无法被启动！因为已经被强制注销。可通过 systemctl unmask 改回原来的状态。
* 第三行 中的 Active 描述服务当前的状态，active (running) 表示服务正在运行中。如果是 inactive (dead) 则表示服务当前没有运行。 
  * active (exited)：仅执行一次就正常结束的服务，目前并没有任何程序在系统中执行。
  * active (waiting)：正在执行当中，不过还再等待其他的事件才能继续处理。 
* 第四行的 Docs 提供了在线文档的地址。  


## 6.3. 3.综合查看命令

systemctl 提供了子命令可以查看系统上的 unit，命令格式为:   
`systemctl [command] [--type=TYPE] [--all]` 

不带任何参数执行 systemctl 命令会列出所有已启动的 unit  
如果添加 -all 选项会同时列出没有启动的 unit。    

**command**  
不带任何参数执行 systemctl 命令会列出所有已启动的 unit  
systemctl list-units: 列出当前已经启动的 unit  
systemctl list-unit-files: 根据 /lib/systemd/system/ 目录内的文件列出所有的 unit, 即列出所有以安装的服务    

systemd-cgls   以树形列出正在运行的进程，它可以递归显示控制组内容

**--type=TYPE**  

可以过滤某个类型的 unit。  
` systemctl list-units --type=service `  

### 6.3.1. 操作环境管理
通过指定 `--type=target` 就可以用 `systemctl list-units` 命令查看系统中默认有多少种 target

| 操作环境          | 功能                                                                                                                                   |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| graphical.target  | 就是文字界面再加上图形界面，这个 target 已经包含了下面的 multi-user.target。                                                           |
| multi-user.target | 纯文本模式！                                                                                                                           |
| rescue.target     | 在无法使用 root 登陆的情况下，systemd 在开机时会多加一个额外的临时系统，与你原本的系统无关。这时你可以取得 root 的权限来维护你的系统。 |
| emergency.target  | 紧急处理系统的错误，在无法使用 rescue.target 时，可以尝试使用这种模式！                                                                |
| shutdown.target   | 就是执行关机。                                                                                                                         |
| getty.target      | 可以设置 tty 的配置。                                                                                                                  |

正常的模式是` multi-user.target `和 `graphical.target `两个，救援方面的模式主要是 `rescue.target` 以及更严重的 `emergency.target`。如果要修改可提供登陆的 tty 数量，则修改 getty.target。


| 命令        | 功能                                     |
| ----------- | ---------------------------------------- |
| get-default | 取得目前的 target。                      |
| set-default | 设置后面接的 target 成为默认的操作模式。 |
| isolate     | 切换到后面接的模式。                     |

我们还可以在不重新启动的情况下切换不同的 target，比如从图形界面切换到纯文本的模式：
`systemctl isolate multi-user.target`

## 6.4. Unit的编写与设置

## 6.5. Unit的基本概念
一般都会有  
Unit小节: 描述,启动时间与条件等等  

### 6.5.1. .service

.service文件定义了一个服务，分为[Unit]，[Service]，[Install]三个小节

`[Unit]` 段: 启动顺序与依赖关系   
`[Service] `段: 启动行为,如何启动，启动类型  
`[Install]` 段: 定义如何安装这个配置文件，即怎样做到开机启动  
### 6.5.2. .mounnt

.mount文件定义了一个挂载点，[Mount]节点里配置了What(名称),Where(位置),Type(类型)三个数据项

```
What=hugetlbfs
Where=/dev/hugepages
Type=hugetlbfs
```
等于执行以下命令  
`mount -t hugetlbfs /dev/hugepages hugetlbfs`  

### 6.5.3. .target

.target定义了一些基础的组件，供.service文件调用

### 6.5.4. .wants文件

`.wants`文件定义了要执行的文件集合，每次执行，`.wants`文件夹里面的文件都会执行  

## 6.6. 2.编写开机启动rc.local

查看`/lib/systemd/system/rc.local.server`,默认会缺少`Install`段,显然这样配置是无效的  

```shell
# rc.local.server的Unit段, 可以看到会执行 /etc/rc.local
[Unit]
Description=/etc/rc.local Compatibility
Documentation=man:systemd-rc-local-generator(8)
ConditionFileIsExecutable=/etc/rc.local
After=network.target


#修改文件进行配置  
[Install]  
WantedBy=multi-user.target  


#之后再在/etc/目录下面创建rc.local文件，赋予执行权限

$ touch /etc/rc.local
$ chmod +x /etc/rc.local

# 在rc.local里面写入
# 注意：'#!/bin/sh' 这一行一定要加

#!/bin/sh
exit 0

# 最后将/lib/systemd/system/rc.local.service 链接到/etc/systemd/system目录

$ ln -s /lib/systemd/system/rc.local.service /etc/systemd/system/
```




# 7. Shell基础  

## 7.1. source 命令 

source命令也称为“点命令”，也就是一个点符号（.）,是bash的内部命令。  
功能：使Shell读入指定的Shell程序文件并依次执行文件中的所有语句  
以下两种都是正确的使用  
`source filename `  
`. filename`（中间有空格）  

几种类似命令的区别  
| 命令            | 功能于区别                        |
| --------------- | --------------------------------- |
| source filename | 读取脚本在**当前**shell里执行     |
| sh filename     | 新建一个子shell，继承当前环境变量 |
| .filename       | 只是执行当前目录下的脚本          |



## 7.2. 开机自动脚本

* Ubuntu18.04 默认是没有 /etc/rc.local 这个文件的，需要自己创建  
* systemd 默认读取 `/etc/systemd/system `下的配置文件，该目录下的文件会`链接/lib/systemd/system/`下的文件。执行 `ls /lib/systemd/system `你可以看到有很多启动脚本，其中就有我们需要的 rc.local.service  
* 查看rc.local.service文件内容

```shell
# This unit gets pulled automatically into multi-user.target by
# systemd-rc-local-generator if /etc/rc.local is executable.
[Unit]
Description=/etc/rc.local Compatibility
Documentation=man:systemd-rc-local-generator(8)
ConditionFileIsExecutable=/etc/rc.local
After=network.target

[Service]
Type=forking
ExecStart=/etc/rc.local start
TimeoutSec=0
RemainAfterExit=yes
GuessMainPID=no
```

剩下的内容在service.md里写了  