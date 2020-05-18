# 1. 快速入门

## 1. 开始使用  

* 当需要root权限时,需要使用`sudo`执行命令  
    使用 `sudo passwd root ` 给root账户设置密码  

    解锁Root账户 `sudo passwd --unlook root  `

    切换到root管理员 `su root 要输入密码     sudo su直接登录`

    进入高级设置 `sudo raspi-config`



## 2. 最基础的命令

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

# 2. linux文件目录

Linux 基金会的 FHS标准 制定了文件目录的标准  
`（Filesystem Hierarchy Standard）`  

规定了 Linux 系统中**所有**一级目录以及部分二级目录（/usr 和 /var）的用途  

## 1. 根目录 系统级文件

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
## 2. 软件目录 /usr

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

## 3. /var 目录 

目录用于存储动态数据，例如缓存、日志文件、软件运行过程中产生的文件等  


# 3. 环境配置  

## 1. 环境变量

可以使用env和echo命令来查看linux中的所有环境变量  
一个相同的环境变会因为用户身份的不同而具有不同的值  
`env`  
`echo $<环境变量名称>`  





环境变量名称|作用
-|-
HOME|用户的主目录（也称家目录）
SHELL |用户使用的 Shell 解释器名称
PATH|定义命令行解释器搜索用户执行命令的路径
EDITOR|用户默认的文本解释器
RANDOM|生成一个随机数字
LANG|系统语言、语系名称
HISTSIZE|输出的历史命令记录条数
HISTFILESIZE|保存的历史命令记录条数
PS1|Bash解释器的提示符
MAIL|邮件保存路径

**创建一个环境变量**  
直接使用 `<名称>=<数值>`  
`WORKDIR=/home/work1`  

这种环境变量不具有全局性，作用范围也有限，可以使用`export` 将其提升到全局变量  
`export WORKDIR`  






# 4. 查找字符文件

## 通配符
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

## 查找文件

    find 命令功能非常强大,通常用来在 特定的目录下 搜索 符合条件的文件  
`find [路径] -name "*.py"`  查找指定路径下扩展名是 .py 的文件,包括子目录

    
    如果省略路径,表示在当前文件夹下查找
    之前学习的通配符,在使用 find 命令时同时可用

## 正则表达式

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


### 字符查找 grep

 `grep `(global search regular expression(RE) and print out the line,全面搜索正则表达式并把行打印出来)  

`grep [OPTION]... PATTERNS [FILE]...`

**例子**   

    查找文件test中出现单词hi，并且若干字符后出现单词Jerry的行
    grep -E "\<hi\>.+\<Jerry\>" test





# 5. Shell基础  

## 1. source 命令 

source命令也称为“点命令”，也就是一个点符号（.）,是bash的内部命令。  
功能：使Shell读入指定的Shell程序文件并依次执行文件中的所有语句  
以下两种都是正确的使用  
`source filename `  
`. filename`（中间有空格）  

几种类似命令的区别  
命令|功能于区别
-|-
source filename|读取脚本在**当前**shell里执行
sh filename|新建一个子shell，继承当前环境变量
.filename|只是执行当前目录下的脚本



## 2. 开机自动脚本

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