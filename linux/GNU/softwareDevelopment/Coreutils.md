- [1. GNU Coreutils](#1-gnu-coreutils)
- [2. Common options](#2-common-options)
- [3. Output of entire files](#3-output-of-entire-files)
- [4. Formatting file contents](#4-formatting-file-contents)
- [5. Output of parts of files](#5-output-of-parts-of-files)
- [6. Summarizing files](#6-summarizing-files)
- [7. Operating on sorted files](#7-operating-on-sorted-files)
  - [7.1. sort: Sort text files](#71-sort-sort-text-files)
- [8. Operating on fields](#8-operating-on-fields)
- [9. Operating on characters](#9-operating-on-characters)
- [10. Directory listing](#10-directory-listing)
- [11. Basic operations](#11-basic-operations)
- [12. Special file types](#12-special-file-types)
  - [12.1. link - Make a hard link via the link syscall](#121-link---make-a-hard-link-via-the-link-syscall)
  - [12.2. ln - Make links between files](#122-ln---make-links-between-files)
  - [12.3. mkdir - Make directories](#123-mkdir---make-directories)
- [13. Changing file attributes](#13-changing-file-attributes)
- [14. File space usage](#14-file-space-usage)
  - [14.1. df: Report file system space usage](#141-df-report-file-system-space-usage)
  - [14.2. du: Estimate file space usage](#142-du-estimate-file-space-usage)
  - [14.3. sync: Synchronize cached writes to persistent storage](#143-sync-synchronize-cached-writes-to-persistent-storage)
- [15. Printing text](#15-printing-text)
  - [15.1. echo: Print a line of text](#151-echo-print-a-line-of-text)
- [16. Conditions](#16-conditions)
- [17. Redirection](#17-redirection)
- [18. File name manipulation](#18-file-name-manipulation)
- [19. Working context](#19-working-context)
  - [19.1. pwd: Print working directory](#191-pwd-print-working-directory)
  - [19.2. stty: Print or change terminal characteristics](#192-stty-print-or-change-terminal-characteristics)
  - [19.3. printenv: Print all or some environment variables](#193-printenv-print-all-or-some-environment-variables)
  - [19.4. tty: Print file name of terminal on standard input](#194-tty-print-file-name-of-terminal-on-standard-input)
- [20. User information](#20-user-information)
  - [20.1. id: Print user identity](#201-id-print-user-identity)
  - [20.2. logname : Print current login name](#202-logname--print-current-login-name)
  - [20.3. whoami : Print effective user name](#203-whoami--print-effective-user-name)
  - [20.4. groups](#204-groups)
  - [20.5. users : Print login names of users currently logged in](#205-users--print-login-names-of-users-currently-logged-in)
  - [20.6. who : Print who is currently logged in](#206-who--print-who-is-currently-logged-in)
- [21. System context](#21-system-context)
  - [21.1. date: Print or set system date and time](#211-date-print-or-set-system-date-and-time)
    - [21.1.1. Time conversion specifiers - 时间转义符](#2111-time-conversion-specifiers---时间转义符)
    - [21.1.2. Date conversion specifiers - 日期转义符](#2112-date-conversion-specifiers---日期转义符)
  - [21.2. arch: Print machine hardware name](#212-arch-print-machine-hardware-name)
  - [21.3. nproc: Print the number of available processors](#213-nproc-print-the-number-of-available-processors)
  - [21.4. uname: Print system information](#214-uname-print-system-information)
  - [21.5. hostname: Print or set system name](#215-hostname-print-or-set-system-name)
  - [21.6. hostid: Print numeric host identifier](#216-hostid-print-numeric-host-identifier)
  - [21.7. uptime: Print system uptime and load](#217-uptime-print-system-uptime-and-load)
- [22. SELinux context](#22-selinux-context)
- [23. Modified command invocation](#23-modified-command-invocation)
  - [23.1. chroot: Run a command with a different root directory](#231-chroot-run-a-command-with-a-different-root-directory)
  - [23.2. env: Run a command in a modified environment](#232-env-run-a-command-in-a-modified-environment)
    - [23.2.1. General options](#2321-general-options)
    - [23.2.2. -S/--split-string usage in scripts](#2322--s--split-string-usage-in-scripts)
    - [23.2.3. -S/--split-string syntax](#2323--s--split-string-syntax)
  - [23.3. nice: Run a command with modified niceness](#233-nice-run-a-command-with-modified-niceness)
  - [23.4. nohup: Run a command immune to hangups](#234-nohup-run-a-command-immune-to-hangups)
  - [23.5. stdbuf: Run a command with modified I/O stream buffering](#235-stdbuf-run-a-command-with-modified-io-stream-buffering)
  - [23.6. timeout: Run a command with a time limit](#236-timeout-run-a-command-with-a-time-limit)
- [24. Process control](#24-process-control)
- [25. Delaying](#25-delaying)
- [26. Numeric operations](#26-numeric-operations)
- [27. File permissions](#27-file-permissions)


# 1. GNU Coreutils

https://www.gnu.org/software/coreutils/manual/coreutils.html

作为GNU的官方完整手册, 并没有为新手加以详细的基本概念解释, 因此官方欢迎为 手册进行更新  

GNU Coreutils 的工具绝大多数都与 POSIX 标准相兼容, 遵循了 POSIX 标准

一些命令 (sort, date) 提供了 `--debug` 命令, 可以用于快速寻找问题

很多以为是 linux 基本命令, 其实是 GNU 的 Coreutils 的一个软件  

# 2. Common options

整个 Coreutils 中所有程序都支持的 options, 和 Linux 的基本概念有一些关联    
` (In fact, every GNU program accepts (or should accept) these options.) `  

对于 options 和 operands 的顺序问题
* options and operands can appear in any order
* all the options appear before any operands
* 以上两种程序的命令行输入类型, 会根据 `POSIXLY_CORRECT` 环境变量来决定, 如果该环境变量被定义, 则 `options must appear before operands`


通用命令:
* `--help`    : 打印程序的使用信息  
* `--version` : 打印程序的版本信息
* `--`        : 对于参数的首字符为 `-` 的情况下, `--` 可以让程序强制把后跟的参数识别为参数, 例如文件名以 `-` 开头的情景



# 3. Output of entire files


    3.1 cat: Concatenate and write files
    3.2 tac: Concatenate and write files in reverse
    3.3 nl: Number lines and write files
    3.4 od: Write files in octal or other formats
    3.5 base32: Transform data into printable data
    3.6 base64: Transform data into printable data
    3.7 basenc: Transform data into printable data


# 4. Formatting file contents

    4.1 fmt: Reformat paragraph text
    4.2 pr: Paginate or columnate files for printing
    4.3 fold: Wrap input lines to fit in specified width

# 5. Output of parts of files

    5.1 head: Output the first part of files
    5.2 tail: Output the last part of files
    5.3 split: Split a file into pieces.
    5.4 csplit: Split a file into context-determined pieces

# 6. Summarizing files


    6.1 wc: Print newline, word, and byte counts
    6.2 sum: Print checksum and block counts
    6.3 cksum: Print and verify file checksums
    6.4 b2sum: Print or check BLAKE2 digests
    6.5 md5sum: Print or check MD5 digests
    6.6 sha1sum: Print or check SHA-1 digests
    6.7 sha2 utilities: Print or check SHA-2 digests

# 7. Operating on sorted files

    7.1 sort: Sort text files
    7.2 shuf: Shuffling text
    7.3 uniq: Uniquify files
    7.4 comm: Compare two sorted files line by line
    7.5 ptx: Produce permuted indexes
        7.5.1 General options
        7.5.2 Charset selection
        7.5.3 Word selection and input processing
        7.5.4 Output formatting
        7.5.5 The GNU extensions to ptx
    7.6 tsort: Topological sort
        7.6.1 tsort: Background

These commands work with (or produce) sorted files. 
与有序的文件工作, 或者生成有序的文件

## 7.1. sort: Sort text files

`sort` 命令, 排序, 合并, 比较 给定的文本文件的所有行.  如果没有 文件输入, 则读取标准输入流的输入, 因此可以和 管道命令配合使用.  

`sort [option]… [file]…`

参数: 排序方法
* `-n --numeric-sort` : 按照数字排序, 数字以行为分隔, 任意个空格开始, 带有可选的 `-` 负号符
  * 支持小数点, 千分符
* `-h --human-numeric-sort` : 人类易读的单位下, 按照数字排序, 从小到大
  * 先符号 sign, 后按照 SI suffix, 即 `KMGTPEZYRQ`, 最后按照 数字数值. 该排序无视具体的幂, 因为默认所有输入都是在统一的幂下置换的单位.
  * 主要用来对 `df du ls` 命令的输出进行排序, 且可以接受对应的 `-h` or `-si` 版本的输出
  * 由于 `-h` 的输出通常比较粗略, 因此可以在非 `-h` 的输出下进行排序, 再通过另一个命令 `numfmt` 重新转换为 人类易读的表达  
* `--sort=**`
  * `numeric` : 同 `-n`
  * `human-numeric` : 等同于 `-h`


# 8. Operating on fields


    8.1 cut: Print selected parts of lines
    8.2 paste: Merge lines of files
    8.3 join: Join lines on a common field
        8.3.1 General options
        8.3.2 Pre-sorting
        8.3.3 Working with fields
        8.3.4 Controlling join’s field matching
        8.3.5 Header lines
        8.3.6 Union, Intersection and Difference of files


# 9. Operating on characters


    9.1 tr: Translate, squeeze, and/or delete characters
        9.1.1 Specifying arrays of characters
        9.1.2 Translating
        9.1.3 Squeezing repeats and deleting
    9.2 expand: Convert tabs to spaces
    9.3 unexpand: Convert spaces to tabs

# 10. Directory listing


    10.1 ls: List directory contents
        10.1.1 Which files are listed
        10.1.2 What information is listed
        10.1.3 Sorting the output
        10.1.4 General output formatting
        10.1.5 Formatting file timestamps
        10.1.6 Formatting the file names
    10.2 dir: Briefly list directory contents
    10.3 vdir: Verbosely list directory contents
    10.4 dircolors: Color setup for ls


# 11. Basic operations


    11.1 cp: Copy files and directories
    11.2 dd: Convert and copy a file
    11.3 install: Copy files and set attributes
    11.4 mv: Move (rename) files
    11.5 rm: Remove files or directories
    11.6 shred: Remove files more securely


# 12. Special file types


    12.1 link: Make a hard link via the link syscall
    12.2 ln: Make links between files
    12.3 mkdir: Make directories
    12.4 mkfifo: Make FIFOs (named pipes)
    12.5 mknod: Make block or character special files
    12.6 readlink: Print value of a symlink or canonical file name
    12.7 rmdir: Remove empty directories
    12.8 unlink: Remove files via the unlink syscall

特殊的文件类型

Unix-like 操作系统的特殊文件类型要少于其他操作系统, 但并不是所有的东西都能够是为 普通文件的 无差别字节流  

除了文件夹以外, 其他的特殊文件类型还可以包括
* named pipes (FIFOs)
* symbolic links
* sockets
* so-call special files

## 12.1. link - Make a hard link via the link syscall

## 12.2. ln - Make links between files

## 12.3. mkdir - Make directories

    mkdir [option]… name…




# 13. Changing file attributes


    13.1 chown: Change file owner and group
    13.2 chgrp: Change group ownership
    13.3 chmod: Change access permissions
    13.4 touch: Change file timestamps

# 14. File space usage


14.1 df: Report file system space usage - 打印磁盘的使用率

14.2 du: Estimate file space usage - 打印文件的空间占用

14.3 stat: Report file or file system status
14.4 sync: Synchronize cached writes to persistent storage
14.5 truncate: Shrink or extend the size of a file

用于文件容量的管理, 可以查看系统容量的空间状态, 文件的容量信息, 缓冲区信息等

## 14.1. df: Report file system space usage

df 命令打印的是文件系统的空间, 即整个电脑的空间使用率

`df [option]… [file]…`  
没有 `file` 参数的话, 会打印所有 mount 挂载的容量信息 (all currently mounted file systems)   
如果有 file 参数的话, 则会打印 file 所在的 mount 挂载的容量信息, 即 file 仅仅只是用来指定 挂载点的

信息的默认单位是 1k-byte (1kb), 不足的部分会被 round up


参数列表:
* `-a --all` : 是否打印全部的挂载点
  * 对于没有 -a 的 df 无参数默认调用, 仅输出文件系统列表中具有最短挂载点的设备, 即 device with the shortest mount point name in the `mtab`, 同时隐藏重复的条目
  * 如果输入该参数, 则会打印: 重复, 虚拟, 无法访问 的文件系统
    * 所谓虚拟文件系统 (dummy file systems) 即用于特殊目的的伪文件系统, 例如 `/proc`, 其没有对应的存储
    * 重复的文件系统 (duplicate file systems) , 即通过 mount 将某个文件系统挂载在了不同的位置, 或者将本地的系统挂载到本地的另一个位置, 通过 `mount --bind`
    * 无法访问的文件系统是那些过期的, 或者访问权限失效的挂载点
* `-h --human-readable` : 以人类易读的方式打印, 自动以 1024为幂进行单位转换, kb -> mb ->gb ...
* `-H --si` : 以 SI-style 的缩写方式打印, 即以 1000 为幂进行单位转换, 而不是 `-h` 那样的 1024


## 14.2. du: Estimate file space usage

打印某个 a set of files 的空间占用

`du [option]… [file]… `
对于无参数调用, du 打印当前目录以及 当前目录下的递归目录所需要的空间, 默认以 1kb 为单位.

du 对于 hard link 会跳过. du 的参数的顺序也会影响 du 输出的数字和条目

参数
* `-a --all` : 同时输出 文件的占用空间, 不仅仅是 目录
* `-s --summarize` : 只输出该目录整个的占用空间 
* `-d depth` `--max-depth=depth` : 只输出深度在 depth 以内的目录空间, root 为0, 因此 `-d 0` 相当于 `-s`
* `-h --human-readable`  `--si` : 作用同上面的 `df`, 要注意 du 的 `-H` 有不同的含义
* `-H -D --dereference-args` : 取消作为 命令行参数的 符号链接, 但是不影响其他的符号连接, 这对于统计某些存在符号链接目录的路径有帮助, 例如 `/usr/tmp`

 

## 14.3. sync: Synchronize cached writes to persistent storage
<!-- 完, 但不太理解 -->

将缓存中或者文件系统中的文件 同步到永久存储中, 用于强制将内存中的数据写入磁盘，以确保数据的持久化和同步。该命令对于确保文件系统的一致性和数据的安全性非常重要   
writes any data buffered in memory out to the storage device.  
This can include (but is not limited to) modified superblocks, modified inodes, and delayed reads and writes.  

`sync [option] [file]…`  

这是一个 kernel 必要的程序, 该程序的实际工作由 system call 来完成, 具体的为调用 `sync, syncfs, fsync, and fdatasync`

内存数据主要用于提高计算机的反应速度, 但如果计算机由于内存占满或者其他原因导致的崩溃, 则需要把内存数据写入到永久存储中防止数据丢失. 操作系统一般会在适当的时候自动执行该命令, 因此大多时候不需要手动执行.    

如果指定了 `file` 参数, 则会仅写入指定的文件, 通过 `fsync` 系统调用

如果指定了 `file`, 则可以使用参数:
* `-d --data` : 仅同步文件相关的缓存, 包括 data for the file, any metadata required to maintain file system consistency.  通过 `fdatasync` 系统调用来实现  
* `-f --file-system` : 使用 `syncfs` 来将各种系统相关的缓存写入
  * all the I/O waiting for the file systems that contain the file


# 15. Printing text

    15.1 echo: Print a line of text
    15.2 printf: Format and print data
    15.3 yes: Print a string until interrupted

## 15.1. echo: Print a line of text

`echo [option]… [string]…`

将 string 打印到 std output, 并且在每一个 string 后面加一个空格, 在最终的 string 后面加换行符  

根据 shell 的不同, 该命令有可能实际上会调用不同的 binary, 因此可以使用 `env echo` 来确保使用的是操作系统的 echo  



# 16. Conditions


    16.1 false: Do nothing, unsuccessfully
    16.2 true: Do nothing, successfully
    16.3 test: Check file types and compare values
        16.3.1 File type tests
        16.3.2 Access permission tests
        16.3.3 File characteristic tests
        16.3.4 String tests
        16.3.5 Numeric tests
        16.3.6 Connectives for test
    16.4 expr: Evaluate expressions
        16.4.1 String expressions
        16.4.2 Numeric expressions
        16.4.3 Relations for expr
        16.4.4 Examples of using expr

# 17. Redirection

17.1 tee: Redirect output to multiple files or processes

# 18. File name manipulation

    18.1 basename: Strip directory and suffix from a file name
    18.2 dirname: Strip last file name component
    18.3 pathchk: Check file name validity and portability
    18.4 mktemp: Create temporary file or directory
    18.5 realpath: Print the resolved file name.
        18.5.1 Realpath usage examples

# 19. Working context

    19.1 pwd: Print working directory
    19.2 stty: Print or change terminal characteristics
        19.2.1 Control settings
        19.2.2 Input settings
        19.2.3 Output settings
        19.2.4 Local settings
        19.2.5 Combination settings
        19.2.6 Special characters
        19.2.7 Special settings
    19.3 printenv: Print all or some environment variables
    19.4 tty: Print file name of terminal on standard input

This section describes commands that display or alter the context in which you are working: the current directory, the terminal settings, and so forth. See also the user-related commands in the next section. 

关于当前工作环境的一些上下文软件  

## 19.1. pwd: Print working directory

## 19.2. stty: Print or change terminal characteristics


## 19.3. printenv: Print all or some environment variables

## 19.4. tty: Print file name of terminal on standard input


# 20. User information

    20.1 id: Print user identity
    20.2 logname: Print current login name
    20.3 whoami: Print effective user name
    20.4 groups: Print group names a user is in
    20.5 users: Print login names of users currently logged in
    20.6 who: Print who is currently logged in

This section describes commands that print user-related information: logins, groups, and so forth. 

<!-- 完 -->

## 20.1. id: Print user identity

打印一个 user 的信息  
`id [option]… [user]…`

这里 `[user]` 可以输入 ID 或者 用户 name, name 的优先级更高, 在加了前缀 `+` 的时候会优先查找 ID

默认动作 : 打印 
* real user ID
* real group ID
* effective user ID if different from the real user ID
* effective group ID if different from the real group ID
* supplemental group IDs. 
* In addition, if SELinux is enabled and the `POSIXLY_CORRECT` environment variable is not set, then print ‘context=c’, where c is the security context. 

信息的构成: Each of these numeric values is preceded by an identifying string and followed by the corresponding user or group name in parentheses. 

参数: 用于指定只打印 默认动作的一部分信息, 在指定信息的时候一次只能指定一种
* `-g --group`  : 只打印 group ID
* `-G --groups` : 只打印 group ID 以及 supplementary groups
* `-u --user`   : 只打印 user ID
* `-n --name`   : Print the user or group name instead of the ID number.  不能单独启用该命令, 必须与 `-u -g -G` 一起
* `-Z --context` :  If neither `SELinux` or `SMACK` is enabled then print a warning and set the exit status to 1. 
* `-z --zero` : 用于一些特殊场景, 会将元素分隔符替换成 `ASCII NUL`, 该选项不能在默认输出的时候启用
  * 主要用于 `--groups` 的输出
  * When multiple users are specified, and the --groups option is also in effect, groups are delimited with a single NUL character, while users are delimited with two NUL characters

<!-- 完 -->
## 20.2. logname : Print current login name

`logname` 所打印的用户名是 calling user's name, 原理上会从 system-maintained file `/var/run/utmp` 或者 `/etc/utmp` 中查找

如果查找到了, 会输出并返回 0.  
如果没有查找到, 则会输出错误 `logname: no login name` 并返回 1.  对于本地用户wsl来说, 因为不存在 login 因此不会查找到  

<!-- 完 -->

## 20.3. whoami : Print effective user name

打印 user name associated with the current effective user ID  

该行为等同于 `id -un`     不存在其他参数

<!-- 完 -->

## 20.4. groups

打印组 names of the primary and any supplementary groups for each given `username`.
`groups [username]…`  同 id 一样,  该查找软件一样可以指定用户名, 但只能指定 name 不能指定 ID

因此该软件等同于 `id -Gn`  , 而且还没有 id 的 -z 更改输出分隔符的功能  

关于组的继承:  在登陆后更改用户的组信息后, 组信息并不会立即反映在当前的 session 上  
Primary and supplementary groups for a process are normally inherited from its parent and are usually unchanged since login. This means that if you change the group database after logging in, groups will not reflect your changes within your existing login session. Running groups with a list of users causes the user and group database to be consulted afresh, and so will give a different result. 

<!-- 完 -->

## 20.5. users : Print login names of users currently logged in

`users [file]`  

打印一个 a single line a blank-separated list of user names  `(of users currently logged in to the current host)`  
* 即对于服务器来说, 查看当前也连入了改服务器的用户名
* Each user name corresponds to `a login session`, so if a user has more than one login session, that user’s name will appear the same number of times in the output.

`[file]` 仅仅只是用于指定其他的 tmp file
* 默认的系统维护文件查找目标是  `/var/run/utmp` or `/etc/utmp`
* 一个常用的查找目标可能是  `/var/log/wtmp`  

The users command is installed only on platforms with the POSIX `<utmpx.h>` include file or equivalent, so portable scripts should not rely on its existence on non-POSIX platforms.   

该软件接口也是 轻量化系统所非必须的

<!-- 完 -->

## 20.6. who : Print who is currently logged in

是一个强化版的 `users`  , 但是是以 processes 为单位的

`who` prints information about users who are currently logged on.  
`who [option] [file] [am i]`

默认行为会打印 :  following information for each user currently logged on
* login name
* terminal line
* login time
* remote hostname or X display. 

`[file]`  : 同样的, 用于指定特殊系统配置下的 系统维护文件目录, 即 `var/run/utmp` or `/etc/utmp` 以外的目录  
`[am i]`  : 变相的 `whoami`, 在服务器上会得到相同的结果, 但是 `whoami` 的查找方式不同因此能够查找到本地用户的名称, 而 `who am i` 不行

参数: 用于指定要打印的信息  
* `-b boot`: 打印 the date and time of last system boot. 实测在 wsl 和 docker 环境里没有输出
* `-d --dead` : 打印 information corresponding to dead processes.  打印停止的 processes
* `-l --login` : 打印正在等待用户登录的进程相对应的条目, 
  * List only the entries that correspond to processes via which the system is waiting for a user to login. 
  * The user name is always ‘LOGIN’. 
* `-p -process`  :  List active processes spawned by init. 打印出由 init 调用的活动进程 
* `-r --runlevel` : Print the current (and maybe previous) run-level of the init process. 
* `-t --time` : Print last system clock change. 
* `-u`        : After the login time, print the number of hours and minutes that the user has been idle. 
  * 打印用户的空闲时间??? (`IDLE`)
  * `.` 表示用户在当前最后一分钟仍然处于活跃
  * `old` 表示用户已经超越24小时保持空闲
* `-w -T --mesg --message --writable` : 在每一个用户名的后面打印一个字符用户  indicating the user’s message status: 
  * ‘+’ allowing write messages
  * ‘-’ disallowing write messages
  * ‘?’ cannot find terminal device
* `-H --heading` :  Print a line of column headings. 在打印的时候加上表头.
* `-a all` : 打印全部, 相当于 `-b -d --login -p -r -t -T -u`. 注意没有: `-H`, 需要手动指定 `who -a -H`
  * 其他的 `-a` 不会代表的选项记录在下面
* `-m`  : 相当于 `who am i`
* `-q --count`   : Print only the login names and the number of users logged on. Overrides all other options. 
* `--lookup` : 尝试通过 DNS 来查找 `utmp` 所出现的主机名, 非默认行为, 可能会导致巨大延迟.
* `-s`  : 兼容性命令, 已不再使用  



The `who` command is installed only on platforms with the POSIX `<utmpx.h>` include file or equivalent, so portable scripts should not rely on its existence on non-POSIX platforms. 
who也是轻量化系统所非必须的软件接口  

<!-- 完 -->

# 21. System context

    21.1 date: Print or set system date and time
        21.1.1 Time conversion specifiers
        21.1.2 Date conversion specifiers
        21.1.3 Literal conversion specifiers
        21.1.4 Padding and other flags
        21.1.5 Setting the time
        21.1.6 Options for date
        21.1.7 Examples of date
    21.2 arch: Print machine hardware name
    21.3 nproc: Print the number of available processors
    21.4 uname: Print system information
    21.5 hostname: Print or set system name
    21.6 hostid: Print numeric host identifier
    21.7 uptime: Print system uptime and load

This section describes commands that print or change system-wide information. 
打印或者更改系统层面的相关信息  
有些难界定是否能输出硬件相关的信息  

需要注意很多不会被默认安装, 因此在编写 script 的时候要留意


## 21.1. date: Print or set system date and time

打印当前时间的命令, 但是意外的很复杂  

```sh
date [option]… [+format]
date [-u|--utc|--universal] [ MMDDhhmm[[CC]YY][.ss] ]
```


### 21.1.1. Time conversion specifiers - 时间转义符

| 转义符(省略前置的 `%` ) | 意思                                                                      |
| ----------------------- | ------------------------------------------------------------------------- |
| H                       | 24小时制的时, 用0 始终占 2 位                                             |
| h                       | 12 小时制的时, 用0 始终占 2 位                                            |
| k                       | 24 小时制的时, 用空格 始终占 2 位, GNU 拓展                               |
| l                       | 12 小时制的时, 用空格 始终占 2 位, GNU 拓展                               |
| M                       | 分钟, 0占 2位                                                             |
| N                       | nanoseconds  ` (‘000000000’…‘999999999’)` GNU 拓展                        |
| p                       | 上午或者下午  `AM PM`, 但是在很多语言环境中都是空白                       |
| r                       | locale’s time representation 当地的 12 小时制的时间, 格式为 `11:11:04 PM` |
| X                       | locale’s time representation 当地的 24 小时制的时间, 格式为 `23:13:48`    |
| R                       | 24 小时制的时间, 格式为 `%H:%M`                                           |
| T                       | 24 小时制的时间 , 相当于 `%H:%M:%S`                                       |
| s                       | seconds since the Epoch, ,经典 但是是GNU 拓展                             |
| S                       | second `(‘00’…‘60’)`  注意, 如果系统支持闰秒的话, 则该值可以是 60         |
| z                       | Four-digit numeric time zone, e.g., ‘-0600’ or ‘+0530’, or ‘-0000’        |
| :z                      | 带冒号 `:`的 Four-digit numeric time zone, e.g. `-06:00`                  |
| ::z                     | 带冒号带秒的 time zone, `-06:00:00`                                       |
| :::z                    | 用最小的准确精度表示的 time zone, e.g. `-06   +05:30`                     |
| Z                       | 用 alphabetic 表示的 time zone                                            |

### 21.1.2. Date conversion specifiers - 日期转义符

| 转义符(省略前置的 `%` ) | 意思     

## 21.2. arch: Print machine hardware name
<!-- 完 -->
arch prints the machine hardware name, and is equivalent to `uname -m`. 
过于简单, 可能仅仅只是为了记不住 uname 的参数而编写的程序  

arch 没有其他任何参数

`arch is not installed by default, so portable scripts should not rely on its existence. `
需要移植的脚本不应该依赖该软件, 因为不会被默认安装


## 21.3. nproc: Print the number of available processors
<!-- 完 -->

`nproc [option]`  
打印可以使用的处理单元个数 number of processing units available, 总是输出大于 0 的数字  

很严谨的说明是, number of processing units available to the `current process`
即有可能因为当前 process 的各种限制实际上输出会少于 online processors

If this information is not accessible, then print the number of processors installed.  

If the `OMP_NUM_THREADS` or `OMP_THREAD_LIMIT` environment variables are set, then they will determine the minimum and maximum returned value respectively.  



参数:
* `--all` : Print the number of installed processors on the system, which may be greater than the number online or available to the current process.
* `--ignore=number` : 如果可能的话, 在基础上减去该数量的个数, 算是为 nproc 的输出进行一个带判定的减法, 即还是保证结果大于 0

## 21.4. uname: Print system information
<!-- 完 -->
uname 看起来像是打印 用户名 User-name 的样子, 实际上是输出系统的名字, 关于用户的信息记录在上一章节 `User information` 中  


`uname [option]…`   默认会输出 `-s` 的内容
  
如果命令带有 `-a` 或者带有多个 options 的话, 会以如下的顺序输出所有信息  
由于信息中有可能带有空格, 所以无法通过空格来可靠的完整解析内容, 因此必要的时候通过指定 options 来限制输出内容
```
kernel-name nodename kernel-release kernel-version
machine processor hardware-platform operating-system
```

参数
* `-s --kernel-name`        : 打印内核名称 (the implementation of the operating system), 可能和操作系统名称一样, 实测为 `Linux`
* `-n --nodename`   : 打印  network node hostname , 即在网络节点上显示的本机器的名称, 一般由机器管理员设定
* `-r --kernel-release`     : 打印 内核 release `5.10.16.3-microsoft-standard-WSL2`
* `-v --kernel-version`     : 内核的版本, 实测为 `#1 SMP Fri Apr 2 22:23:49 UTC 2021`
* `-m --machine`    : 打印 hardware name (hardware class or hardware type), 实测输出也是 `x86_64`
* `-p --processor`  : 处理器类型 (instruction set architecture or ISA), 实测输出也是 `x86_64`
* `-i --hardware-platform`  : 类似于 `x86_64` 的平台名称
* `-o --operating-system`   : 输出操作系统的名称 实测为`GNU/Linux`, 而并非发行版的名称
* `-a --all`        : 打印所有, except omit the `processor` type and the `hardware platform name` if they are unknown. 


## 21.5. hostname: Print or set system name
<!-- 完 -->
`hostname [name]`

输出或者这设置当前的 hostname, `name of the current host system` , 如果设置的话需要有对应的系统权限  

该命令的输出 与 uname -n 的 nodename (network node hostname) 在实测中是相同的, 但是意思可能有些不同

没有其他任何参数

hostname is not installed by default, and other packages also supply a hostname command, so portable scripts should not rely on its existence or on the exact behavior documented above. 


## 21.6. hostid: Print numeric host identifier

<!-- 完 -->
打印数字的当前 host 的 id, 以 16 进制数字输出, 该程序没有任何参数, 除了 help 和 version

一般 id 会是长度为 8 的16进制数字, 即 32位信息, 理论上这与 ip 地址的 32 位有关, 但是事实上可能并非如此

`hostid` 只会安装在有 `gethostid` 的机器上. 

同样的 `portable scripts should not rely on its existence.`


## 21.7. uptime: Print system uptime and load

好像是非常有用的命令, 输出的信息都很关键    



# 22. SELinux context

    22.1 chcon: Change SELinux context of file
    22.2 runcon: Run a command in specified SELinux context

在不适用 SELinux 的情况下好像用不到该命令  



# 23. Modified command invocation

    23.1 chroot: Run a command with a different root directory
    23.2 env: Run a command in a modified environment
        23.2.1 General options
        23.2.2 -S/--split-string usage in scripts
            Testing and troubleshooting
        23.2.3 -S/--split-string syntax
            Splitting arguments by whitespace
            Escape sequences
            Comments
            Environment variable expansion
    23.3 nice: Run a command with modified niceness
    23.4 nohup: Run a command immune to hangups
    23.5 stdbuf: Run a command with modified I/O stream buffering
    23.6 timeout: Run a command with a time limit

主要用于更改 command 的调用行为  

## 23.1. chroot: Run a command with a different root directory

## 23.2. env: Run a command in a modified environment

在一个指定的修改后的 environment 下运行命令, 语法:

```sh  
env [option]… [name=value]… [command [args]…]
env -[v]S'[option]… [name=value]… [command [args]…]'
env

# 熟知的 env 常常被用来书写 #! 解释器
#!/usr/bin/env command
#!/usr/bin/env -[v]S[option]… [name=value]… command [args]…
```
* `env [option]… [name=value]… [command [args]…]`
  * 除去 option 以外, 第一个不包含 `=` 的参数会作为 command 来在 env 的环境下运行
  * command 会根据 `PATH` 环境变量来查找
  * 剩下的 args 则会直接传入 command 里
  * 对于应用程序的名称包含了 `=` 的极其特殊情况, 可以使用其他的解释器或者终端作为中介, 程序名称以 args 传入, 例如 `sh bash python` 等
* `env` : 打印所有环境变量, 同 `printenv` 的行为一样  


关于环境变量:
* 通常都是 `k=v` 的键值对形式
* 可以设置 `key=` 即值为空, 值为空  与  键值对未定义是 `不一样`  的
* 运行时对于环境变量是从左往右读取的, 因此如果 读取了相同的 环境变量两次, 则前一次的会被覆盖, 被 ignore.
* 环境变量的命名规则
  * 从语法上, variable names 可以为空, 并且可以包括出去 `=` 和 `ASCII NUL` 之外的任何空格
  * 从惯例上, 最好限制只由 下划线, 数字, 字母组成,  并且以非数字开头


### 23.2.1. General options

### 23.2.2. -S/--split-string usage in scripts

### 23.2.3. -S/--split-string syntax

## 23.3. nice: Run a command with modified niceness

## 23.4. nohup: Run a command immune to hangups

## 23.5. stdbuf: Run a command with modified I/O stream buffering

## 23.6. timeout: Run a command with a time limit

# 24. Process control

    24.1 kill: Send a signal to processes

# 25. Delaying

    25.1 sleep: Delay for a specified time

# 26. Numeric operations

    26.1 factor: Print prime factors
    26.2 numfmt: Reformat numbers
        26.2.1 General options
        26.2.2 Possible units:
        26.2.3 Examples of using numfmt
    26.3 seq: Print numeric sequences

# 27. File permissions

    27.1 Structure of File Mode Bits
    27.2 Symbolic Modes
        27.2.1 Setting Permissions
        27.2.2 Copying Existing Permissions
        27.2.3 Changing Special Mode Bits
        27.2.4 Conditional Executability
        27.2.5 Making Multiple Changes
        27.2.6 The Umask and Protection
    27.3 Numeric Modes
    27.4 Operator Numeric Modes
    27.5 Directories and the Set-User-ID and Set-Group-ID Bits
- [1. GNU Coreutils](#1-gnu-coreutils)
- [2. Common options](#2-common-options)
- [3. Output of entire files](#3-output-of-entire-files)
- [4. Formatting file contents](#4-formatting-file-contents)
- [5. Output of parts of files](#5-output-of-parts-of-files)
- [6. Summarizing files](#6-summarizing-files)
- [7. Operating on sorted files](#7-operating-on-sorted-files)
  - [7.1. sort: Sort text files](#71-sort-sort-text-files)
- [8. Operating on fields](#8-operating-on-fields)
- [9. Operating on characters](#9-operating-on-characters)
- [10. Directory listing](#10-directory-listing)
- [11. Basic operations](#11-basic-operations)
- [12. Special file types](#12-special-file-types)
  - [12.1. link - Make a hard link via the link syscall](#121-link---make-a-hard-link-via-the-link-syscall)
  - [12.2. ln - Make links between files](#122-ln---make-links-between-files)
  - [12.3. mkdir - Make directories](#123-mkdir---make-directories)
- [13. Changing file attributes](#13-changing-file-attributes)
- [14. File space usage](#14-file-space-usage)
  - [14.1. df: Report file system space usage](#141-df-report-file-system-space-usage)
  - [14.2. du: Estimate file space usage](#142-du-estimate-file-space-usage)
  - [14.3. sync: Synchronize cached writes to persistent storage](#143-sync-synchronize-cached-writes-to-persistent-storage)
- [15. Printing text](#15-printing-text)
  - [15.1. echo: Print a line of text](#151-echo-print-a-line-of-text)
- [16. Conditions](#16-conditions)
- [17. Redirection](#17-redirection)
- [18. File name manipulation](#18-file-name-manipulation)
- [19. Working context](#19-working-context)
  - [19.1. pwd: Print working directory](#191-pwd-print-working-directory)
  - [19.2. stty: Print or change terminal characteristics](#192-stty-print-or-change-terminal-characteristics)
  - [19.3. printenv: Print all or some environment variables](#193-printenv-print-all-or-some-environment-variables)
  - [19.4. tty: Print file name of terminal on standard input](#194-tty-print-file-name-of-terminal-on-standard-input)
- [20. User information](#20-user-information)
  - [20.1. id: Print user identity](#201-id-print-user-identity)
  - [20.2. logname : Print current login name](#202-logname--print-current-login-name)
  - [20.3. whoami : Print effective user name](#203-whoami--print-effective-user-name)
  - [20.4. groups](#204-groups)
  - [20.5. users : Print login names of users currently logged in](#205-users--print-login-names-of-users-currently-logged-in)
  - [20.6. who : Print who is currently logged in](#206-who--print-who-is-currently-logged-in)
- [21. System context](#21-system-context)
  - [21.1. date: Print or set system date and time](#211-date-print-or-set-system-date-and-time)
    - [21.1.1. Time conversion specifiers - 时间转义符](#2111-time-conversion-specifiers---时间转义符)
    - [21.1.2. Date conversion specifiers - 日期转义符](#2112-date-conversion-specifiers---日期转义符)
  - [21.2. arch: Print machine hardware name](#212-arch-print-machine-hardware-name)
  - [21.3. nproc: Print the number of available processors](#213-nproc-print-the-number-of-available-processors)
  - [21.4. uname: Print system information](#214-uname-print-system-information)
  - [21.5. hostname: Print or set system name](#215-hostname-print-or-set-system-name)
  - [21.6. hostid: Print numeric host identifier](#216-hostid-print-numeric-host-identifier)
  - [21.7. uptime: Print system uptime and load](#217-uptime-print-system-uptime-and-load)
- [22. SELinux context](#22-selinux-context)
- [23. Modified command invocation](#23-modified-command-invocation)
  - [23.1. chroot: Run a command with a different root directory](#231-chroot-run-a-command-with-a-different-root-directory)
  - [23.2. env: Run a command in a modified environment](#232-env-run-a-command-in-a-modified-environment)
    - [23.2.1. General options](#2321-general-options)
    - [23.2.2. -S/--split-string usage in scripts](#2322--s--split-string-usage-in-scripts)
    - [23.2.3. -S/--split-string syntax](#2323--s--split-string-syntax)
  - [23.3. nice: Run a command with modified niceness](#233-nice-run-a-command-with-modified-niceness)
  - [23.4. nohup: Run a command immune to hangups](#234-nohup-run-a-command-immune-to-hangups)
  - [23.5. stdbuf: Run a command with modified I/O stream buffering](#235-stdbuf-run-a-command-with-modified-io-stream-buffering)
  - [23.6. timeout: Run a command with a time limit](#236-timeout-run-a-command-with-a-time-limit)
- [24. Process control](#24-process-control)
- [25. Delaying](#25-delaying)
- [26. Numeric operations](#26-numeric-operations)
- [27. File permissions](#27-file-permissions)
