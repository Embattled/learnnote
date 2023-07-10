- [1. GNU Coreutils](#1-gnu-coreutils)
- [2. Common options](#2-common-options)
- [3. Output of entire files](#3-output-of-entire-files)
- [4. Formatting file contents](#4-formatting-file-contents)
- [5. Output of parts of files](#5-output-of-parts-of-files)
- [6. Summarizing files](#6-summarizing-files)
- [7. Operating on sorted files](#7-operating-on-sorted-files)
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
- [15. Printing text](#15-printing-text)
- [16. Conditions](#16-conditions)
- [17. Redirection](#17-redirection)
- [18. File name manipulation](#18-file-name-manipulation)
- [19. Working context](#19-working-context)
- [20. User information](#20-user-information)
- [21. System context](#21-system-context)
  - [21.1. date: Print or set system date and time](#211-date-print-or-set-system-date-and-time)
  - [21.2. arch: Print machine hardware name](#212-arch-print-machine-hardware-name)
  - [21.3. nproc: Print the number of available processors](#213-nproc-print-the-number-of-available-processors)
  - [21.4. uname: Print system information](#214-uname-print-system-information)
  - [21.5. hostname: Print or set system name](#215-hostname-print-or-set-system-name)
  - [21.6. hostid: Print numeric host identifier](#216-hostid-print-numeric-host-identifier)
  - [21.7. uptime: Print system uptime and load](#217-uptime-print-system-uptime-and-load)
- [22. SELinux context](#22-selinux-context)
- [23. Modified command invocation](#23-modified-command-invocation)
- [24. Process control](#24-process-control)
- [25. Delaying](#25-delaying)
- [26. Numeric operations](#26-numeric-operations)
- [27. File permissions](#27-file-permissions)


# 1. GNU Coreutils

https://www.gnu.org/software/coreutils/manual/coreutils.html

作为GNU的官方完整手册, 并没有为新手加以详细的基本概念解释, 因此官方欢迎为 手册进行更新  

GNU Coreutils 的工具绝大多数都与 POSIX 标准相兼容, 遵循了 POSIX 标准

一些命令 (sort, date) 提供了 `--debug` 命令, 可以用于快速寻找问题


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


    14.1 df: Report file system space usage
    14.2 du: Estimate file space usage
    14.3 stat: Report file or file system status
    14.4 sync: Synchronize cached writes to persistent storage
    14.5 truncate: Shrink or extend the size of a file

# 15. Printing text

    15.1 echo: Print a line of text
    15.2 printf: Format and print data
    15.3 yes: Print a string until interrupted

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

# 20. User information

    20.1 id: Print user identity
    20.2 logname: Print current login name
    20.3 whoami: Print effective user name
    20.4 groups: Print group names a user is in
    20.5 users: Print login names of users currently logged in
    20.6 who: Print who is currently logged in

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
uname 看起来像是打印 用户名 User-name 的样子, 实际上是输出系统的名字

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
- [15. Printing text](#15-printing-text)
- [16. Conditions](#16-conditions)
- [17. Redirection](#17-redirection)
- [18. File name manipulation](#18-file-name-manipulation)
- [19. Working context](#19-working-context)
- [20. User information](#20-user-information)
- [21. System context](#21-system-context)
  - [21.1. date: Print or set system date and time](#211-date-print-or-set-system-date-and-time)
  - [21.2. arch: Print machine hardware name](#212-arch-print-machine-hardware-name)
  - [21.3. nproc: Print the number of available processors](#213-nproc-print-the-number-of-available-processors)
  - [21.4. uname: Print system information](#214-uname-print-system-information)
  - [21.5. hostname: Print or set system name](#215-hostname-print-or-set-system-name)
  - [21.6. hostid: Print numeric host identifier](#216-hostid-print-numeric-host-identifier)
  - [21.7. uptime: Print system uptime and load](#217-uptime-print-system-uptime-and-load)
- [22. SELinux context](#22-selinux-context)
- [23. Modified command invocation](#23-modified-command-invocation)
- [24. Process control](#24-process-control)
- [25. Delaying](#25-delaying)
- [26. Numeric operations](#26-numeric-operations)
- [27. File permissions](#27-file-permissions)
