# 1. Archive Software

打包与压缩
* 打包是不经过压缩, 只是把几个文件存储在一个文件下
* 压缩是经过算法, 使得打包的文件体积变小

## 1.1. tar

Linux中最常用的打包命令
* 使用 tar 命令归档的包成为 tar 包, 文件以 `.tar` 结尾
* `tar [选项] 源文件或目录`  可以同时打包多个文件或目录, 用空格分开名称
* 最常用的打包指令就是 
  * `-cvf 输出名 输入文件`
* 最常用的解包指令是 
  * `-xvf tar文件名 -C 输出目录`
* 如果需要在打包的同时进行压缩, 或者解压缩, 使用 压缩相关的专用命令
  * `-z` 对 .tar.gz 格式的文件进行解压缩或者压缩成该格式 , 即用 `-zcvf`
  * `-j` 对 .tar.bz2 格式的文件进行解压缩或者压缩成该格式, `-jcvf `

打包命令表
| 选项    | 功能                                     |
| ------- | ---------------------------------------- |
| -c      | 多个文件或者目录进行打包                 |
| -A      | 追加 tar 文件到一个已归档的文件          |
| -f 包名 | 指定输出的文件名, 注意要指定正确的拓展名 |
| -v      | 显示打包过程                             |

解包命令表
| 选项    | 功能                                     |
| ------- | ---------------------------------------- |
| -x      | 对 tar 文件进行解包                      |
| -f 包名 | 指定要解包的包名                         |
| -t      | 不进行解包, 只输出包中的文件名和一级目录 |
| -C      | 指定解打包的位置                         |
| -v      | 显示打包过程                             |




## 1.2. 7-Zip

7-Zip 是一款拥有极高压缩比的开源压缩软件  
p7zip - Linux/Posix 平台的命令行移植版本  

支持的:  
压缩:  7z, XZ, BZIP2, GZIP, TAR, ZIP and WIM
解压:  ARJ, CAB, CHM, CPIO, CramFS, DEB, DMG, FAT, HFS, ISO, LZH, LZMA, MBR, MSI, NSIS, NTFS, RAR, RPM, SquashFS, UDF, VHD, WIM, XAR, Z




## 1.3. zip 命令

zip是 windows 和 linux 通用的文件压缩方法, 属于主流之一
* zip 类型的压缩和解压缩的命令是分开的
* `zip [选项] 压缩包名 源文件或源目录列表`, 压缩命令要先输入输出文件名
* `unzip [选项] 压缩包名`

zip命令表
| 选项      | 含义                                                                |
| --------- | ------------------------------------------------------------------- |
| -r        | 递归压缩目录，及将制定目录下的所有文件以及子目录全部压缩。          |
| -m        | 将文件压缩之后，删除原始文件，相当于把文件移到压缩文件中。          |
| -v        | 显示详细的压缩过程信息。                                            |
| -q        | 在压缩的时候不显示命令的执行过程。                                  |
| -压缩级别 | 压缩级别是从 1~9 的数字，-1 代表压缩速度更快，-9 代表压缩效果更好。 |
| -u        | 更新压缩文件，即往压缩文件中添加新文件。                            |


unzip命令表
| 选项          | 含义                                                       |
| ------------- | ---------------------------------------------------------- |
| -d 目录名     | 将压缩文件解压到指定目录下。                               |
| -n            | 解压时并不覆盖已经存在的文件。                             |
| -o            | 解压时覆盖已经存在的文件，并且无需用户确认。               |
| -v            | 不解压查看压缩文件的详细信息，文件大小、文件名以及压缩比等 |
| -t            | 测试压缩文件有无损坏，但并不解压。                         |
| -x 文件名列表 | 解压文件，但不包含文件列表中指定的文件。                   |

## 1.4. rar 命令

rar 是一种专利文件格式, 通常情况比ZIP压缩比高, 但压缩/解压缩速度较慢