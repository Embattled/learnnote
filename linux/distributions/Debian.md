
# 1. Debian系列 DPKG APT

## 1.1. 基础关系  

apt会解决和安装模块的依赖问题,并会咨询软件仓库, 但不会安装本地的deb文件, apt是建立在dpkg之上的软件管理工具  

dpkg是用来安装.deb文件,但不会解决模块的依赖关系,且不会关心ubuntu的软件仓库内的软件,可以用于安装本地的deb文件  

dpkg绕过apt包管理数据库对软件包进行操作，所以你用dpkg安装过的软件包用apt可以再安装一遍，系统不知道之前安装过了，将会覆盖之前dpkg的安装。

## 1.2. DPKG

dpkg 是Debian package的简写，为”Debian“ 操作系统 专门开发的套件管理系统，用于软件的安装，更新和移除。

所有源自"Debian"的Linux的发行版都使用 dpkg,   例如"Ubuntu"

| 命令                | 实例                           | 说明                                                        |
| ------------------- | ------------------------------ | ----------------------------------------------------------- |
| -i <.deb file name> | dpkg -i  ~/mozybackup_i386.deb | 安装手动下载下来的包                                        |
| -I                  | dpkg -I                        | 查看当前系统中已经安装的软件包的信息                        |
| -l package          | dpkg -l mozybackup             | 显示包的版本以及在系统中的状态                              |
| -L package          | dpkg -L mozybackup             | 安装完包后，可以用此命令查看软件安装到什么地方.**重要命令** |
| -r package          | dpkg -r mozybackup             | 删除软件但是保留配置                                        |
| -P package          | dpkg -P mozybackup             | 删除软件且删除配置                                          |
| -s package          | dpkg -s mozybackup             | 查看已经安装的软件的详细信息                                |
| -S                  | dpkg -S                        | 查看某个文件属于哪一个软件包                                |
| -c pac.deb          | dpkg -c mozybac.deb            | 查看一个安装包的内容                                        |
| -A pac.deb          | dpkg -A package_file           | 查看一个安装包的软件信息                                    |

---
`#dpkg --get-selections isc-dhcp-server`  确认软件已经成功安装  
`#dpkg -s isc-dhcp-server`  用另一种方式确认成功安装]  

dpkg –unpack package.deb     解开 deb 包的内容  
dpkg -S keyword     搜索所属的包内容  
dpkg –configure package     配置包   