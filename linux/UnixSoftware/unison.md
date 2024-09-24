
# 1. Unison 双向同步软件

官方网站以 github 形式维护
https://github.com/bcpierce00/unison

# 2. Reference Guide


## 2.1. Running Unison - 启动方法




## 2.2. The .unison Directory

## Archive Files


## 2.3. Preferences 


Basic options:
* How to sync:
  * `-batch` batch mode: ask no questions at all, 无交互的同步, 不进行任何询问 
* How to sync (text interface (CLI) only):
  * `-auto` 通过 CLI 操作 unison 的必备选项, automatically accept default (nonconflicting) actions



Advanced options:
* `-prefer xxx` choose this replica’s version for conflicting changes, 指定发生冲突时候的高优先级对象
  * -prefer root  : 以本地的为优先 (除了那些被标记为 merge 的目录)
  * -prefer newer/older : 另两种特殊标记
  * `preferpartial` 会覆盖该 flag


Other:
* `-log` record actions in logfile (default true), 将日志存储为文件
* `-logfile` xxx logfile name, 指定日志文件路径
