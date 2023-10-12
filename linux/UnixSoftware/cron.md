# cron - daemon to execute scheduled commands (Vixie Cron)

cron by Paul vixie 



# crontab - maintain crontab files for individual users (Vixie Cron)


```sh
crontab [ -u user ] file
crontab [ -u user ] [ -i ] { -e | -l | -r }

# -u user 用户指定要操作的用户名, 一般只有 sudo 用户才能(才需要) 进行指定
# 一般用户省略即可, 即对自己的 crontab 进行操作  
```

crontab 是一个用于 install, deinstall 或者 list 用于驱动 cron daemon 的 tables 的软件, 每一个用户可以定义其自己的crontab.  


文件系统
* crontab tables 是存储在 `/var/spool/cron/crontabs` 里的, 这些文件不应该被直接编辑
* 如果`/etc/cron.allow` 文件存在的话, 则该文件 用于管理可以使用 crontab 命令的 用户, 在使用前确保自己被加入到该文件中 (每个用户一行)
* 同理, 如果 `/etc/cron.deny` 存在的话, 则需要确保自己不在该文件中
* 如果 `allow` 和 `deny` 文件都不存在, 则根据 on site-dependent configuration parameters, 要么所有用户都可以使用, 要么只有 super user 可以使用


CLI 用法
* `crontab -e`     : 在编辑器中编辑当前的 (即当前 user的)  crontab, 并在结束编辑后 crontab 会被自动安装
* `crontab -r`     : 移除当前 crontab, `crontab -i` 用于在删除之前让用户进行确认
* `crontab -l`     : 打印对当前 crontab 在 std output 中

## crontab 语法


*/5 用于每 5 (分钟, 小时) 执行一次


