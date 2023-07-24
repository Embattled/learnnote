# GNU Mcron

在查找 cron 的使用方法的时候, 因为更想使用 GNU 的软件, 一搜索还真有  

    The GNU package mcron (Mellor's cron) is a 100% compatible replacement for Vixie cron. It is written in pure Guile, and allows configuration files to be written in scheme (as well as the POSIX crontab format) for infinite flexibility in specifying when jobs should be run. Mcron was designed and written by Dale Mellor. 

原本的 cron 作者就是 Paul Vixie, mcron 就是 cron 的完整替代, 100% 的兼容性?  

    

## introduction

original cron:
     The original idea was to have a daemon that wakes up every minute, scans a set of files under a special directory, and determines from those files if any shell commands should be executed in this minute. 

