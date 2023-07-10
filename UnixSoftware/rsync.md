# Remote Sync  (rsync)

通过网络快速同步多台主机之间的文件和目录, 名利用差分算法减少数据传输量  

rsync 有两种认证方式, 一种是 rsync-daemon, 另一种是 ssh  
* rsync-daemon 方式是主要的认证方式, 会默认监听 TCP 的 873 端口, 需要客户端服务器都安装 rsync   


rsync 的软件不区分客户端和服务器端  