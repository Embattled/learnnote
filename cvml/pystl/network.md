# Networking and Interprocess Communication

* 这个模块组存储了所有网络设备相关的内容, 包括内部的处理函数  


# socket - Low-level networking interface¶

* 该模组在 all modern Unix systems, Windows, MacOS, and probably additional platforms. 都可用
* 非常低级的模组


## 网络相关的附加函数


### socket.get* 类


* gethostname()     返回当前 python 运行环境的 hostname, 不需要参数