# 1. 终端复用器

tmux, screen, byobu, nohup
  
## 2.5. 快捷键总结

总结 Ctrl+b 模式的快捷键  

全局:
* `$`   : 重命名当前 session
* `d`   : detach
* `[`   : 进入浏览输出模式 `q` 退出
* `PgUp`: 同样进入浏览输出模式

会话:


窗口 :                

窗格:
* `%`   : 左右窗格划分
* `"`   : 上下窗格划分
* `x`   : 关闭当前窗格
* `arrow`: 窗格光标移动
* `{ }` : 当前窗格与上/下一个窗格交换位置 


# 3. Windows Terminal

wt.exe  

微软独立于 cmd 和 Powershell 的单独终端程序, 同时具有一定的窗口分割功能

* Alt   + Enter     : 全屏显示
* Ctrl  + ,         : 打开 setting.json

## 3.1. 窗口分割操作

基于当前光标的窗口进行分割 (新建)
* shift + Alt + `-` 水平
* shift + Alt + `+` 垂直
* Alt + 方向键       分割窗口中光标移动
