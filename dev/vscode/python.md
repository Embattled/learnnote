# 在vscode上使用python

## 1. 安装
1. python的安装
2. 在vscode中安装python拓展
3. 确认
   1. windows中 在命令行中输入 `py -3 --version`  可以确定版本
   2. linux中   在命令行中输入 `python3 --version`  可以确定版本
   3. 在终端中输入 `py -0` 可以查看系统中有的版本以及当前解释器选择的版本(当前版本前面有个 `*`)
   4. 在vscode 的命令行输入 `Python: Select Interpreter` 来选择当前解释器
4. 使用code打开文件
    * `mkdir` 创建一个新文件夹
    * `cd` 到其中
    * `code .` 命令在当前文件夹打开vscode
5. 运行的三种方法
   1. 选择一个`.py`文件,点击右上方的`▶`来运行这个文件
   2. 在窗口的任意位置右键,选择`Run Python File in Terminal` 
   3. 运行部分代码,使用<kbd>Shift</kbd>+<kbd>Enter</kbd> 或者右键选择`Run Selection/Line in Python Terminal` 
6. REPL 终端
   在vscode命令行输入  `Python: Start REPL` 来进入 REPL 交互式终端

## 2. 调试

1. 设置断点
   * 使用 <kbd>F9</kbd> 或者点击左边的行号来设置断点,表现为一个红点
2. 运行调试
   * 使用 <kbd>F5</kbd> 开始调试
   * 在对一个文件第一次运行调试时,会出现窗口来选择调试配置
   * 停顿行会以黄色标识出,在左侧会有变量显示区
3. 调试工具条
   * continue (F5)
   * step over (F10)
   * step into (F11)
   * step out (Shift+F11)
   * restart (Ctrl+Shift+F5)and 
   * stop (Shift+F5)
4. 调试控制台
   * 使用调试控制台可以手动精确的查看变量

   
## Linting

在vscode的命令行输入 `Python: Select Linter` 选择一个Linter,也可以禁用所有的Linter

## Environment
python 支持对某一个文件夹配置单独的运行环境,包括包以及指定解释器  
在vscode中新建一个虚拟环境并进行配置

* windows :`py -3 -m venv .venv`
* mac os/linux: `python3 -m venv .venv`



