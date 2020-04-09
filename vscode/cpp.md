# 在vscode上配置gcc编译环境
## 1. 安装与第一次编译与调试
### 1. 安装mingw和vscode的c++拓展
`MinGW`,是`Minimalist GNUfor Windows`的缩写  

下载并安装MinGW后,将`/bin`添加到PATH  
例 `c:\mingw-w64\x86_64-8.1.0-win32-seh-rt_v6-rev0\mingw64\bin.`

在命令行输入代码检查  
```
g++ --version
gdb --version
```

### 2. 配置编译环境
在vscode的命令行输入 `Configure Default Build Task` 选择`g++` 配置  
这将会在工作区的 `.vscode` 下生成 `tasks.json` 文件  
* `args`: specifies the command-line arguments that will be passed to g++
* `label`: what you will see in the tasks list; you can name this whatever you like.
* `isDefault": true`: specifies that this task will be run when you press <kbd>Ctrl+Shift+B</kbd>. This property is for convenience only; if you set it to false, you can still run it from the Terminal menu with `Tasks: Run Build Task`.
* 查看更多vscode配置文件中可以使用的[变量](https://code.visualstudio.com/docs/editor/variables-reference)

### 3. 编译

使用按键 <kbd>Ctrl+Shift+B</kbd> 开始编译  
将会在下方终端里显示是否成功的信息  


### 4. 调试

在文件中按下 <kbd>F5</kbd> 打开调试  
使用快捷键<kbd>F9</kbd> 添加断点  
第一次会弹窗选择要运行的调试器,选择`gbd/lldb`,之后会在工作区的`.vscode`下新建一个 `launch.json`文件  
* `program` : specifies the program you want to debug.
* `stopAtEntry` : value to true to cause the debugger to **stop** on the `main` method when you start debugging.

使用变量和监视窗口可以方便的在程序暂停时查看变量  

