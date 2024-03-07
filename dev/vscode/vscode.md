# Visual Studio Code

vscode 完整文档:
https://code.visualstudio.com/docs


# User Guide

VScode 内建了一套完整的变量系统, 通过 JSON 即可对 vscode 的各种动作模式进行配置和指定

# C++

在 Vscode 中配置 C++ IntelliSense, 并进行 debug, 需要的 microsoft 官方插件包括  
* C/C++

Vscode 本身是不包括 compiler 和 debugger 的




在vscode的命令行输入 `Configure Default Build Task` 选择`g++` 配置  
这将会在工作区的 `.vscode` 下生成 `tasks.json` 文件  

json属性的说明：
* `command` : 编译调用的命令, 应该指向的是系统编译器的路径
* `args`    : 调用命令(g++) 时自动输入的参数, 默认会在源文件路径下生成同名的可执行文件
* `label`   : 该操作的名字, 可以自己指定来更容易识别
* `isDefault": true`: 指定是不是编译的默认操作 <kbd>Ctrl+Shift+B</kbd>. 
  This property is for convenience only; if you set it to false, you can still run it from the Terminal menu with `Tasks: Run Build Task`.

通过配置该json包, 可以进行自定义编译操作
* 输入多个文件进行编译 `"${workspaceFolder}\\*.cpp"`
* 自定义输出文件 `"${workspaceFolder}\\myProgram.exe"`


* 查看更多vscode配置文件中可以使用的[变量](https://code.visualstudio.com/docs/editor/variables-reference)


## 1.3. Configure debugging

同编译的 `task.json` 文件一样  
第一次会弹窗选择要运行的调试器,选择`gbd/lldb`,之后会在工作区的`.vscode`下新建一个 `launch.json`文件  

* 在文件中按下 <kbd>F5</kbd> 打开调试  
* 使用快捷键<kbd>F9</kbd> 添加断点  

* `program`     : 需要指定要去 debug 哪一个文件
* `miDebuggerPath` : 需要在这里指定 gdb 的路径  
* `stopAtEntry` : 指定为 true 的时候程序会在 main 处自动添加一个 breakpoint  

使用变量和监视窗口可以方便的在程序暂停时查看变量  


## 1.4. Settings

* 在控制台输入 `C++ edit config` 可以调出专门对 vscode C++ extension 的配置
* 会创建一个 `c_cpp_properties.json` 的文件, 用于指定该工作区对于 C++拓展的配置
* such as the path to the compiler, include paths, C++ standard (default is C++17), and more

在自定义项目的时候, 如果 header 不在工作路径或者 STL, 只需要修改该配置文件的 `Include path` 即可  

## Configure IntelliSense



# Remote 


## 在CODE中使用WSL

要用CODE打开WSL中的文件夹:  
`code --remote wsl+<distro name> <path in WSL>`  
for example: `code --remote wsl+Ubuntu /home/jim/projects/c`  

