# 如何使用Git以及Github

## 1. 安装Git

### 在Windows上安装

安装程序完成后需要在命令行输入
```
git config --global user.name "Your Name"
git config --global user.email "email.example.com"
```
Git是分布式版本控制系统，所以，每个机器都必须自报家门：你的名字和Email地址。

注意`git config`命令的`--global` 参数，用了这个参数，表示你这台机器上所有的Git仓库都会使用这个配置，当然也可以对某个仓库指定不同的用户名和Email地址

## 2. 创建一个版本库

**一些基础的Linux命令**
`mkdir learngit` 创建一个文件夹  
`cd learngit` 进入文件夹  
`pwd`  输出当前所在目录  
`ls -ah` ls用来输出当前文件夹的所有文件及文件夹  -ah用来输出包括被隐藏的全部

**新建一个版本库**
`git init`

## 3. 添加文件及提交
`git add readme.txt`  
**`add`**命令用来添加新文件到仓库

` git commit -m "wrote a readme file"`  
`git commit` 提交更改  
每一次commit都是一个保存点,可以从这里还原版本库  
`-m` 用来标注提交的说明

## 4. 版本查看及穿梭
### **查看状态**  
`git status` 用来查看当前的状态,例如有哪些文件被修改但是还没提交

### **查看区别**  
`git diff` 用来查看difference,这里比较的是**暂存区**和**工作区**   
注意!显示的格式是<u>**Unix通用的diff格式**</u>  

`git diff HEAD --readme.txt` 可以查看**工作区**和**版本库**里面最新版本的区别

### **查看版本**
`git log` 用来查看最新三个版本的更新时间以及说明

后面加上 `--pretty=oneline` 使得输出只有一行,更简洁

### **版本穿梭**
要想退回上一个版本,使用命令  
`git reset --hard HEAD`

想要重做(撤销刚才的撤销动作),在命令行窗口还没关掉的时候,寻找`commit id` 可以根据id来回到具体的哪一个版本  
`git reset --hard 1094a`  
版本号不必写全

Git的版本回退仅仅是更改一个名为`HEAD`的内部指针,HEAD指向的版本就是当前的版本

若忘记了想要退回的版本id,使用  
`git reflog`  
会显示每一次的命令

## 5. 工作区和暂存区
**工作区**就是电脑里能看到的目录  
**隐藏目录**`.git`不算工作区,而是版本库  

**暂存区** 在使用`git add`的时候,就是把文件修改添加到暂存区  
而`git commit`就是把暂存区的内容提交到分支上