# 1. 如何使用Git

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

### 添加文件
`git add readme.txt`  
**`add`**命令用来添加新文件或者对文件的新修改到暂存区

`git rm readme.txt` 用来添加一个删除文件的修改到暂存区

` git commit -m "wrote a readme file"`  
`git commit` 提交更改,将暂存区的内容提交到版本库  
每一次commit都是一个保存点,可以从这里还原版本库  
`-m` 用来标注提交的说明

## 4. 版本查看及穿梭
### **查看状态**  
`git status` 用来查看当前的状态,例如有哪些文件被修改但是还没提交

### **查看区别**  
`git diff` 用来查看difference,这里比较的是**工作区**和**暂存区**   
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
`git reset commitID test.txt`    
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

## 6. 撤销修改

`git restore <file>`撤销工作区的修改,使这个文件回到暂存区的样子,(`add`或者`commit`的状态)

`git restore --staged <file>`可以把已经`add`的内容撤销,将暂存区的内容撤销掉(unstage),重新放回工作区  

# 使用Github进行远程推送

## 1. 第一次的初始化

本地Git仓库和GitHub仓库之间的传输是通过SSH加密的  

* 创建SSH key
  * 在用户主目录下看有没有`.ssh`目录,没有则输入命令创建
  * `ssh-keygen -t rsa -C "youremail@example.com"`  
  * 在`.ssh`目录中找到<kbd>id_ras.pub</kbd>,就是SSH公钥
* 在github上将自己的SSH KEY公钥添加

## 2. 添加远程版本库

`git remote add <命名远程库> <url>`  
例如  
`git remote add origin  https://github.com/embattled/learnnote.git`  
origin是默认的远程库叫法,也可以自定义

## 3. 推送到远程库

`git push <远程库名称> <分支名称>`
使用  
`git push -u origin master` 命令来第一次推送  
`master`代表推送的是master分支  
`-u` 在将本地分支推送到远程的基础上,还将本地和远程的该分支关联了起来,在以后的推送或者拉去时可以简化命令

在这之后的推送使用  
`git push origin master`即可

如果两个库有不相干的历史记录而无法合并，这时我们可以加上一个参数  
`--allow-unrelated-histories `  
即可成功pull：  
`$ git pull origin master --allow-unrelated-histories`  
这时会可能会提示必须输入提交的信息，默认会打开vim编辑器  

##  4. 从远端克隆一个项目

`git clone git@https://github.com/Embattled/learnnote.git`  
GitHub给出的地址不止一个，还可以用`https://github.com/michaelliao/gitskills.git`这样的地址。实际上，Git支持多种协议，默认的`git://`使用`ssh`，但也可以使用https等其他协议。


# 3. Git的分支管理
## 1. 分支的基础操作
### 创建分支
使用  `git checkout -b <分支名称>`  
`-b`参数代表创建并切换,相当于  
`git branch <分支名称>`新建一个分支  
`git checkout <分支名称>`切换到这个分支  

`git branch`  命令会列出所有分支,并在当前分支前方标识一个 <kbd>*</kbd>  

### 切换分支
由于`checkout`还具有撤销修改的功能,所以防止迷惑性,可以使用更科学的命令 `switch`  
`git switch -c dev` 创建并切换


### 合并一个分支
使用 `git merge <分支名称>` 来将分支的工作成果合并到`master` 分支上

### 删除一个分支
合并完成后,原有的分支就不需要了  
使用 `git branch -d <分支名称>`可以删除一个分支

因为创建、合并和删除分支非常快，所以Git鼓励你使用分支完成某个任务，合并后再删掉分支，这和直接在master分支上工作效果是一样的，但过程更安全。

## 2.解决分支的冲突

当Git无法执行“快速合并”，只能试图把各自的修改合并起来，但这种合并就可能会有冲突  
必须手动解决冲突后再提交

使用`git status`可以告诉我们冲突的文件
Git用<kbd><<<<<<<</kbd>，<kbd>=======</kbd>，<kbd>>>>>>>></kbd>标记出不同分支的内容

用带参数的`git log`也可以看到分支的合并情况

` git log --graph --pretty=oneline --abbrev-commit`  
* `--graph` 可以画分支图
* `--abbrev-commit` 可以简化commit的ID

## 3. 分支管理策略

通常合并分支时,git会在可能的时候使用`Fast forward`模式,这种模式的缺点就是删除分支后就会丢失分支的信息

使用  
`git merge --no-ff` 来强制禁用 `fast forward`模式,这种模式就会在`merge`的时候自动生成新的`commit`

通常`master`分支是用来发布稳定版本的,而`dev`分支是用来保存不稳定版本的  
在此之下才是各个团队工作人员自己的分支

