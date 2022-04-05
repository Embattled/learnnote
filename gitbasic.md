# 1. Git config

**一些基础的Linux命令**
* `mkdir learngit` 创建一个文件夹  
* `cd learngit` 进入文件夹  
* `pwd`  输出当前所在目录 
* `ls -ah` ls用来输出当前文件夹的所有文件及文件夹  -ah用来输出包括被隐藏的全部



git配置文件有三个位置:
1. `/etc/gitconfig` 系统配置文件，对应git config --system
2. `~/.gitconfig` 用户全局配置文件，对应git config --global
3. 项目目录中`.git/config` 仅该项目配置文件，对应git config --local


## 1.1. 自报家门

Git是分布式版本控制系统，所以，每个机器都必须自报家门：你的名字和Email地址  
安装程序完成后需要在命令行输入
```
git config --global user.name "Your Name"
git config --global user.email "email.example.com"

git config -l  # 测试编辑好的机器信息
```

注意`git config`命令的`--global` 参数，用了这个参数，表示你这台机器上所有的Git仓库都会使用这个配置，当然也可以对某个仓库指定不同的用户名和Email地址

## 1.2. 定义Git全局的 .gitignore 文件

创建针对所有库的全局的 .gitignore 文件，可以放在任意位置。然后在使用以下命令配置Git:  
`git config --global core.excludesfile ~/.gitignore`

## 1.3. 更换默认编辑器

`git config --global core.editor` <编辑器, 例如 vim>

# 2. 版本库操作

* `git init` 在当前目录创建 git 内部文件, 使当前路径称为一个 git 项目

* 工作区    : 就是电脑里能看到的目录  
* 暂存区    : 在使用`git add`的时候,就是把文件修改添加到暂存区  
* 隐藏目录  : `.git`不算工作区,而是版本库  

而`git commit`就是把暂存区的内容正式提交到分支上

## 2.1. gitignore 过滤文件
在根目录新建 `.gitignore` 来过滤不想跟踪的文件  
  * 以斜杠`/`结尾表示目录  `mtk/` 过滤整个文件夹  `/mtk/do.c` 过滤某个具体文件  `!/mtk/one.txt` 追踪(不过滤)某个具体文件
  * 以星号`*`通配多个字符  `*.zip` 过滤所有.zip文件  
  * 以问号`?`通配单个字符  
  * 以方括号`[]`包含单个字符的匹配列表  
  * 以叹号`!`表示不忽略(跟踪)匹配到的文件或目录。  
  * 注意： git 对于 .gitignore配置文件是按行从上到下进行规则匹配的  

只追踪特定后缀名的方法, 按顺序在 gitignore 文件种输入下列过滤
1. `*` 过滤所有文件
2. `!*/` 追踪所有文件夹
3. `!*.cpp` 追踪特定后缀名


如果新添加了过滤关键字, 而相关文件已经被推送, 可以用 `--cashed` 命令  
1. `git rm -r --cached .`  点`.` 表示所有文件  
2. 以缓存形式删除所有文件, 并不会真的删除系统中的文件  
3. 再 `git add -A` 提交推送即可


## 2.2. 身兼数职的 checkout

0. 在 git 2.23 版本以前, 对于丢弃工作区的修改
   * git 会提示使用 `git checkout -- file`
   * 事实上 `checkout` 还兼任了切换分支的功能, 这容易导致操作混淆
1. 在 Git 2.23 版本开始引入了两个新的命令 `switch` 和 `restore`
   * 通过直接访问 `man git` 可以查看到最新版本对这 3 个命令的定义
   * `git-checkout(1)   Switch branches or restore working tree files.`
   * `git-switch(1)     Switch branches.`
   * `git-restore(1)                Restore working tree files.`
2. 可以看到两个新的命令完美替换掉了 checkout, 因此应该尽量避免使用 checkout

## 2.3. 文件更改 未commit下的

1. `add` 命令用来添加新文件或者对文件的新修改到暂存区
   * `git add readme.txt`  单独提交一个文件  
   * `git add .` 会监控工作区的状态树, 提交文件修改(modified), 以及新文件(new), 但不包括被删除的文件
   * `git add -u[pdate] ` 仅监控已经被add的文件(tracked file) 他会将被修改的文件提交到暂存区, 不会提交新文件 (untracked file), 提交被修改(modified)和被删除(deleted)文件, 不包括新文件(new)
   * `add .` 和 `add -u` 有着微妙的区别, 在于对新文件和删除文件的态度上
   * `git add -A[ll] ` 是上面两个功能的合集 (提交所有变化)

2. `rm` 用于 git 追踪的删除文件
   * `git rm readme.txt`   用来添加一个删除文件的修改到暂存区
   * `git rm --cached txt` 用于缓存的删除, 即不更改文件

3. `mv`  追踪下的 重命名 移动文件, 直接在文件系统重命名或移动会导致无法跟踪 
   * `git mv <old_name> <new_name>`来重命名文件  
   * `git mv -f * *` 来进行一个强制重命名, 这会覆盖原本名为`new_name`的文件  
   * 使用 `git mv string.c src/` 来将一个文件移入文件夹,使用 `/` 来代表目录

4. `status` 状态查看, 主要查看有哪些文件被修改但是还没提交
   * `git status` 

5. `diff` 直接在命令行里查看改动
   *  `git diff` 默认比较的是工作区和暂存区, 即如果刚刚 `git add -A` 了后是看不到改动的
   *  `git diff [<options>] [<commit> [<commit>]] [--] [<path>...]` 完整命令
      *  输入一个 commit 则是工作区和参数 commit
      *  输入两个 commit 则是参数 commit 之间的比较
      *  `--path` 用来指定文件进行改动查看

6. `restore` 撤销修改
   * 默认是回到将工作区的内容还原为到暂存区, 即舍弃所有更改
   * 也可以使  版本库中的文件覆盖暂存区的文件, 即回退 `add`
   * `git restore [<options>] [--source=<branch>] <file>...` 完整使用方法
   * `git restore <file>`撤销工作区的修改, 使这个文件回到暂存区的样子 (上一次`add`或者`commit`的状态)
   * `git restore --staged <file>` 将暂存区的内容(已经add的)撤销掉 (unstage)
     * 撤销掉 add 的内容还可以使用 `git reset HEAD <files>`, 是冗余的命令


## 2.4. 版本提交与回退

1. `commit` 将暂存区的内容提交到版本库, 每一次commit都是一个保存点,可以从这里还原版本库  
   * `git commit -c <commit>` 懒人代码, 直接复制参数 commit 的 log msg
   * `git commit -C <commit>` 复制参数 commit 的 log msg 并进入编辑界面
   * ` git commit -m "wrote a readme file"`  
   * 如果commit注释写错了，只是想改一下注释，只需要：
   * `git commit --amend` 用于提交后发现忘记了暂存某些需要的修改
     * 相当于 `git reset --soft HEAD^` 再重新提交 `git commit -c ORIG_HEAD`
     * 使用示例
```sh
$ git commit -m 'initial commit'
$ git add forgotten_file
$ git commit --amend
```
  
2. `log` 查看 commit 的历史
   * `git log` 用来查看全部版本的更新时间以及说明
   * 后面加上 `--pretty=oneline` 使得输出只有一行,更简洁

3. `reflog` 查看 每一个操作的 log 历史
   * 相比较于 `log`, 记录的操作更加详细, 主要用于 restore

4. `reset` 版本穿梭, 退回指定版本, 即撤销 commit
   * Git的版本回退仅仅是更改一个名为`HEAD`的内部指针,HEAD指向的版本就是当前的版本
   * `git reset HEAD <files>` 回到最新版本, 是一个冗余的命令
     * 主要用于把暂存区的修改回退到工作区
     * 该命令相当于 `git restore --staged <file_name>`
   * `git reset HEAD^` 回到当前版本的上一个版本
     * 可以填入版本号,版本号不必写全
     * 可以填入 `HEAD^`, 代表上一个版本, `^` 的个数代表上 N 个版本
     * `HEAD~N` , N是数字, 代表返回上N个版本, 用于不适合用 `^` 表示的过早的版本
   * 对工作区的影响参数
     * `--hard` , 删除工作空间改动的代码, 即代码文件也会被返回
     * `--mixed`, 默认的参数, 不删除工作区代码, 撤销 commit 和 add
     * `--soft` , 不改动代码, 只撤销 commit, git add 也不会撤销
   * 重做(撤销刚才的撤销动作):
     * 在命令行窗口还没关掉的时候,寻找`commit id` 可以根据id来回到具体的哪一个版本  
     * `git reset --hard 1094a`  
     * `git reset commitID test.txt`    
     * 若忘记了想要退回的版本id,使用 `git reflog`  



# 3. 云
## 3.1. 使用Github

### 3.1.1. 第一次的初始化

本地Git仓库和GitHub仓库之间的传输是通过SSH加密的  

* 创建SSH key
  * 在用户主目录下看有没有`.ssh`目录,没有则输入命令创建
  * `ssh-keygen -t rsa -C "youremail@example.com"`  
  * 在`.ssh`目录中找到<kbd>id_ras.pub</kbd>,就是SSH公钥
* 在github上将自己的SSH KEY公钥添加

### 3.1.2. 添加远程版本库

使用`git remote`查看远程库的名称  
使用`git remote -v`可以显示包括url的详细信息  

**添加一个新的远程库**
`git remote add <命名远程库> <url>`  
例如  
`git remote add origin  https://github.com/embattled/learnnote.git`  
origin是默认的远程库叫法,也可以自定义



### 3.1.3. 推送到远程库

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

###  3.1.4. 从远端克隆一个项目

`git clone git@github.com:Embattled/learnnote.git`  
GitHub给出的地址不止一个，还可以用`https://github.com/Embattled/learnnote.git`这样的地址。实际上，Git支持多种协议，默认的`git://`使用`ssh`，但也可以使用https等其他协议。

使用ssh的话，在github中保存公钥了后不用输入密码  
使用http的话需要每次输入密码  

### 3.1.5. 从远端更新代码

克隆后,若要更新代码,使用git pull

## 3.2. 使用gitlab




# 4. Git的分支管理

`man git` 的 分支相关的命令说明
 * switch              Switch branches.
 * branch              List, create, or delete branches.
 * merge               Join two or more development histories together.
 * (deprecate)checkout Switch branches or restore working tree files.

## 4.1. 分支的基础操作

1. `branch` 分支管理
使用  `git checkout -b <分支名称>`  
`-b`参数代表创建并切换,相当于  
`git branch <分支名称>`新建一个分支  
`git checkout <分支名称>`切换到这个分支  

`git branch`  命令会列出所有分支,并在当前分支前方标识一个 <kbd>*</kbd>  

2. `switch` 切换分支
由于`checkout`还具有撤销修改的功能,所以防止迷惑性,可以使用更科学的命令 `switch`  
`git switch -c dev` 创建并切换



### 4.1.4. 删除一个分支
合并完成后,原有的分支就不需要了  
使用 `git branch -d <分支名称>`可以删除一个分支

因为创建、合并和删除分支非常快，所以Git鼓励你使用分支完成某个任务，合并后再删掉分支，这和直接在master分支上工作效果是一样的，但过程更安全。

## 4.2. 解决分支的冲突

1. `merge` 合并一个分支
使用 `git merge <分支名称>` 来将分支的工作成果合并到`master` 分支上
当Git无法执行`快速合并`，只能试图把各自的修改合并起来，但这种合并就可能会有冲突  
必须手动解决冲突后再提交

使用`git status`可以告诉我们冲突的文件
Git用<kbd><<<<<<<</kbd>，<kbd>=======</kbd>，<kbd>>>>>>>></kbd>标记出不同分支的内容

用带参数的`git log`也可以看到分支的合并情况

` git log --graph --pretty=oneline --abbrev-commit`  
* `--graph` 可以画分支图
* `--abbrev-commit` 可以简化commit的ID

## 4.3. 分支管理策略

通常合并分支时,git会在可能的时候使用`Fast forward`模式,这种模式的缺点就是删除分支后就会丢失分支的信息

使用  
`git merge --no-ff` 来强制禁用 `fast forward`模式,这种模式就会在`merge`的时候自动生成新的`commit`,因此还应加上`-m`来描述这个提交  
`git merge --no-ff -m "merge with no-ff" dev`


通常`master`分支是用来发布稳定版本的,而`dev`分支是用来保存不稳定版本的  
在此之下才是各个团队工作人员自己的分支

开发一个新的feature,最好新建一个分支  
如果要丢弃一个没有被合并过的分支,可以通过  
`git branch -D <name>`强行删除  

### 4.3.1. 团队协作
在团队中,对于dev分支,同事的的最新提交和你试图推送的提交有冲突  

对于一个分支,也需要指定本地与远端的链接
`git branch --set-upstream-to dev origin/dev`  
此时会提示  
`Branch 'dev' set up to track remote branch 'dev' from 'origin'.`  

链接好后即可pull,   使用`git pull`将远端的分支抓取下来,在本地合并,解决冲突后再提交  


## 4.4. Bug处理以及现场保存

git提供了一个保存现场的功能  
`git stash`  
此时再用`git status`查看工作区就是干净的状态

建立新的分支并处理好bug后**还原现场**

`git switch dev`  切换回当前工作的分支  

有两个办法恢复现场
1. `git stash apply`恢复,但是stash的内容不删除,需要手动`git stash drop`删除
2. `git stash pop`恢复的同时也将stash的内容删除了  

使用`git stash list`命令查看现场列表  
在多次stash后,可以指定要恢复的`stash`  
`git stash apply stash@{0}

### 4.4.1. 小结 

* 查看远程库信息，使用`git remote -v`

* 本地新建的分支如果不推送到远程，对其他人就是不可见的

* 从本地推送分支，使用`git push origin branch-name`,如果推送失败，先用`git pull`抓取远程的新提交

* 在本地创建和远程分支对应的分支，使用`git checkout -b branch-name origin/branch-name`,本地和远程分支的名称最好一致

* 建立本地分支和远程分支的关联，使用`git branch --set-upstream <branch-name> <origin/branch-name>`

* 从远程抓取分支，使用`git pull`，如果有冲突，要先处理冲突。


### 4.4.2. Bug处理

对于一个已经存在于所有分支的bug,git提供了一个`cherry-pick`,能够将一个特定的分支复制到当前分支  
`git switch dev`
`git cherry-pick <ID>`  
此时git会自动给dev分支进行一次提交,该commit会获得一个不同的ID, 使用该命令可以省去重复修复bug的过程


