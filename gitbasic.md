# 1. Git config

**一些基础的Linux命令**
* `mkdir learngit` 创建一个文件夹  
* `cd learngit` 进入文件夹  
* `pwd`  输出当前所在目录 
* `ls -ah` ls用来输出当前文件夹的所有文件及文件夹  -ah用来输出包括被隐藏的全部



git配置文件有三个位置:
1. `/etc/gitconfig` 系统配置文件,            对应 `git config --system`
2. `~/.gitconfig` 用户全局配置文件,          对应 `git config --global`
3. 项目目录中`.git/config` 仅该项目配置文件,  对应 `git config --local`


## 1.1. 自报家门

Git是分布式版本控制系统，所以，每个机器都必须自报家门：你的名字和Email地址  
安装程序完成后需要在命令行输入
```
git config --global user.name "Your Name"
git config --global user.email "email.example.com"

git config -l  # 测试编辑好的机器信息
```

注意`git config`命令的`--global` 参数，用了这个参数，表示该用户在这台机器上所有的Git仓库都会使用这个配置，当然也可以对某个仓库指定不同的用户名和Email地址

## 1.2. 定义Git全局的 .gitignore 文件

创建针对所有库的全局的 .gitignore 文件，可以放在任意位置。然后在使用以下命令配置Git:  
`git config --global core.excludesfile ~/.gitignore`

## 1.3. 更换默认编辑器

`git config --global core.editor` <编辑器, 例如 vim>

# 2. 离线操作

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
   * `git-restore(1)    Restore working tree files.`
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

6. `stash` 现场保存, 用于非 commit 下的工作进度保存
   * `git stash [push]` : 立刻将工作区的改动存于后台, 并将工作区的文件还原到 HEAD 状态, 此时可以进行切换分支等操作
     * `-m <message>` : 给 stash 添加描述
   * `git stash list` : 列出所有 stash 
   * `git stash show` : 打印 stash 与 commit back 的 diff
   * `git stash (pop | apply) ` :
     * apply 恢复,但是stash的内容不删除
     * pop   恢复的同时也将stash的内容删除了  
   * `git stash drop` : 删除 stash 

7. `restore` 撤销修改
   * 默认是回到将工作区的内容还原为到暂存区, 即舍弃所有更改
   * 也可以使  版本库中的文件覆盖暂存区的文件, 即回退 `add`
   * `git restore [<options>] [--source=<branch>] <file>...` 完整使用方法
   * `git restore <file>`撤销工作区的修改, 使这个文件回到暂存区的样子 (上一次`add`或者`commit`的状态)
   * `git restore --staged <file>` 将暂存区的内容(已经add的)撤销掉 (unstage)
     * 撤销掉 add 的内容还可以使用 `git reset HEAD <files>`, 是冗余的命令


使用`git stash list`命令查看现场列表  
在多次stash后,可以指定要恢复的`stash`  
`git stash apply stash@{0}

## 2.4. 版本提交与回退

1. `commit` 将暂存区的内容提交到版本库, 每一次commit都是一个保存点,可以从这里还原版本库  
   * `git commit -c <commit>` 懒人代码, 直接复制参数 commit 的 log msg
   * `git commit -C <commit>` 复制参数 commit 的 log msg 并进入编辑界面
   * ` git commit -m "wrote a readme file"`  
   * 如果commit注释写错了，只是想改一下注释，只需要：
   * `git commit --amend` 用于提交后发现忘记了暂存某些需要的修改, 或者更改最后一次提交的信息
     * 相当于 `git reset --soft HEAD^` 再重新提交 `git commit -c ORIG_HEAD`
     * 使用示例
```sh
$ git commit -m 'initial commit'
$ git add forgotten_file
$ git commit --amend
```
  
2. `log` 查看 commit 的历史, 还可以画图来体现项目的 merge 历史
   * `git log` 用来查看全部版本的更新时间以及说明
   * 后面加上 `--pretty=oneline` 使得输出只有一行,更简洁
   * `--graph` 可以画分支图
   * `--abbrev-commit` 可以简化commit的ID
   * `git log --graph --pretty=oneline --abbrev-commit`  


3. `reflog` 查看 每一个操作的 log 历史
   * 相比较于 `log`, 记录的操作更加详细, 主要用于 restore

4. `reset` 版本穿梭, 退回指定版本, 即撤销 commit
   * Reset current HEAD to the specified state.
   * Git reset 的版本回退仅仅是更改一个名为`HEAD`的内部指针,HEAD指向的版本就是当前的版本
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

5. `revert` Revert some existing commits.
   * 同 reset 最大的不同是, reset 是真正意义上的版本穿梭, revert 则是礼貌的回退
     * revert是用一次新的commit来回滚之前的commit, 此次提交之后的commit都会被保留
     * reset是版本库回退到某次 commit , 此commit id之后的修改都会被删除
   * revert 的真正用于是 删除某一个版本的所有改动
     * 在某个版本添加或删除的文件将被复原
     * 对于持续更改的文件, 很容易引起冲突, 需要手动消除冲突

## 2.5. commit message

良好的 Message 格式有助于团队理解  

`type[(scope)]: Capitalized short summary`
* type: 本次提交的类型
  * 按照是否依次往下排查, 如果有大改动可以在末尾增加 `!`
  * revert : 回退之前的提交
  * chore  : 与版本release相关, 或者 submodule subtree 相关的杂项更新
  * ux     : 用户体验改善, UI界面, CLI 相关
  * feat   : 新功能追加
  * fix    : 修正既存功能
  * perf   : 优化既存功能性能
  * test   : 只修改了测试代码
  * docs   : 只修改了文档
  * style  : 代码风格修改, 空行之类的
  * refactor : 代码重构
  * ci     : 与 Github Action 相关的配置文件修改
  * build  : 编译环境, 配置文件修改
  * chore  : github 设置
* scope (可选) 用于指定更改的模组, 文件, 机能等
* Message body:
  * 首字母大写
  * 命令型英文
  * 不写句号
  * 省略 the
  * 需要多行进行详细说明的情况下, 第二行留空行

# 3. 云

## 3.1. 远程版本库

1. `remote` 命令用于管理和远程相关的事物
   * 使用`git remote`查看远程库的名称  
   * 使用`git remote -v`可以显示包括url的详细信息  
   * 使用`git remote add <命名远程库> <url>`  添加一个远程库
     * `git remote add origin  https://github.com/embattled/learnnote.git`  
     * origin是默认的远程库叫法,也可以自定义

2. `push` 命令用于推送所有 commit 到远程版本库
   * `git push <远程库名称> <分支名称>`
   * `-f  --force` 参数用于强制推送, 可以用于撤销已经 push 的 commit
   * `-u --set-upstream`   set upstream for git pull/status
     * 一般用于执行第一次推送
     * `git push -u origin master` 命令来第一次推送  
     * `-u` 在将本地分支推送到远程的基础上,还将本地和远程的该分支关联了起来,在以后的推送或者拉去时可以简化命令
     * 在这之后的推送使用  `git push origin master`即可
     * 该配置在 branch 上有必要

3. `clone` 用于克隆版本库
   * `git clone git@github.com:Embattled/learnnote.git`  
   * `-b, --branch <branch>` 指定要克隆的分支

4. `fetch` 用于从远端拉取代码
   * Download objects and refs from another repository.
   * `git fetch [<options>] [<repository> [<refspec>...]]` or `git fetch [<options>] <group>`
   * `git fetch origin master` 为默认操作
   * `git fetch origin master:tmp` 用 冒号来指定下载的目标分支, 默认是 `远端名/分支名`

5. `pull` 用于懒人的拉取代码, 会自动merge到本地工作区上
   * Fetch from and integrate with another repository or a local branch.
   * 相当于先 fetch 再 `git merge origin/master`
     * 先从远程的origin的master主分支下载最新的版本到origin/master分支上
     * 然后比较本地的master分支和origin/master分支的差别并合并
   * 实际使用中, 用 fetch 更加安全, 因为在中间可以更精细化的比较什么改动被 fetch 下来了
     * `git fetch origin master:tmp`
     * `git diff tmp `
     * `git merge tmp`

6. `rebase` 非记录分支的 merge
   * 在团队协作的时候, 如果总使用 `pull` 或者 `merge` 来拉取更改, 会因为有许多 merge 导致分支图上不是一条直线, 有许多无意义的分支
   * rebase 用于将最新更改合并到当前工作区中, 但是不创建 commit
   * `git pull --rebase` 能够实现 `get fetch + git rebase` 的作用
   * 对于合并时候的冲突
     * 文件里解决冲突
     * `git add `
     * `git rebase --continue`
   * `git rebase --continue | --abort | --skip | --edit-todo`
   * `git rebase -i` 交互式进行 rebash

## 3.2. 使用Github



本地Git仓库和GitHub仓库之间的传输是通过SSH加密的  

* 创建SSH key
  * 在用户主目录下看有没有`.ssh`目录,没有则输入命令创建
  * `ssh-keygen -t rsa -C "youremail@example.com"`  
  * 在`.ssh`目录中找到<kbd>id_ras.pub</kbd>,就是SSH公钥
* 在github上将自己的SSH KEY公钥添加



GitHub给出的地址不止一个，还可以用`https://github.com/Embattled/learnnote.git`这样的地址。实际上，Git支持多种协议，默认的`git://`使用`ssh`，但也可以使用https等其他协议。

使用ssh的话，在github中保存公钥了后不用输入密码  
使用http的话需要每次输入密码  

# 4. Git的分支管理

`man git` 的 分支相关的命令说明
 * switch              Switch branches.
 * branch              List, create, or delete branches.
 * merge               Join two or more development histories together.
 * (deprecate)checkout Switch branches or restore working tree files.

- 通常`master`分支是用来发布稳定版本的
- `dev`分支是用来保存不稳定版本的  
- 在此之下才是各个团队工作人员自己的分支
- 开发一个新的feature,最好新建一个分支  

## 4.1. 分支的团队命名

同 commit message 一样, 合理的命令可以增加团队配合  
`type/module/snake_case_short_summary`  

* type : 分支的任务类型, 参考[commit message里的type说明](#25-commit-message)
* module : 要修改的文件名, 或者广泛一点的模组名
* snake_case_short_summary: 
  * 和具体的实现内容有关
  * 和最终的 PR 题目一一对应
  * 用下划线连接多个小写单词
  * 如果对应的是一个 issue, 则直接写成 issue 编号也行


## 4.2. 分支的基础操作



1. `branch` 分支管理
   * 默认会打印所有本地分支 并在当前分支前方标识一个 <kbd>*</kbd>  
     * `git branch [<options>] [-r | -a] [--merged | --no-merged]`
   * 创建分支
     * `git branch [<options>] [-l] [-f] <branch-name> [<start-point>]`
     * `-f` 用于覆盖的创建分支
   * 重命名 `-m -M`  复制 `-c  -C`  `[<old-branch>] <new-branch>`
     * 使用 `-m/c` 对一个已存在的分支进行重命名, 或复制出一个新的分支
     * 大写的 字母 用于覆盖目标名称原本的分支
   * 删除 `-d -D`
     * ` git branch [<options>] [-r] (-d | -D) <branch-name>...`
     * 因为创建、合并和删除分支非常快，所以Git鼓励你使用分支完成某个任务，合并后再删掉分支，这和直接在master分支上工作效果是一样的，但过程更安全。
     * 使用 `git branch -d <分支名称>`可以删除一个分支
     * `-D` 命令用于强制删除一个还没有被 merge 的分支
   * 链接远程分支
     * 对于一个分支,也需要指定本地与远端的链接
     * `git branch --set-upstream-to dev origin/dev`  


2. `switch` 切换分支  不建议使用 `checkout` 命令
   * `git switch [<options>] [<branch>]`
   * `git switch -c dev` 创建并切换, 省去 `branch` 命令, 同理 `-C` 也是覆盖的创建


## 4.3. 分支合并


1. `merge` 合并一个分支
   * `git merge [<options>] [<commit>...]`
   * 出现冲突时:
     * `git merge --abort` 用于放弃本次合并
     * `git merge --continue` 用于手动处理好冲突后再次合并
   * 使用 `git merge <分支名称>` 来将分支的工作成果合并到`master` 分支上
   * 分支本身也会作为一个 commit 被记录





当Git无法执行`快速合并`，只能试图把各自的修改合并起来，但这种合并就可能会有冲突  
必须手动解决冲突后再提交

使用`git status`可以告诉我们冲突的文件
Git用<kbd><<<<<<<</kbd>，<kbd>=======</kbd>，<kbd>>>>>>>></kbd>标记出不同分支的内容


## 4.4. 分支管理策略

通常合并分支时,git会在可能的时候使用`Fast forward`模式,这种模式的缺点就是删除分支后就会丢失分支的信息

使用  
`git merge --no-ff` 来强制禁用 `fast forward`模式,这种模式就会在`merge`的时候自动生成新的`commit`,因此还应加上`-m`来描述这个提交  
`git merge --no-ff -m "merge with no-ff" dev`


### 4.4.1. Bug处理

对于一个已经存在于所有分支的bug,git提供了一个`cherry-pick`,能够将一个特定的分支复制到当前分支  
`git switch dev`
`git cherry-pick <ID>`  
此时git会自动给dev分支进行一次提交,该commit会获得一个不同的ID, 使用该命令可以省去重复修复bug的过程


# 5. Git 的标签管理 tag

tag是git版本库的一个标记, 指向某个 commit
* branch 对应一系列commit, 是很多点连成的一根线
* tag 对应某次commit, 是一个点
* tag主要用于发布版本的管理. 一个版本发布后可以打上 v.1.0.1 的标签

1. `git tag`
   * `git tag [-a | -s | -u <key-id>] [-f] [-m <msg> | -F <file>] <tagname> [<head>]` : 创建 tag
     * `-f` 强制创建, 会覆盖已存在的 tag
     * `-m <msg> | -F <file>` : 给 tag 添加描述信息
   * `git tag -l` : 列出 tag 信息
   * `git tag -d <tagname>` : 删除对应 tag

2. `git push origin <tagName>`
   * 因为 tag 是基于本地分支的 commit, 与分支的推送是默认分开的
   * 如果需要执行 tag 推送, 需要单独的命令
   * `git push origin --tags` 推送本地所有标签
   * `git push origin :refs/tags/<tagName>` 推送一个 tag 的删除操作

3. `git branch <branchName> <tagName>`: 依照 tag 对应的 commit 来建立分支