# 1. Git 的配置 Setup and Config

**一些基础的Linux命令**
* `mkdir learngit` 创建一个文件夹  
* `cd learngit` 进入文件夹  
* `pwd`  输出当前所在目录 
* `ls -ah` ls用来输出当前文件夹的所有文件及文件夹  -ah用来输出包括被隐藏的全部


## 1.1. git信息命令



## 1.2. git-config 命令

属于 git 的一个命令行配置命令, 通过CLI的方式直接对 git 配置文件进行一些修改, 语法特别多, 直接对配置文件通过编辑器修改也是可行的  

git配置文件有多个位置:
* 系统层级的配置文件,对应 `git config --system`, `$(prefix)/etc/gitconfig ` `/etc/gitconfig` 
* 用户全局配置文件, 对应 `git config --global` 
  * `$XDG_CONFIG_HOME/git/config ` 环境变量 `XDG_CONFIG_HOME` 如果没有被定义或者为空, 则会使用默认值 `$HOME/.config/`, 即   `$HOME/.config/git/config`
  * `~/.gitconfig`
  * 用户的配置文件是有两个路径的, 即以上两个都会被载入, 但前者优先级更高
* 项目配置 `$GIT_DIR/config `,  对应 `git config --local`, 默认 `$GIF_DIR` 的值是 `.git/`, 即配置文件为 `.git/config`
* 与 local 类似, 但是在工作树功能启动的时候会不同  `git config --worktree`
  * This is optional and is only searched when `extensions.worktreeConfig` is present in `$GIT_DIR/config`.


修改项的值以及在配置文件中的格式, 例如 `user.name` 的修改, 在配置文件中具体的格式为, 要注意直接修改时候的格式
```.gitconfig
[user]
      name = name
```


非修改性命令:
* 打印所有修改后的值 `-l --list`


### 1.2.1. core 核心功能

#### 1.2.1.1. 更换默认编辑器

`git config --global core.editor` <编辑器, 例如 vim>

#### 1.2.1.2. 定义Git全局的 .gitignore 文件

创建针对所有库的全局的 .gitignore 文件, 可以放在任意位置。然后在使用以下命令配置Git:  
`git config --global core.excludesfile ~/.gitignore`

### 1.2.2. 自报家门

Git是分布式版本控制系统, 所以, 每个机器都必须自报家门：你的名字和Email地址  
安装程序完成后需要在命令行输入
```sh
git config --global user.name "Your Name"
git config --global user.email "email.example.com"

git config -l  # 测试编辑好的机器信息
```

注意`git config`命令的`--global` 参数, 用了这个参数, 表示该用户在这台机器上所有的Git仓库都会使用这个配置, 当然也可以对某个仓库指定不同的用户名和Email地址



### 1.2.3. pull


# 2. Getting and Creating Projects

开始一个项目的操作

## 2.1. init 初始化

初始化一个空的仓库, 或者重新初始化一个已经存在的仓库
* 在既存的仓库里调用 `git init` (重新初始化) 是安全的, 不会覆盖任何已经存在的内容
* The primary reason for rerunning git init is to pick up newly added templates (or to move the repository to another place if --separate-git-dir is given).

```sh
git init [-q | --quiet] [--bare] [--template=<template-directory>]
	  [--separate-git-dir <git-dir>] [--object-format=<format>]
	  [-b <branch-name> | --initial-branch=<branch-name>]
	  [--shared[=<permissions>]] [<directory>]
```
所谓的初始化, 实际上执行的操作是:
* 在目标目录下, 创建一个 `.git` 的文件夹, 里面存放了 git 正常追踪运行所需要的所有文件
* 初始化的 git 项目会有一个初始的 branch
* 这个目标目录实际上使用的是 `$GIT_DIR`, 而如果该环境变量未定义, 则使用 `./.git`, 意为在当前 pwd 下创建 .git 文件夹, 当前 pwd 是仓库根目录
* `$GID_DIR` 下面存放git 正常追踪运行所需要的所有文件 
  * objects : 用途 TODO, 似乎是各个追踪的哈希文件, 如果通过 `$GIT_OBJECT_DIRECTORY` 环境变量将 objects 文件夹指定到别的地方, 则会在对应的文件夹下创建 sha1 文件夹

基本参数
* `-q | --quiet` : 安静执行
* `-b <branch-name> | --initial-branch=<branch-name>` : 初始的分支名称, 默认是 `master`, 可以通过 config 里的 `init.defaultBranch` 来修改
* `<directory>` : 在最后提供一个目录, 会将仓库创建在对应的文件夹里, 如果文件夹不存在也会创建新文件夹




## 2.2. clone 克隆

从远端克隆一个仓库到一个新的文件夹, creates remote-tracking branches for each branch in the cloned repository (visible using `git branch --remotes`), and creates and checks out an initial branch that is forked from the cloned repository’s currently active branch.

完整命令
```sh
git clone [--template=<template-directory>]
	  [-l] [-s] [--no-hardlinks] [-q] [-n] [--bare] [--mirror]
	  [-o <name>] [-b <name>] [-u <upload-pack>] [--reference <repository>]
	  [--dissociate] [--separate-git-dir <git-dir>]
	  [--depth <depth>] [--[no-]single-branch] [--no-tags]
	  [--recurse-submodules[=<pathspec>]] [--[no-]shallow-submodules]
	  [--[no-]remote-submodules] [--jobs <n>] [--sparse] [--[no-]reject-shallow]
	  [--filter=<filter> [--also-filter-submodules]] [--] <repository>
	  [<directory>]
```

命令相关:
* clone 以后
  * 执行 `git fetch` 会更新所有的 remote-tracking branches
  * `git pull`  会更新所有的 remote-tracking branches 并将更新的内容合并到本地

常用选项: 
* `-b <name>, --branch <name>` 指定要克隆的分支


子模组:
* `--recurse-submodules`  递归的克隆全部的 submodule, initialize and clone submodules .
  * `--recurse-submodules[=<pathspec>]` : 指定具体的 submodules, 如果不指定的话会克隆所有的 submodules
  * This option can be given multiple times for pathspecs consisting of multiple entries.
  * 执行克隆的时候会该设置 repo 的 git 的配置  `submodule.active`
    * set to the provided pathspec
    * or `.` (meaning all submodules) if no pathspec is provided.
  * 整个命令相当于一个 clone 后接了一个 `git submodule update --init --recursive <pathspec>`
  * 如果克隆过程不产生 worktree 的话, 则该命令会被忽视, 即 (if any of `--no-checkout/-n`, `--bare`, or `--mirror` is given, 详情查看各自的说明)


# 3. 离线操作

* `git init` 在当前目录创建 git 内部文件, 使当前路径称为一个 git 项目

* 工作区    : 就是电脑里能看到的目录  
* 暂存区    : 在使用`git add`的时候,就是把文件修改添加到暂存区  
* 隐藏目录  : `.git`不算工作区,而是版本库  

而`git commit`就是把暂存区的内容正式提交到分支上



## 3.1. gitignore 过滤文件
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


## 3.2. 身兼数职的 checkout

0. 在 git 2.23 版本以前, 对于丢弃工作区的修改
   * git 会提示使用 `git checkout -- file`
   * 事实上 `checkout` 还兼任了切换分支的功能, 这容易导致操作混淆
1. 在 Git 2.23 版本开始引入了两个新的命令 `switch` 和 `restore`
   * 通过直接访问 `man git` 可以查看到最新版本对这 3 个命令的定义
   * `git-checkout(1)   Switch branches or restore working tree files.`
   * `git-switch(1)     Switch branches.`
   * `git-restore(1)    Restore working tree files.`
2. 可以看到两个新的命令完美替换掉了 checkout, 因此应该尽量避免使用 checkout

## 3.3. 文件更改 未commit下的


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

## 3.4. 版本提交与回退

1. `commit` 将暂存区的内容提交到版本库, 每一次commit都是一个保存点,可以从这里还原版本库  
   * `git commit -c <commit>` 懒人代码, 直接复制参数 commit 的 log msg
   * `git commit -C <commit>` 复制参数 commit 的 log msg 并进入编辑界面
   * ` git commit -m "wrote a readme file"`  
   * 如果commit注释写错了, 只是想改一下注释, 只需要：
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

## 3.5. commit message

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

# 4. Basic Snapshotting - 基础的版本记录

在不存在分支的情况下用到的各种版本控制命令

## 4.1. add - Add file contents to the index

```sh
git add [--verbose | -v] [--dry-run | -n] [--force | -f] [--interactive | -i] [--patch | -p]
	  [--edit | -e] [--[no-]all | --[no-]ignore-removal | [--update | -u]] [--sparse]
	  [--intent-to-add | -N] [--refresh] [--ignore-errors] [--ignore-missing] [--renormalize]
	  [--chmod=(+|-)x] [--pathspec-from-file=<file> [--pathspec-file-nul]]
	  [--] [<pathspec>…​]
```

`add` 命令用来添加新文件或者对文件的新修改到暂存区


   * `git add readme.txt`  单独提交一个文件  
   * `git add .` 会监控工作区的状态树, 提交文件修改(modified), 以及新文件(new), 但不包括被删除的文件
   * `git add -u[pdate] ` 仅监控已经被add的文件(tracked file) 他会将被修改的文件提交到暂存区, 不会提交新文件 (untracked file), 提交被修改(modified)和被删除(deleted)文件, 不包括新文件(new)
   * `add .` 和 `add -u` 有着微妙的区别, 在于对新文件和删除文件的态度上
   * `git add -A[ll] ` 是上面两个功能的合集 (提交所有变化)



# 5. Branching and Merging 分支管理

`man git` 的 分支相关的命令说明
 * switch              Switch branches.
 * branch              List, create, or delete branches.
 * merge               Join two or more development histories together.
 * (deprecate)checkout Switch branches or restore working tree files.

    branch
    checkout
    switch
    merge
    mergetool
    log
    stash
    tag
    worktree


- 通常`master`分支是用来发布稳定版本的
- `dev`分支是用来保存不稳定版本的  
- 在此之下才是各个团队工作人员自己的分支
- 开发一个新的feature,最好新建一个分支  

## 5.1. branch - List, create, or delete branches

```sh
git branch [--color[=<when>] | --no-color] [--show-current]
	[-v [--abbrev=<n> | --no-abbrev]]
	[--column[=<options>] | --no-column] [--sort=<key>]
	[--merged [<commit>]] [--no-merged [<commit>]]
	[--contains [<commit>]] [--no-contains [<commit>]]
	[--points-at <object>] [--format=<format>]
	[(-r | --remotes) | (-a | --all)]
	[--list] [<pattern>…​]
git branch [--track[=(direct|inherit)] | --no-track] [-f]
	[--recurse-submodules] <branchname> [<start-point>]

# 链接远程默认分支, 简化之后的 pull 命令
git branch (--set-upstream-to=<upstream> | -u <upstream>) [<branchname>]
git branch --unset-upstream [<branchname>]

# 重命名
git branch (-m | -M) [<oldbranch>] <newbranch>

# 复制
git branch (-c | -C) [<oldbranch>] <newbranch>

# 删除
git branch (-d | -D) [-r] <branchname>…​
git branch --edit-description [<branchname>]
```


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
 * 因为创建、合并和删除分支非常快, 所以Git鼓励你使用分支完成某个任务, 合并后再删掉分支, 这和直接在master分支上工作效果是一样的, 但过程更安全。
 * 使用 `git branch -d <分支名称>`可以删除一个分支
 * `-D` 命令用于强制删除一个还没有被 merge 的分支
* 链接远程分支 `--set-upstream-to`
 * 就算是从远端克隆的repo, 也不会自动关联到远程对应的分支, 需要手动指定本地与远端的链接
 * 链接的唯一好处就是简化之后的 pull 命令
 * `git branch --set-upstream-to dev origin/dev`  

## 5.2. checkout - Switch branches or restore working tree files

```sh
git checkout [-q] [-f] [-m] [<branch>]
git checkout [-q] [-f] [-m] --detach [<branch>]
git checkout [-q] [-f] [-m] [--detach] <commit>

# 从 start-points 来复制的创建一个新的 branch, 相当于  git branch <branch> [<start-point>]
git checkout [-q] [-f] [-m] [[-b|-B|--orphan] <new-branch>] [<start-point>]

git checkout [-f|--ours|--theirs|-m|--conflict=<style>] [<tree-ish>] [--] <pathspec>…​
git checkout [-f|--ours|--theirs|-m|--conflict=<style>] [<tree-ish>] --pathspec-from-file=<file> [--pathspec-file-nul]
git checkout (-p|--patch) [<tree-ish>] [--] [<pathspec>…​]
```

## 5.3. switch - Switch branches

`switch` 切换分支  不建议使用 `checkout` 命令
   * `git switch [<options>] [<branch>]`
   * `git switch -c dev` 创建并切换, 省去 `branch` 命令, 同理 `-C` 也是覆盖的创建



## 5.4. merge - Join two or more development histories together

```sh
git merge [-n] [--stat] [--no-commit] [--squash] [--[no-]edit]
	[--no-verify] [-s <strategy>] [-X <strategy-option>] [-S[<keyid>]]
	[--[no-]allow-unrelated-histories]
	[--[no-]rerere-autoupdate] [-m <msg>] [-F <file>]
	[--into-name <branch>] [<commit>…​]
git merge (--continue | --abort | --quit)
```

`merge` 合并一个分支
   * `git merge [<options>] [<commit>...]`
   * 出现冲突时:
     * `git merge --abort` 用于放弃本次合并
     * `git merge --continue` 用于手动处理好冲突后再次合并
   * 使用 `git merge <分支名称>` 来将分支的工作成果合并到`master` 分支上
   * 分支本身也会作为一个 commit 被记录

当Git无法执行`快速合并`, 只能试图把各自的修改合并起来, 但这种合并就可能会有冲突  
必须手动解决冲突后再提交

使用`git status`可以告诉我们冲突的文件
Git用<kbd><<<<<<<</kbd>, <kbd>=======</kbd>, <kbd>>>>>>>></kbd>标记出不同分支的内容



### 5.4.1. merge options

提交
* `--commit`      : merge 然后提交结果, 这算是默认的行动, 也可以在默认行动被更改为 `--no-commit` 的时候用于覆盖默认行动
* `--no-commit `  : merge 然后不提交 merge commit, 让用户有机会进一步检查合并的结果
  * Note : fast-forward updates do not create a merge commit and therefore there is no way to stop those merges with --no-commit. If you want to ensure your branch is not changed or updated by the merge command, use --no-ff with --no-commit.

分支属性
* `--allow-unrelated-histories` : 默认下, merge 会拒绝合并不存在共同祖先的两个分支, 如果要合并两个独立的 branch 项目, 可以用该命令.
  * 因为该操作非常罕见, 因此不存在于配置文件中, 即无法设置该行动为默认行动


## 5.5. stash - Stash the changes in a dirty working directory away

`stash` 现场保存, 用于非 commit 下的工作进度保存, 拥有一整套子命令组

```sh
git stash list [<log-options>]
git stash show [-u | --include-untracked | --only-untracked] [<diff-options>] [<stash>]
git stash drop [-q | --quiet] [<stash>]
git stash pop [--index] [-q | --quiet] [<stash>]
git stash apply [--index] [-q | --quiet] [<stash>]
git stash branch <branchname> [<stash>]
git stash [push [-p | --patch] [-S | --staged] [-k | --[no-]keep-index] [-q | --quiet]
	     [-u | --include-untracked] [-a | --all] [(-m | --message) <message>]
	     [--pathspec-from-file=<file> [--pathspec-file-nul]]
	     [--] [<pathspec>…​]]
git stash save [-p | --patch] [-S | --staged] [-k | --[no-]keep-index] [-q | --quiet]
	     [-u | --include-untracked] [-a | --all] [<message>]
git stash clear
git stash create [<message>]
git stash store [(-m | --message) <message>] [-q | --quiet] <commit>

```



* `git stash [push]` : 立刻将工作区的改动存于后台, 并将工作区的文件还原到 HEAD 状态, 此时可以进行切换分支等操作
  * `-m <message>` : 给 stash 添加描述
* `git stash list` : 列出所有 stash 
* `git stash show` : 打印 stash 与 commit back 的 diff
* `git stash (pop | apply) ` :
  * apply 恢复,但是stash的内容不删除
  * pop   恢复的同时也将stash的内容删除了  
* `git stash drop` : 删除 stash 


## 5.6. 分支的团队命名

同 commit message 一样, 合理的命令可以增加团队配合  
`type/module/snake_case_short_summary`  

* type : 分支的任务类型, 参考[commit message里的type说明](#25-commit-message)
* module : 要修改的文件名, 或者广泛一点的模组名
* snake_case_short_summary: 
  * 和具体的实现内容有关
  * 和最终的 PR 题目一一对应
  * 用下划线连接多个小写单词
  * 如果对应的是一个 issue, 则直接写成 issue 编号也行


## 5.7. 分支管理策略

通常合并分支时,git会在可能的时候使用`Fast forward`模式,这种模式的缺点就是删除分支后就会丢失分支的信息

使用  
`git merge --no-ff` 来强制禁用 `fast forward`模式,这种模式就会在`merge`的时候自动生成新的`commit`,因此还应加上`-m`来描述这个提交  
`git merge --no-ff -m "merge with no-ff" dev`


### 5.7.1. Bug处理

对于一个已经存在于所有分支的bug,git提供了一个`cherry-pick`,能够将一个特定的分支复制到当前分支  
`git switch dev`
`git cherry-pick <ID>`  
此时git会自动给dev分支进行一次提交,该commit会获得一个不同的ID, 使用该命令可以省去重复修复bug的过程



# 6. Sharing and Updating Projects

## 6.1. 远程版本库

1. `remote` 命令用于管理和远程相关的事物
   * 使用`git remote`查看远程库的名称  
   * 使用`git remote -v`可以显示包括url的详细信息  
   * 使用`git remote add <命名远程库> <url>`  添加一个远程库
     * `git remote add origin  https://github.com/embattled/learnnote.git`  
     * origin是默认的远程库叫法,也可以自定义

2. `push` 命令用于推送所有 commit 到远程版本库
   * `git push <远程库名称> <分支名称>`
   * `-f  --force` 参数用于强制推送, 可以用于撤销已经 push 的 commit
   * `-u --set-upstream`   set upstream for git pull/status, 该命令在 `branch` 中也可以设置 
     * 一般用于执行第一次推送
     * `git push -u origin master` 命令来第一次推送  
     * `-u` 在将本地分支推送到远程的基础上,还将本地和远程的该分支关联了起来,在以后的推送或者拉去时可以简化命令
     * 在这之后的推送使用  `git push origin master`即可
     * 该配置在 branch 上有必要

3. `clone <url>` 用于克隆版本库
   * `-b, --branch <branch>` 指定要克隆的分支
   * `--recurse-submodules`  递归的克隆全部的 submodule, initialize and clone submodules within based on the provided pathspec.

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


# 7. Git 的标签管理 tag

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

# 8. Sharing and Updating Projects

共享项目以及协同作业
    fetch
    pull
    push
    remote
    submodule
    subtree (貌似非官方)

## 8.1. fetch 

git-fetch - Download objects and refs from another repository  

```sh
git fetch [<options>] [<repository> [<refspec>…​]]
git fetch [<options>] <group>
git fetch --multiple [<options>] [(<repository> | <group>)…​]

# 从当前 repo 的所有 remote 中进行 fetch
git fetch --all [<options>]
```



## 8.2. submodule 子模块


主要用于在项目庞大的时候进行模块文件抽离, 抽离出来的文件可以单独成为一个 git repo

开发流程:
* 不管主项目中的是否有更新, 只要子项目中修改过代码并提交过, 主项目就需要再提交一次
* 1. 进入到子项目中对本次更新的代码进行一次提交操作, 并推送到远程
* 2. 退到主项目中, 对主项目进行一次提交并推送

```sh
git submodule [--quiet] [--cached]
git submodule [--quiet] add [<options>] [--] <repository> [<path>]
git submodule [--quiet] status [--cached] [--recursive] [--] [<path>…​]
git submodule [--quiet] init [--] [<path>…​]
git submodule [--quiet] deinit [-f|--force] (--all|[--] <path>…​)
git submodule [--quiet] update [<options>] [--] [<path>…​]
git submodule [--quiet] set-branch [<options>] [--] <path>
git submodule [--quiet] set-url [--] <path> <newurl>
git submodule [--quiet] summary [<options>] [--] [<path>…​]
git submodule [--quiet] foreach [--recursive] <command>
git submodule [--quiet] sync [--recursive] [--] [<path>…​]
git submodule [--quiet] absorbgitdirs [--] [<path>…​]

# 通用参数
-q, --quiet  : 安静执行, Only print error messages.
```

* 配置 submodule 后:
  * 会在项目根目录下新建可见 `.gitmodules` 文件用于管理子仓库
  * 同时在不可见的 git 内部配置文件 `.git/config` 中添加了响应的配置
* `(空)`  : 打印当前仓库的 submodule 信息

相关概念
* `.gitmodules` `$GIT_DIR/config` 文件用于描述 子模块
* `git <command> --recurse-submodules`  在进行仓库操作的时候, 可以递归操作所有子模块来简化操作

* 子父仓库的代码没有双向关联
  * 不能在 父版本库中 修改子版本库的代码, 即子库的代码是单向传递给父库的 (某些时候可能是需要该特性的)
  * 子仓库的代码的修改不会自动反映到父仓库里, 需要手动同步




### 8.2.1. add 添加子模块

`add [-b <branch>] [-f|--force] [--name <name>] [--reference <repository>] [--depth <depth>] [--] <repository> [<path>]`

将给定的 repo 作为子模块添加到当前 repo


`git submodule add git@地址 my_sub_module` 将地址对应的远程仓库作为子库, 保存到当前版本库的对应目录下, 通过 git status 查看父库的状态有:  
* 对应的 创建/修改 `.gitmodules` 文件
* 新的子库的文件夹  

查看子库的信息 `.gitmodules` 则有  
```.gitmodules
[submodule "子库名称"]
    path = 父库目录下子库的相对路径
    url  = 子库的远程url
```


### 8.2.2. 


  * `init` : Initialize the submodules recorded in the index, 将对应的 submodule URL 写入本地 .git/config



submodule: 子模块 Mounting one repository inside another
* 将一个版本库作为子库引入到另一个版本库中
  * 子库 submodule 有自己的历史
  * 嵌入子库的库成为 superproject
* 如果要克隆完整仓库, 
  * 直接默认的 clone 父仓库的时候, 所有子仓库都只有一个空文件夹  
  * 需要额外的步骤对子库进行 `init` 和 `update`
  * `git submodule init`
  * `git submodule update --recursive`
* 通过 `.gitmodules` 来进行文件记录和 submodule 的版本信息
* 如果要删除一个 submodule, 步骤比较繁琐, 没有专用的命令
  * 需要删除对应子模块的文件夹
  * 从 `.gitmodules` 中删除对应的项
  * 提交以上两个改动





### 8.2.3. status 查看子库的信息

`status [--cached] [--recursive] [--] [<path>…​] `  

打印所有子库的名字以及版本信息  .

* `status`  : 打印 submodule 的版本信息
  * 对于主仓库来说, git submodule 是以 分支:版本 为基础管理的, 即 子仓库的内容会维持同一个分支的同一个版本不变
  * 若需要更新子仓库的内容, 需要进入子仓库进行 `git switch`, 这个switch也需要在主仓库里进行一次 commit 
  * Git认为移动submodule的指针和其他变化一样: 如果我们保存这个改动, 就必须提交到仓库里

### 8.2.4. init 初始化子库

`init [--] [<path>…​] `  

将所有子库的目录初始化为 git 版本控制目录, 该命令也不会正式下载子库的代码, 具体的下载在 `update` 命令中, 该步骤可以通过 `update --init` 直接省略掉

### 8.2.5. update 子库更新核心命令

```sh
update [--init] [--remote] [-N|--no-fetch] [--[no-]recommend-shallow]
      [-f|--force] [--checkout|--rebase|--merge] [--reference <repository>]
      [--depth <depth>] [--recursive] 
      [--jobs <n>] [--[no-]single-branch] [--filter <filter spec>] 
      [--] [<path>…​] 
```

更新所有登录的子模块, 使得满足 superproject 的期望, 其中包括
* 克隆缺失的子模块
* fetching missing commits in submodules
* 更新子模块的工作树 working tree

该命令的无参默认动作可以通过配置文件中的 `submodule.<name>.update` 来指定, 否则执行真正软件层的默认操作, 即 `checkout`. 


参数说明
* `--checkout|--rebase|--merge]` 执行具体的怎么样的 update
  * checkout : git 的默认操作
    * commit recorded in the superproject will be checked out in the submodule `on a detached HEAD`.
    * 将子模块记录中的 commit checkout
    * If --force is specified, 即使子模块中的 git commit log 与 superfroject 中的记录匹配, 也会重新执行 checkout
  * rebase  : 子模块的 current branch, 会 rebase 到 superproject 记录的 commit
  * merge   : 到 superproject 记录的 commit 会 merge 到子模块当前的 current branch 中



* `[--init]` : 自动初始化子模块


### 8.2.6. foreach 遍历所有子库

## 8.3. subtree 

从1.5.2版本开始, 官方新增Git Subtree并推荐使用这个功能来替代Git Submodule管理仓库共用(子仓库, 子项目)  
但是官方文档里没有 subtree 的介绍, 不知为何  

git submodule/subtree 允许其他的仓库指定以一个commit嵌入仓库的子目录
* git subtree 作为 git submodule 的替代命令, 拥有更加强大的功能且据说被官方推荐
* 但是在 git 官方文档里查不到任何 git subtree 的说明
* google 上也没有 git subtree 的官方说明网站, 只有仓库里的[说明文件](https://github.com/git/git/blob/master/contrib/subtree/git-subtree.txt)


![不同点](https://imgconvert.csdnimg.cn/aHR0cDovL2FodW50c3VuLmdpdGVlLmlvL2Jsb2dpbWFnZWJlZC9pbWcvZ2l0L2xlc3NvbjEwLzI5LnBuZw?x-oss-process=image/format,png)
subtree 的不同:
* 不增加 `.gitmodule`, 可以直接一条命令删除子库
* 子项目对其他成员透明, 可以不用知道 subtree 的存在
  * 可以在父库中修改子库的内容, 甚至可以把修改的子库内容推送到远程子库中
* 本质就是把子项目目录作为一个普通的文件目录, 对于父级的主项目来说是完全透明的, 原来是怎么操作现在依旧是那么操作
  * 在远程中 (github)
  * submodules 的子库目录是一个链接, 点进去会跳转到别的仓库页面
  * subtree 的子库就是文件, 没有链接
* 主子仓库的分支同步, 切换主项目分支的时候, 子仓库也会同步切换
* 缺点:
  * 无法直接单独查看子仓库的修改记录, 因为子仓库的修改包含在父仓库的记录中了
  * 子仓库的分支管理较为麻烦
  * 单独的一整套指令, 且复杂度较高, 需要学习

```sh
[verse]
'git subtree' [<options>] -P <prefix> add <local-commit>
'git subtree' [<options>] -P <prefix> add <repository> <remote-ref>
'git subtree' [<options>] -P <prefix> merge <local-commit> [<repository>]
'git subtree' [<options>] -P <prefix> split [<local-commit>]

[verse]
'git subtree' [<options>] -P <prefix> pull <repository> <remote-ref>
'git subtree' [<options>] -P <prefix> push <repository> <refspec>
```

### 8.3.1. add 添加子库

由于 subtree 的工作原理是基于分支的, 因此这里添加子库的方法还需要用到一些基础命令  
1. 单纯添加子库的地址到 remote :    `git remote add subtree-origin git@子库链接` 
   * 该步骤其实是可选的
   * 将链接添加到库的 remote 中会方便接下来的操作, 如果不添加的话则在下面的命令中使用完整链接
2. 将remote地址链接成子库   : `git subtree add -P submodule subtree-origin master --squash`
   * 该命令表示链接的对象是 remote 链接中的 `subtree-origin` 的 master 分支
   * 由于链接以及变成子库的对象是一个具体的分支, 因此可以把一个远程库的不同分支当作不同的子库同时放进一个父项目中
   * `-P <prefix>` 表示具体到本地目录的目录名, 如果不指定应该就是 子库原本的仓库名
   * `--squash` 是一个常用的参数, 表示合并/压缩, 表示会把远程子库上的历史提交合并成一次提交, 再拉去到本地, 这样子库的历史分支就看不到了
     * 如果不加的话, 会拷贝子库的所有历史提交, 造成 git log 污染
     * 对于一个子库的使用, 需要在整个使用中保持 `--squash` 值的相同, 即要么永远不用, 要么永远用
     * `远程子库上的历史提交合并成一次提交` 会导致在之后不加 squash 的拉取中由于找不到公共父节点导致拉取失败

### 8.3.2. pull 拉取更新

### 8.3.3. push 

### 8.3.4. split

subtree 的强大功能, 抽离子库: 开发的过程中发现某些功能可以剥离出来当作公用的子库的时候, 在保留该新子库的所有历史 log 的情况下, 生成一个新的库    
`git subtree split --prefix=<prefix> [OPTIONS] [<commit>]`

假设某个文件夹 module1 可以作为一个新的子库独立出去
1. 剥离该目录 `git subtree split -P module1 -b childb`, -b 表示创建了一个新的分支 childb 用于单独保存该分支的文件夹
2. 在别的地方创建正式的子库目录 `mkdir module1` `git init`
3. 从父库剥离除去的子库文件分支中, 拷贝代码到子库新的独立的位置 `git pull 父库路径 childb(剥离分支名称)`
4. 在新的仓库创建远程关联, 推送等
   * `cd module1`
   * `git remote add origin git@module1`
   * `git push -u origin +master`


# 9. git-lfs Large File Storage (LFS)

replaces large files such as audio samples, videos, datasets, and graphics with `text pointers` inside Git.

配置好的 git-lfs 会截取 push 中的那些被追踪的大文件, 然后用 text pointers 替换他们, 并将原本的大文件发送到 `git lfs 服务器`


基础操作
1. 安装该插件(各用户执行单词) `git lfs install`
2. 追踪指定文件 `git lfs track "*.psd"`
3. 查看当前 track 的目录或类型 `git lfs track`
4. track 后, 需要再通过普通的 git add/ commit 方式进行版本管理
5. 查看被 track 的具体的文件 `git lfs ls-files`