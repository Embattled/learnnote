# 1. pip    package installer for Python
 
用于从 PyPI ( Python Package Index ) 以及其他 package indexes 下载并管理包

check out the following resources:
* [Getting Started](https://pip.pypa.io/en/latest/getting-started/)
* [Python Packaging User Guide](https://packaging.python.org/)


pip是目前最流行的Python包管理工具
* 是 easy_install的替代品
* 有大量的功能建立在setuptools之上。  

- Python 2.7.9及后续版本: 默认安装, 命令为pip
- Python 3.4及后续版本: 默认安装, 命令为pip3

pip的使用非常简单, 并支持从任意能够通过 VCS 或浏览器访问到的地址安装 Python 包   

和 Python 本体 保持独立状态  

## 1.1. Requirements File

Requirements files serve as a list of items to be installed by pip.

一般的包需求文件都回被命名为 `requirements.txt` , 这只是个习惯, 不是规定  

requirements.txt 的基础语法轻量的, 被其他包管理器所兼容, 但是也有更加全面且完整的语法, 涉及到 pip 的运行形式, 这部分语法是不兼容的



# 2. pip Commands



Environment Management and Introspection: 
* pip install
* pip uninstall
* pip inspect
* pip list
* pip show
* pip freeze
* pip check

Handling Distribution Files:
* pip download
* pip wheel
* pip hash

Package Index information:
* pip search

Managing pip itself:
* pip cache
* pip config
* pip debug



## 2.1. General Options - 通用选项

可以适用于其他所有选项的通用命令  
https://pip.pypa.io/en/latest/cli/pip/#general-options




## 2.2. pip install - Installing Packages

```sh
python -m pip install [options] <requirement specifier> [package-index-options] ...
python -m pip install [options] -r <requirements file> [package-index-options] ...
python -m pip install [options] [-e] <vcs project url> ...
python -m pip install [options] [-e] <local project path> ...
python -m pip install [options] <archive url/path> ...
```

Install packages from: 安装的包的来源
* PyPI (and other indexes) using reuirement specifiers.
* VCS project urls.
* Local project directories.
* Local or remote source archives.
* 以及便捷的 requirements files  


- 安装:  pip install SomePackage
  - 加上 `==` 来指定安装版本 `pip install scipy==0.15.1`

### 2.2.1. install options


安装选项
* `-U, --upgrade`                   : 即使包满足了条件, 也升级到最新的版本
* `--force-reinstall`               : 即使包的版本满足了条件且是最新, 也重新安装
* `--pre`                           : 在查找包索引的時候索引开发版本, 否则只会索引 stable versions
* `--user`                          : 安装到平台上的用户目录, 即 `~/.local/` 等等 (在 conda 下似乎没必要)

便捷
* `-r, --requirement <file>`        : 指定參考文件安裝, 可以多次使用

开发
* `-e, --editable <path/url>`       : 以 editable 模式從本地或者 `VCS url` 安裝一個包




## 2.3. pip uninstall

## 2.4. pip list



## 2.5. pip freeze - Output pkgs list


- pip freeze 另一种查看方法
  - `pip freeze > requirements.txt` 将输出存入文件 可以使别人安装相同版本的相同包变得容易
  - `pip install -r requirements.txt`
  - `pip install -u ModuleName` 下载一个已经安装包的更新版本  

