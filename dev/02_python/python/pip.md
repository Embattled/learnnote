# 1. pip    package installer for Python
 
用于从 PyPI ( Python Package Index ) 以及其他 package indexes 下载并管理包

check out the following resources:
* [Getting Started](https://pip.pypa.io/en/stable/getting-started/)
* [Python Packaging User Guide](https://packaging.python.org/)


pip是目前最流行的Python包管理工具
* 是 easy_install的替代品
* 有大量的功能建立在setuptools之上。  

- Python 2.7.9及后续版本: 默认安装, 命令为pip
- Python 3.4及后续版本: 默认安装, 命令为pip3

pip的使用非常简单, 并支持从任意能够通过 VCS 或浏览器访问到的地址安装 Python 包  

## 1.1. Requirements File

Requirements files serve as a list of items to be installed by pip.

一般的包需求文件都回被命名为 `requirements.txt` , 这只是个习惯, 不是规定  

requirements.txt 的基础语法轻量的, 被其他包管理器所兼容, 但是也有更加全面且完整的语法, 涉及到 pip 的运行形式, 这部分语法是不兼容的





# 2. pip command


## 2.1. pip install - Installing Packages

```sh
python -m pip install [options] <requirement specifier> [package-index-options] ...
python -m pip install [options] -r <requirements file> [package-index-options] ...
python -m pip install [options] [-e] <vcs project url> ...
python -m pip install [options] [-e] <local project path> ...
python -m pip install [options] <archive url/path> ...
```

Install packages from: 安装的包的来源
* PyPI (and other indexes) using requirement specifiers.
* VCS project urls.
* Local project directories.
* Local or remote source archives.
* 以及便捷的 requirements files  




- 安装:  pip install SomePackage
  - 加上 `==` 来指定安装版本 `pip install scipy==0.15.1`
- 卸载:  pip uninstall SomePackage

- pip list 查看已安装包的列表

## pip freeze - Output pkgs list


- pip freeze 另一种查看方法
  - `pip freeze > requirements.txt` 将输出存入文件 可以使别人安装相同版本的相同包变得容易
  - `pip install -r requirements.txt`
  - `pip install -u ModuleName` 下载一个已经安装包的更新版本  

