# Python 中虚拟环境配置 

## 1. 安装 

使用pip3安装  
`pip3 install virtualenv`  

## 2. 创建 进入 退出  

```shell
# 在当前文件夹创建独立运行环境-命名
# 得到独立第三方包的环境，并且指定解释器是python3
virtualenv <环境名>  
virtualenv --no-site-packages --python=python3  venv

# 进入虚拟环境  
source venv/bin/activate  

#接下来就可以在该虚拟环境下pip安装包或者做各种事了，比如要安装django2.0版本就可以：
pip install django==2.0

# 退出venv环境
deactivate

```
## 3. 其他的管理工具 virtualenvwrapper

安装  
`pip install virtualenvwrapper`  

创建虚拟环境  
该工具是统一在当前用户的envs文件夹下创建，并且会自动进入到该虚拟环境下  
`mkvirtualenv 环境名`  

开启 退出 删除 列出全部  
```
workon 环境名
deactivate
rmvirtualenv 环境名
lsvirtualenv
```

进入虚拟环境目录  
`cdvirtualenv 环境名`
