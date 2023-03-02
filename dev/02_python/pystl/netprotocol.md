- [1. Internet Protocols and Support](#1-internet-protocols-and-support)
  - [1.1.](#11)
- [2. urllib](#2-urllib)
  - [2.1. urllib.request - Extensible library for opening URLs](#21-urllibrequest---extensible-library-for-opening-urls)
    - [2.1.1. 函数](#211-函数)
    - [2.1.2. class urllib.request.Request](#212-class-urllibrequestrequest)
    - [2.1.3. 其他类](#213-其他类)
  - [2.2. urllib.response — Response classes used by urllib](#22-urllibresponse--response-classes-used-by-urllib)
  - [2.3. urllib.parse — Parse URLs into components](#23-urllibparse--parse-urls-into-components)
  - [2.4. urllib.error — Exception classes raised by urllib.request](#24-urlliberror--exception-classes-raised-by-urllibrequest)
  - [2.5. urllib.robotparser — Parser for robots.txt](#25-urllibrobotparser--parser-for-robotstxt)
- [3. http](#3-http)


# 1. Internet Protocols and Support

* 网络高级API, 提供了对多种协议的支持和处理
* The modules described in this chapter implement internet protocols and support for related technology.


## 1.1. 

```py
def http_post(url,data_json):
    jdata = json.dumps(data_json)
    req = requests.post(url, jdata)

    # convert response to dict
    re= json.loads(req.text)
    return re['remaining']
```


# 2. urllib

urllib is a package that collects several modules for working with URLs 

* urllib 是 python 写网络爬虫的核心库
* 在python2各种教程中的 `urllib, urllib2, urlparse` 被重新归一到 urllib 中

* urllib.request        for opening and reading URLs
* urllib.error          containing the exceptions raised by urllib.request
* urllib.parse          for parsing URLs
* urllib.robotparser    for parsing robots.txt files


使用流程:
1. urllib.request 实现核心功能
2. urllib.error 用于 try except 的异常捕捉

## 2.1. urllib.request - Extensible library for opening URLs

该库定义了打开 URLs 的相关操作, 算是 python STL 的核心 url 库
* Use `HTTP/1.1` and includes `Connection:close` header in its HTTP requests


### 2.1.1. 函数

函数总结:
* 核心函数 urlopen
  * 可以直接传入 url, 但是使用 Request 对象可以进行更细致的配置
  * 返回一个没有具体文档定义的对象, 但是可以作为 context manager
  * 返回的对象有 properties: url, headers, and status.
  * 返回的对象是 bytes object, 因为不能轻易判断 HTTP server 的编码类型
* 两个小的类型转换函数 pathname2url  url2pathname

```py
# urlopen  打开一个 url, 根据 url 的内容/协议类型, 返回的对象也不同
urllib.request.urlopen(
    url,                # 传入的url, 可以是 str 或者 Request 对象
    data=None,          # an object specifying additional data to be sent to the server
    [timeout, ]*,       # specifies a timeout in seconds for blocking operations
                        # like the connection attempt, only works for HTTP, HTTPS and FTP connections.
                        # if not specified, the global default timeout setting will be used
    cafile=None,        # Deprecated since version 3.6
    capath=None,        # Deprecated since version 3.6
    cadefault=False,    # Deprecated since version 3.6
    context=None        # a ssl.SSLContext instance describing the various SSL options
)
# 返回:  an object which can work as a context manager and has the properties url, headers, and status.

import urllib.request
with urllib.request.urlopen('http://www.python.org/') as f:
    print(f.read(300))


urllib.request.pathname2url(path)
    # Convert the pathname path from the local syntax for a path to the form used in the path component of a URL. This does not produce a complete URL. The return value will already be quoted using the quote() function.

urllib.request.url2pathname(path)

    # Convert the path component path from a percent-encoded URL to the local syntax for a path. This does not accept a complete URL. This function uses unquote() to decode path.

```

### 2.1.2. class urllib.request.Request 

该子包的同名类, 可以作为参数传入 urlopen 函数

```py
class urllib.request.Request(
    url,                    # 传入 str 的有效 url
    data=None,              # additional data to be sent to the server
                            # 支持 bytes, file-like objects, and iterables of bytes-like objects
    headers={},             # dictionary, 
    origin_req_host=None, 
    unverifiable=False, 
    method=None
)

```

* 类方法  `Request.*`:
  * add_header(key, val)    : 相当于给对象的 headers 字典添加一条新的键值对


### 2.1.3. 其他类



## 2.2. urllib.response — Response classes used by urllib

## 2.3. urllib.parse — Parse URLs into components

## 2.4. urllib.error — Exception classes raised by urllib.request

## 2.5. urllib.robotparser — Parser for robots.txt


# 3. http

和 HTTP 协议相关的低级库, 算得上 urllib 库的基础  

http is a package that collects several modules for working with the HyperText Transfer Protocol:
* http.client       is a low-level HTTP protocol client; for high-level URL opening use urllib.request
* http.server       contains basic HTTP server classes based on socketserver
* http.cookies      has utilities for implementing state management with cookies
* http.cookiejar    provides persistence of cookies
