# libc 

libc 是 Linux 下的 ANSI C 函数库, 包括了 C 语言最基本的库函数 15个头文件    

libc 实际上是一个泛指。凡是符合实现了 C 标准规定的内容，都是一种 libc  
* 微软也有自己的 libc 实现，叫 msvcr
* glibc 是 GNU 组织对 libc 的一种实现。它是 unix/linux 的根基之一
* 嵌入式行业里还常用 uClibc ，是一个迷你版的 libc   

# glibc

glibc 是 Linux 下的 GUN C 函数库。 

1. GNU C 函数库是一种类似于第三方插件的东西。
2. 由于 Linux 是用 C 语言写的，所以 Linux 的一些操作是用 C 语言实现的，因此，GUN 组织开发了一个 C 语言的库以便让我们更好的利用 C 语言开发基于 Linux 操作系统的程序。
3. 不过现在的不同的 Linux 的发行版本对这两个函数库有不同的处理方法，有的可能已经集成在同一个库里了
4. glibc是linux系统中最底层的api 几乎其它任何运行库都会依赖于glibc


查看系统 glibc 版本:
```sh

/lib/libc.so.6

ldd --version

```


# glib

错误观点：glib 前面有个 "g" ，所以认为 glib 是 GNU 的产物；同时认为 glibc 是 glib 的一个子集  

其实，glib 和 glibc 基本上没有太大联系，可能唯一的共同点就是，其都是 C 编程需要调用的库而已  
glib是GTK+的基础库, 是一个综合用途的实用的轻量级的C程序库

