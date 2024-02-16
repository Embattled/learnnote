# 1. Android Development

在一个标准安卓项目中:  
Types of modules include:
* Android app modules
* Library modules
* Google App Engine modules


安卓项目目录结构, 根据 IDE 不同有些微的差异

* `app/`          : 程序文件
  * `build/`      : 编译生成的结果目录, 类似于 bin
  * `libs/`       : 依赖库目录 (jar包)
  * `src/`        : 项目源文件
    * `androidTest/` : Android Test 用例, 对项目进行自动化测试
    * `test/`        : Unit Test 用例, 对项目进行自动化测试
    * `main/`     : 代码
      * `java/`   : java代码目录
      * `jni/`    : 原生代码目录 (C/C++)
      * `res/`    : 其他资源目录, 图像资源 布局资源, 菜单资源 
      * `libs/`   : 如果项目内容是库的话, 编译生成的 .so 放在这里   
    * `build.gradle` : app模块的 gradle 构建脚本

# 2. Android Studio

是 安卓官方的 IDE
* 以 IntelliJ IDEA 为基础构建而成

# 3. jni Java Native Interface 

Java1.1开始, JNI标准成为java平台的一部分 

通过使用 Java本地接口书写程序, 可以确保代码在不同的平台上方便移植
* 允许JAVA程序调用C/C++写的程序
* 书写步骤:
  1. 编写带有native声明的方法的java类
  2. 使用javac命令编译所编写的java类
  3. 然后使用javah(已被舍弃) + java类名生成扩展名为h的头文件
  4. 使用C/C++实现本地方法
  5. 将C/C++编写的文件生成动态连接库

## 3.1. JNI层的 java 源文件

编写带有 `native` 声明的方法的java类
* native 是java啥关键字不记得了 放置

要注意:
* 因为只是声明, 所以IDE报错也没关系

