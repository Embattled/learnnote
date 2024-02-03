- [1. Markdown 语言](#1-markdown-语言)
  - [1.1. CommonMark](#11-commonmark)
  - [1.2. GitHub Flavored Markdown GFM](#12-github-flavored-markdown-gfm)
  - [1.3. Markdown Extra](#13-markdown-extra)
- [2. 语法](#2-语法)
  - [2.1. 显示保留字符](#21-显示保留字符)
  - [2.2. 标题](#22-标题)
  - [2.3. 段落](#23-段落)
    - [2.3.1. 字体](#231-字体)
    - [2.3.2. 分割线](#232-分割线)
    - [2.3.3. 删除线](#233-删除线)
    - [2.3.4. 下划线](#234-下划线)
    - [2.3.5. 脚注](#235-脚注)
  - [2.4. 列表](#24-列表)
  - [2.5. 区块](#25-区块)
  - [2.6. 代码](#26-代码)
  - [2.7. 超链接](#27-超链接)
  - [2.8. 表格](#28-表格)
  - [2.9. 高级技巧](#29-高级技巧)
    - [2.9.1. 支持使用HTML元素](#291-支持使用html元素)
    - [2.9.2. TeX LaTeX 数学公式](#292-tex-latex-数学公式)

# 1. Markdown 语言

`John Gruber` and `Aaron Swartz` created Markdown in 2004
* 在 `John Gruber` 的个人网站 [daringfireball](https://daringfireball.net) 于 2004 年被创建 (ver 1.0.1)
* 目的是  text-to-HTML conversion tool for web writers
* Markdown 在发布的时候
  * 包括了一整套语法
  * 包括一个工具, 由 Perl 语言编写, 用于将纯文本 markdown 文件转化为 HTML
  * 并不是一个标准的规范, 因此官方版本迅速被各种衍生兼容版埋没
  * Gruber 本人也拒绝为 Markdown 定义一套规范, 而是赞成根据各自需求自己定义
  * 原版 Markdown 保留了花括号的定义, 用于各种实现根据需求去设定功能 
* 目前唯一的 Markdown 官方网站仍然是他的个人博客 [Markdown Offical](https://daringfireball.net/projects/markdown/)

和同类型的其他语言相比, Markdown 是 Lightweight Markup Language (LML) 语言中最出名的
* Markdown 的名称由来就是为了与标记语言(Markup)相反, 例如 RTF 或者 HTML


目前 Markdown语言已经被列入 RFC中 
* Internet media type	: `text/markdown` (RFC 7763)
* 其他变种 (RFC 7764)


## 1.1. CommonMark

`Jeff Atwood` and `John MacFarlane` 于 2012 年发起的项目[官网](https://spec.commonmark.org/)
* 该项目旨在: 定义一个 Markdown 兼容的标准, 但是消除原版语法中的所有歧义
* 该项目引起了原作者 `John Gruber` 的强烈反感, 并使得该项目进行了多次重命名 `Standard Markdown` -> `Commom Markdown` -> `CommonMark`

* Internet media type	: `text/markdown; variant=CommonMark`

* 目前各个网站以及项目中的 Markdown 大多都基于 CommonMark
  * Discourse
  * GitHub
  * GitLab
  * Reddit
  * Qt
  * Stack Exchange (Stack Overflow)
  * Swift

## 1.2. GitHub Flavored Markdown GFM

Github 于2009年开始使用了自己的 Markdown 变种, 用于
* support for additional formatting such as tables and nesting block content inside list elements
* as well as GitHub-specific features such as auto-linking references to commits, issues, usernames
* 一直到 2017 年, Github 正式将自己的 Markdown 变种 GFM 的正式规范
  * 相比于 CommonMark 更加严格


## 1.3. Markdown Extra

加入了非常多的功能, 基于 PHP, Python, Ruby 的 Markdown 实现而开发的


# 2. 语法
## 2.1. 显示保留字符
在字符前加反斜杠代表正常显示  
`\* \#`  
可以正常显示星号和井号  
## 2.2. 标题
使用`#`号可以表示1-6级标题，一级对应一个`#`号  
`# 一级标题`  
`## 二级标题`  
## 2.3. 段落
Markdown的段落没有特殊的格式，要换行需要在末尾加上两个空格再回车  
`文字  (跟两个空格再加回车)`  
或者直接两个回车，在代码中空一行代表重新开始一个新段落

### 2.3.1. 字体
`*斜体文本*` *斜体*  
使用一个星号包围表示斜体文本 


`**粗体文本**` **粗体**  
两个星号代表粗体文本

`***粗斜体***` ***粗斜体***  
三个星号代表粗斜体

### 2.3.2. 分割线
可以在一行中用三个以上的星号、减号、底线来建立一个分隔线，行内不能有其他东西。  
也可以在星号或是减号中间插入空格。下面每种写法都可以建立分隔线：  
`***`  
`* * *`  
`___`  
`----------`
显示效果如下

---
在分隔线中的文字
***

### 2.3.3. 删除线
在文字上添加删除横线，使用双波浪线包围  
`~~文字~~`  
~~要删除的文字~~


### 2.3.4. 下划线
下划线需要通过HTML的`<u>`标签实现  
`<u>带下划线的文本</u>`  
<u>带下划线的文本</u>

### 2.3.5. 脚注
对文本提供补充说明，当鼠标移动到文本上时显示额外说明  
鼠标移动到[^这里]  
[^这里]: 脚注就出现了

创建脚注格式类似这样[^RUNOOB]。  
[^RUNOOB]: 菜鸟教程 -- 学的不仅是技术，更是梦想！！！

生成一个脚注1[^footnote].
  [^footnote]: 这里是 **脚注** 的 *内容*.
生成一个脚注2[^foot].  
[^foot]: 这里是**脚注2**的*内容*.

## 2.4. 列表
Markdown支持有序列表和无序列表

**无序列表**可以使用`*`或者`+`或者`-`来表示
可以识别缩进代表子项，或者**2个空格**
+ 第一项
  + 子项
    + 再子项  
- 第二项
  - 二子
    - 二再子
* 第三项
  * 三子
    * 三再子

有序列表使用数字并加上 `.` 号来表示，如
`1. 第一项`
1. 第一项

## 2.5. 区块
在段落开头使用 `>` 符号，再跟一个空格
> 最外层嵌套
>>第二层
>>>第三层

**区块中使用列表**

> 1. 第一项
> 2. 第二项
> > 1. 嵌套第一项
> > + 嵌套无序项

## 2.6. 代码
`printf()` 函数

**代码区块**
```c++
for(i=0;i<=5;i++)
```

## 2.7. 超链接
[链接名称](链接地址)  

`这是一个链接 [菜鸟教程](https://www.runoob.com)`  
这是一个链接 [菜鸟教程](https://www.runoob.com)  

或者  
<链接地址>  
`<https://www.runoob.com>`  
<https://www.runoob.com>

**高级链接** 具有变量性质的链接

`[链接名称][链接变量]`
`[Google][1]`
[Google][1]
[1]: www.google.com/

**图片链接-带有属性的链接**

```
![属性文本](图片地址)

![属性文本](图片地址 "可选标题")
```
+ 开头一个感叹号 !
+ 接着一个方括号，里面放上图片的替代文字
+ 接着一个普通括号，里面放上图片的网址，最后还可以用引号包住并加上选择性的 'title' 属性的文字

`![image](link)`
![RUNOOB 图标](https://www.runoob.com/wp-content/uploads/2019/03/A042DF30-C232-46F3-8436-7D6C35351BBD.jpg "name of graph")


**总结** 都是链接的一种
使用直接链接时  
`[名称](地址)`  
变量时  
`[名称][变量名]`  
`[变量名]: 地址`  
图片的时候方括号前加`!`

Markdown功能简洁  
还没有办法指定图片的高度与宽度，如果你需要的话，你可以使用普通的 <img> 标签。  
`<img src="http://static.runoob.com/images/runoob-logo.png" width="50%">`

## 2.8. 表格
Markdown使用 `|`来制作表格 `-` 来分隔表头和其他行  
```markdown
表头|表头
--|--  
格子|格子
```  
表头|表头
--|--  
格子|格子

**设置表格的对齐方式**  
`-:` 设置内容和标题栏居右对齐。  
`:- `设置内容和标题栏居左对齐。  
`:-:` 设置内容和标题栏居中对齐。  
```
| 左对齐 | 右对齐 | 居中对齐 |
| :-----| ----: | :----: |
| 单元格 | 单元格 | 单元格 |
| 单元格 | 单元格 | 单元格 |
```  
| 左对齐 | 右对齐 | 居中对齐 |
| :-----| ----: | :----: |
| 单元格 | 单元格 | 单元格 |
| 单元格 | 单元格 | 单元格 |

`:`代表内容 `--`代表空白

## 2.9. 高级技巧  

### 2.9.1. 支持使用HTML元素
目前支持的 HTML 元素有：`<kbd> <b> <i> <em> <sup> <sub> <br>`等  
`使用 <kbd>Ctrl</kbd>+<kbd>Alt</kbd>+<kbd>Del</kbd> 重启电脑`
使用 <kbd>Ctrl</kbd>+<kbd>Alt</kbd>+<kbd>Del</kbd> 重启电脑

### 2.9.2. TeX LaTeX 数学公式

* 使用 `$公式内容$` 表示行内公式
  * $c = \sqrt{a^{2}+b_{xy}^{2}+e^{x}}$
* 使用 `$$表示块公式$$`, 会居中显示
  * $$c = \sqrt{a^{2}+b_{xy}^{2} +e^{x}}$$
