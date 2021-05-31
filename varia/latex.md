# 1. latex 

## 1.1. tex 世界构成

TeX的组成
1. Distributions: 以软件的形式存在的 TeX的具体发行版, 常见的有 :MiKTeX, TeX Live
2. Front ends and editors: 编辑器, 用于书写 TeX 文档
3. Engines: 一个用于解释TeX脚本的应用程序, 例如 `TeX, pdfTeX, XeTeX, LuaTeX`.
   * `pdfTeX` implements direct PDF output (which is not in Knuth's original TeX)
   * `LuaTeX` provides access to many internals via the embedded Lua language
4. Formats: 指实际书写的TeX语言格式
   * 例如 `LaTeX, plain TeX`, 他们都是 TeX-based languages. 
   * (`LaTeX` has meant “LaTeX2e” for many years now)
5. Packages: `geometry, lm`, … These are add-ons to the basic TeX system. 用于提供附加的 书写特性,字体,文档等.
   * 有些包可能与当前工作的 Engine 或者 Format 不兼容


根据Engine的不同, 有不同的TeX的输出格式:
1. The `pdfTeX` engine (despite its name) can output both `DVI` and `PDF` files. 
2. If you run `latex` (which implements the LaTeX format), you will get `DVI`
3. If you run `pdflatex` (which also implements the LaTeX format), you will get `PDF`. 
4. The `tex4ht` program can be used (run `htlatex`) to get `HTML, XML`, etc., output.
5. The `lwarp` LaTeX package causes LaTeX to output `HTML5`. 
6. The `LaTeXML` Perl program independently parses LaTeX documents and generates various output formats. 


## 1.2. latex 版本简介

## 1.3. latex 安装



# 2. latex 数学

过于重要, 在markdown也能用, 因此先介绍

## 2.1. 特殊字符及转义

* latex的特殊字符:
  * `# $ % & ~ _ ^ \ { }`
  * 使用这些特殊字符时需要加上转义符 `\`
* 在公式中添加空格 `\quad`
* 加减乘除等于 + - * / = 是直接输入的


### 2.1.1. 希腊字符

* 希腊字符在latex中有对应的表示代码
* 有些希腊字母的大写和英文一样因此没有特殊代码

字母表:
| 读音    | latex代码         | 示例                |
| ------- | ----------------- | ------------------- |
| 阿尔法  | \alpha A          | $\alpha A$          |
| 贝塔    | \beta B           | $\beta B$           |
| 伽马    | \gamma \Gamma     | $\gamma \Gamma$     |
| 得塔    | \delta \Delta     | $\delta \Delta$     |
| 伊普松  | \epsilon E        | $\epsilon E$        |
| 泽塔    | \zeta Z           | $\zeta Z$           |
| eta     | \eta H            | $\eta H$            |
| theta   | \theta \Theta     | $\theta \Theta$     |
| iota    | \iota I           | $\iota I$           |
| kappa   | \kappa K          | $\kappa K$          |
| lambda  | \lambda \Lambda   | $\lambda \Lambda$   |
| mu      | \mu M             | $\mu M$             |
| mu      | \mu N             | $\mu N$             |
| xi      | \xi \Xi           | $\xi \Xi$           |
| o       | o O               | $o O$               |
| pi      | \pi \PI           | $\pi \Pi$           |
| rho     | \rho P            | $\rho P$            |
| sigma   | \sigma \Sigma     | $\sigma \Sigma$     |
| tau     | \tau T            | $\tau T$            |
| upsilon | \upsilon \Upsilon | $\upsilon \Upsilon$ |
| phi     | \phi Phi          | $\phi \Phi$         |
| chi     | \chi X            | $\chi X$            |
| psi     | \psi \Psi         | $\psi \Psi$         |
| omega   | \omega \Omega     | $\omega \Omega$     |

### 2.1.2. 特殊运算符

特殊运算符:
| 名称   | 代码    | 显示       |
| ------ | ------- | ---------- |
| 加减   | \pm     | $\pm$      |
| 大乘   | \times  | $\times$   |
| 大除   | \div    | $\div$     |
| 点乘   | \cdot   | $\cdot$    |
| 合     | \cap    | $\cap$     |
| 并     | \cup    | $\cup$     |
| 大于   | \gt \ge | $\gt \geq$ |
| 小于   | \lt \le   | $\lt \le$    |
| 不等于 | \ne     | $\ne$      |
| 约等于 | \approx | $\approx$  |
| 全等   | \equiv  | $\equiv$   |

### 范围运算符
范围运算符
| 名称   | 代码  | 显示              |
| ------ | ----- | ----------------- |
| 求和   | \sum  | $\sum_{i=1}^n$    |
| 求积   | \prod | $\prod_{i=1}^n$   |
| 求极限 | \lim  | $\lim_{x\to1}x^2$ |
求积分|\int|$\int_{a}^{b}x^2$ 
多重积分|\多个i + nt| $\iiint$

## 2.2. 上下标和根号

* 上下标的字符多于一个的时候, 需要用 {} 括起来
  * ^ 表示商标
  * _ 表示下标
* \sqrt表示开方, 类似于一个函数, 默认开平方
  * 完整写法是 `\sqrt[开方字符]`
* $\sqrt[\alpha]\Gamma^{25}$


## 2.3. 分数

* 分数的表示也是类似函数, 需要填入两个空
  * `\frac{分子}{分母}`
  * 分数可以设置字号大小, 用引申出来的两个命令
    * `\dfrac`命令把字号设置为独立公式中的大小 默认
    * `\tfrac`则把字号设置为行间公式中的大小

## 多行公式

```
\begin{equation}
	\begin{split}
	\cos 2x &= \cos^2 x - \sin^2 x\\
	&= 2\cos^2 x - 1
	\end{split}
\end{equation}


\begin{equation}
	D(x) = \begin{cases}
	1, &\text{如果} x \in \mathbb{Q}\\%mathbb花体字符
	0, &\text{如果} x \in \mathbb{R}\setminus\mathbb{Q}	
		   \end{cases}%\text是为了在数学公式中处理中文
\end{equation}

```

# 3. latex 文档基础



## 3.1. 文档的基础

* `\documentclass{}`  定义在一开始, 说明文档的类型
* `begin{document}` `\end{documnet}`  内容的开始与结束

```
\documentclass{article}
\begin{document}

First document. 

\end{document}
```

## 3.2. preamble

定义在 ` \begin{document}` 叫做 latex 的preamble, 一般包含了:  
1.  the type of document you are writing
2.  the language
3.  load extra packages you will need
4.  set several parameters. 

`\documentclass[12pt, letterpaper]{article}`  
在方括号中加入别的参数, 用逗号隔开, 这里指定了字体和纸张大小

`\usepackage[utf8]{inputenc}`
指定要在该文档中使用的包  
解包 utf8 编码, 一般都会用该编码, 基本都有这一句  


文章介绍,  在这里定义不会直接显示在文档中
```
\title{First document}
\author{Hubert Farnsworth \thanks{funded by the Overleaf team}}
\date{February 2014}
```
## 3.3. 显示文档标题

```
\title{First document}
\author{Hubert Farnsworth \thanks{funded by the Overleaf team}}
\date{February 2014}

\begin{document}

\begin{titlepage}
\maketitle
\end{titlepage}

In this document.
\end{document}
```

* `\begin{titlepage} \end{titlepage}`  创建了一个名为 titlepage 的环境    
*  `\maketitle` 语句一键以默认格式创建标题    
*  `\thanks{}` 用于在作者中加入致谢, It will add a superscript and a footnote with the text inside the braces. 
  

# 4. latex 基础API

### 4.0.1. documentclass