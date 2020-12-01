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



# 2. latex 基础

## 2.1. 文档的基础

* `\documentclass{}`  定义在一开始, 说明文档的类型
* `begin{document}` `\end{documnet}`  内容的开始与结束

```
\documentclass{article}
\begin{document}

First document. 

\end{document}
```

## 2.2. preamble

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
## 2.3. 显示文档标题

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
  

# 3. latex 基础API

### 3.0.1. documentclass