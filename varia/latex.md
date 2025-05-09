# 1. latex 系统
* LaTeX 是一个免费软件, 遵循 terms of the LaTeX Project Public License (LPPL), 是核心引擎
* LaTeX 一般会作为一部分被包括在各种 Tex distribution 中, 并在 TeX User Group (TUG) or third parties 上发布

* Linux 和 Windows 上都可以使用 `TeX Live`, 是最被推荐的原汁原味的
* Windows 上还可以使用 `MiKTex` 
* MacOS 上就有对应的 `MacTeX`
* 目前有很多在线编辑的 TeX 服务, Papeeria, Overleaf, ShareLaTeX, Datazar, and LaTeX base 


## 1.1. LaTeX 的组成

1. Front ends and editors: 编辑器, 用于书写 TeX 文档
2. Engines: 一个用于解释TeX脚本的应用程序, 例如 `TeX, pdfTeX, XeTeX, LuaTeX`.
   * `pdfTeX` implements direct PDF output (which is not in Knuth's original TeX)
   * `LuaTeX` provides access to many internals via the embedded Lua language
3. Formats: 指实际书写的TeX语言格式
   * 例如 `LaTeX, plain TeX`, 他们都是 TeX-based languages. 
   * LaTeX 是建立在TeX基础之上的宏语言, 每一个命令会被转化成复数个TeX命令
   * (`LaTeX` has meant “LaTeX2e” for many years now)
4. Packages: `geometry, lm`, … These are add-ons to the basic TeX system. 用于提供附加的 书写特性,字体,文档等.
   * 有些包可能与当前工作的 Engine 或者 Format 不兼容


根据Engine的不同, 有不同的TeX的输出格式:
1. If you run `latex` (which implements the LaTeX format), you will get `DVI`
2. If you run `pdflatex` (which also implements the LaTeX format), you will get `PDF`. 
3. The `tex4ht` program can be used (run `htlatex`) to get `HTML, XML`, etc., output.
4. The `lwarp` LaTeX package causes LaTeX to output `HTML5`. 
5. The `LaTeXML` Perl program independently parses LaTeX documents and generates various output formats. 


TeX Live includes executables for TeX, LaTeX2e, ConTeXt, Metafont, MetaPost, BibTeX and many other programs.  

### 1.1.1. Engines

1. TeX: 由 高纳德编写 Donald Ervin Knuth
   - 1989年发布版本 3.0
   - TeX 只有300个命令, 且晦涩难懂
2. Plain TeX : 由 高纳德对 TeX 进行封装使之便于使用
   - 约 600 个命令
3. 基于 TeX 的系统, 由 Leslie Lamport 编写
   - 1984 年将自己使用TeX编写的宏封装, 方便没有程序设计知识的用户
   - 1993 年由 Latex3小组编写了 Latex2e, 并作为统一版本混乱局面的标准版本
* e-Tex: 1992 年提出的改进的 TeX
   - 没有推翻正式的 TeX 版本
   - 作为 e-TeX 被保留, 同时后续的 TeX 引擎大多数都是基于 e-TeX
* pdfTeX: TeX的附加组件
   - 传统 TeX 是针对印刷的
   - 该工具省略了 dvipdf 直接得到 pdf 文件
* LuaTex: TeX的附加组件
   - 由 pdfTex 的作者开发, 基于 pdfTex 并嵌入 Lua 脚本引擎
   - 成功篡位了 pdfTex
* XeTeX: 改进了原始 TeX 系统的字符集和字体缺陷
   - 2004 年发布, 支持 Unicode字符集, 2006年支持Windows和Linux
   - 2007 年纳入了 TeX Live 和 MikTeX 发行版
* pTeX & upTeX : 日系引擎
   - 由日本 ASCII公司的 大野和仓泽开发
   - 在 Unicode 时代前实现了 TeX 系统的日语化
   - upTeX 是pTeX 的 Unicode版本

### 1.1.2. latex 的语法

* latex 语法一般都以反斜杠 `\` 开始, 后接一串字符, 非英文字母的字符一般都不会出现命令名中
* latex 有一些内部语法, 它们以 `@` 开头, 要想正确处理 `@` 号来使用内部命令, 需要两个特殊命令
  * `\makeatletter`  表示从此处开始 `@` 是一个普通字符, 因此可以调用内部命令了 
  * `\makeatother`   表示对 `@` 视作普通字符的段落结束
  * 一些特殊格式需要用到内部语法时, 需要将相关命令写在这两个命令中间


## 1.2. Tex Live


预定义的 texmf 目录树
| 宏名           | 功能                                                     |
| -------------- | -------------------------------------------------------- |
| TEXMFDIST      | 存储除了二进制文件以外的几乎所有文件, 脚本, 宏包, 字体等 |
| TEXMFSYSVAR    | 全局 VAR                                                 |
| TEXMFVAR       | 用户 VAR                                                 |
| TEXMFSYSCONFIG | 全局 config                                              |
| TEXMFCONFIG    | 用户 config                                              |
| TEXMFLOCAL     | 管理员用来安装供整个系统使用的额外的或更新过的宏包、字体 |
| TEXMFHOME      | 用户存放它们自己独立安装的的宏包、字体                   |
| TEXMFCACHE     | 保存运行时候的缓存数据, 默认等同于 TEXMFSYSVAR           |

* 配置文件 config 管理的程序 : texconfig-sys, updmap-sys 和 fmtutil-sys


* Tex系统目录
  * Unix  : /usr/local/texlive/202* (不同版本的目录不同)
    * bin  存放可执行文件, 根据平台的不同有不同的子文件
      * x86_64-linux  GNU/Linux 二进制文件 (64 位)
      * win32         Windows 二进制文件
    * texmf-dist      TEXMFDIST, TEXMFMAIN
    * texmf-var       TEXMFSYSVAR, TEXMFCACHE
    * texmf-config    TEXMFSYSCONFIG

* 用户目录 (个人生成和配置的数据)
  * Unix  : ~/texlive202* (不同版本的目录不同)
    * texmf-var       TEXMFVAR, TEXMFCACHE
    * texmf-config    TEXMFCONFIG



如果要删除 
* rm -rf /usr/local/texlive/2021
* rm -rf ~/.texlive2021


### 1.2.1. 安装

- linux 下安装

0. 事先需要安装 perl 运行环境
1. 使用 `sudo perl install-tl` 进行安装, 不用 sudo 的话需要更改安装位置
2. 大约7000mb空间, 一个小时时间

配置环境变量
```sh
PATH=/usr/local/texlive/2020/bin/i386-linux:$PATH; export PATH
MANPATH=/usr/local/texlive/2020/texmf-dist/doc/man:$MANPATH; export MANPATH
INFOPATH=/usr/local/texlive/2020/texmf-dist/doc/info:$INFOPATH; export INFOPATH
```
### 1.2.2. tlmgr  TexLive的管理程序

管理安装后的系统: 
* 列出 方案 (scheme)，集合和安装包；
* 安装、升级、备份、恢复、卸载软件包，并且能自动计算依赖关系；
* 查找和列出软件包以及它们的描述；
* 列出、添加和删除不同平台的可执行文件；
* 改变安装选项，比如纸张大小和源文件位置 (参见第 3.3.1 节)

[完整文档](https://tug.org/texlive/doc/tlmgr.html#info)

## 1.3. Latex 编译
                                                                                                                                                                     
源文件:  
* tex     : 即书写文档的 latex 文件
* cls     : 定义 latex 的格式文件, 定义了排版格局, 通过 `\documentclass{}` 导入, 称为类文件
* sty     : 宏包文件 `package` 使用 `\usepackage{}` 导入, 称为风格文件
* bst     : 参考文件的格式文件, 通过 `\bibliographystyle{}` 导入的就是这种文件
* bib     : 用户定义的参考文献库, 通过 `\bibliography{}` 导入

带参考文献的文档编译需要4次编译:  
1. `(xe/pdf)latex main.tex`:
   * 生成 `.aux .log .pdf`
   * aux    : 引用标记记录文件, 用于再次编译时生成参考文献和超链接
   * pdf    : 此时正文中的引用显示为`[?]`且文章末尾没有参考文献
2. `bibtex main.aux`
   * 生成 `.bbl blg`
   * bbl    : 编译好的bib文件, 将原本的bib格式转化成了 `thebibliography` 原始的引用类型
   * blg    : 日志文件
3. `(xe/pdf)latex main.tex`
   * 更新了 `.aux .log .pdf`
   * 此时 pdf 文件末尾已经有了参考文献列表, 但是在正文中的引用仍为 `[?]`
4. `(xe/pdf)latex main.tex`
   * 再次更新了 `.aux .log .pdf`
   * 生成了最终的 pdf 文件, 正文中的引用也标好了序号


### 1.3.1. 多种编译器的区别

* tex     : 编译 tex 源文件生成 dvi 文件
* pdftex  : 编译 tex 源文件生成 dvi 文件
* latex   : 编译 latex 源文件生成 dvi 文件
* pdflatex: 编译 latex 源文件生成 pdf 文件

### 1.3.2. 附带程序

* bibtex, biber     : 参考文献支持
* makeindex, xindy  : 索引支持

* dvips             : dvi 文件转换成 postscript 文件
* dvipdf            : dvi 文件转化成 pdf 文件
* dvipdfmx          : dvi 转换成 pdf 的改进程序

### 1.3.3. latex 家族

根据使用语言的不同, latex 编译器被区分出来了数个家族

p系列(中日韩)
| 名称    | 说明                                                             |
| ------- | ---------------------------------------------------------------- |
| pTex    | pTex 汉字假名记号使用的是 日本的标准 (JIS X 0208)                |
| upTex   | 在pTex 的基础上增加了 CJK和朝鲜语的支持, 以及utf-8编码的支持     |
| e-upTex | 合并 upTex 和一些 eTex 的功能, 目前 upTex 已经完全合并了 e-upTex |
即当前编译日文文章的话, 直接使用 uptex 即可

## 1.4. latex 包总结

文字格式
* bm          定义 `\bm` 命令用于粗体化数学公式

文章格式
* titlesec    更改默认的 Section 等 格式

图
* epsfig      用于插 eps 格式的图片的专有包
* graphicx    高级插图命令
* subfigure   大小子图命令


布局风格 上下栏
* fancyheadings
* fancyhdr

引用
* cite
* overcite    所有引用自动作为上标


特殊
* comment     注释




## 1.5. latex-workshop vscode

[url](https://github.com/James-Yu/LaTeX-Workshop/wiki/Install)

配置自定义的编译流程, 在 `settings.json` 中加入字段
* `latex-workshop.latex.tools` 工具选项, 配置 tools
* `latex-workshop.latex.recipes` 配置编译流程, 可以使用多个 tools




# 2. latex 数学

* latex 数学被称为 math-mode
* 和文本模式 `text-mode` 是区分开的
* 相关的环境会自动进入 `math-mode`

* latex的特殊字符:
  * `# $ % & ~ _ ^ \ { }`
  * 使用这些特殊字符时需要加上转义符 `\`
* 在公式中添加空格 `\quad`
* 加减乘除等于 + - * / = 是直接输入的


* 希腊字符在latex中有对应的表示代码
* 有些希腊字母的大写和英文一样因此没有特殊代码

### 2.0.1. 希腊字母表
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

特殊运算符:
| 名称         | 代码                 | 显示        |
| ------------ | -------------------- | ----------- |
| 加减         | \pm                  | $\pm$       |
| 大乘         | \times               | $\times$    |
| 大除         | \div                 | $\div$      |
| 点乘         | \cdot                | $\cdot$     |
| Hadamard积   | \odot                | $\odot$     |
| 不知道       | \oplus               | $\oplus$    |
| 不知道       | \ominus              | $\ominus$   |
| 不知道       | \otimes              | $\otimes$   |
| 合           | \cap                 | $\cap$      |
| 并           | \cup                 | $\cup$      |
| 大于         | \gt \ge \textgreater | $\gt \geq$  |
| 小于         | \lt \le \textless    | $\lt \le$   |
| 不等于       | \ne                  | $\ne$       |
| 约等于       | \approx              | $\approx$   |
| 约等于等于   | \approxeq            | $\approxeq$ |
| 波浪号, 相关 | \sim                 | $\sim$      |
| 波浪号等于   | \simeq               | $\simeq$    |
| 原型         | \propto              | $\propto$   |
| 全等         | \equiv               | $\equiv$    |
| 属于         | \in                  | $\in$       |
| 存在         | \exists \exist       | $\exists$   |
| 不存在       | \nexist \nexists     | $\nexists$  |
| 全部         | \forall              | $\forall$   |
| 逼近, 箭头   | \to                  | $\to$       |

## 2.1. 基础数学commands

求和, 上下标等大部分数学符号都需要用命令来输入  

### 2.1.1. 上下标和根号

* 上下标的字符多于一个的时候, 需要用 {} 括起来
  * ^ 表示上标
  * _ 表示下标
* \sqrt表示开方, 类似于一个函数, 默认开平方
  * 完整写法是 `\sqrt[开方字符]`
  * $\sqrt[\alpha]\Gamma^{25}$
* 小撇直接用 ' 即可

字符上方加标识
| 功能 | 代码        | 显示           |
| ---- | ----------- | -------------- |
| 点   | \dot{}      | $\dot{a}$      |
| 多点 | \ddot{}     | $\ddot{a}$     |
| 帽子 | \hat{}      | $\hat{a}$      |
| 反帽子 | \check{}      | $\check{a}$      |
| 波浪 | \tilde      | $\tilde{a}$    |
| 箭头 | \vec{}      | $\vec{x}$      |
| 小圆 | \mathring{} | $\mathring{a}$ |
| 横线 | \bar        | $\bar{a}$      |

### 2.1.2. 大运算符 范围运算符

范围运算符号, 基本上表示范围的算是都是直接用上下标方法来输入
| 名称         | 代码        | 显示               |
| ------------ | ----------- | ------------------ |
| 求积分       | \int        | $\int_{a}^{b}x^2$  |
| 多重积分     | \多个i + nt | $\iiint$           |
| 曲线积分     | \oint       | $\oint$            |
| 多重曲线积分 | \oiint      | $\oiint$           |
| 求和         | \sum        | $\sum_{n=1}^Na_n$  |
| 求积         | \prod       | $\prod_{n=1}^Na_n$ |
| 求开方         | `\sqrt[n]`      | $\sqrt[n]{x}$ |


### 2.1.3. 特殊运算
| 名称           | 代码                   | 显示                      |
| -------------- | ---------------------- | ------------------------- |
| 导数           | \nabla{f}              | $\nabla{f}$               |
| 微分           | \partial{y}            | $\partial{y}$             |
| 求极限         | \lim                   | $\lim_{x\rArr1}x^2$       |
| 文字置于正下方 | underset{under}{upper} | $\underset{x\to 0}{\lim}$ |



### 2.1.4. 特殊格式字符

| 名称         | 代码          | 显示             |
| ------------ | ------------- | ---------------- |
| 实数集空心字 | \mathbb{R}    | $\mathbb{R}$     |
| 李代数       | \mathfrak     | $\mathfrak{so}$  |
| 粗体整体     | \mathbf       | $\mathbf{x}$     |
| 粗体保持斜体 | \boldsymbol{} | $\boldsymbol{x}$ |



### 2.1.5. 分数

* 分数的表示也是类似函数, 需要填入两个空
  * `\frac{分子}{分母}`
  * 分数可以设置字号大小, 用引申出来的两个命令
    * `\dfrac`命令把字号设置为独立公式中的大小 默认的 frac
    * `\tfrac`则把字号设置为行间公式中的大小


## 2.2. 界定符

像是跨于多行的圆括号, 方括号, 用来表示矩阵等内容, 都可以用界定符的形式来描述 
* 界定符可以是 `( [ ] ) | \{ \}` 大括号需要加转义字符
* 左侧:
  * `\left` 后面跟符号 `\left.` 代表空界定符
  * `\bigl`
  * `\Bigl`
  * `\biggl`
  * `\Biggl`
* 右侧同理 `\right` 或者相应的 l 改为 r


## 2.3. 公式环境

在正式的文本中书写公式需要引入公式环境

简写公式环境, 不会产生公式编号：
1. `$equation$`
2. `$$equation$$`       该格式的公式会自动居中, 单独占一行, 不会嵌入到正文中
3. `\[ equation \]`     该格式的公式会根据配置的全局对齐方式来对齐


* 除了简写公式环境以外, 有公式专用的标准环境, 这些环境都会将里面的所有字符是做公式字符, 不用再在里面输入 `$ $`
* 非简写公式环境都会自动参与编号, 编号会自动生成在最右边, 排版时可以认为页面的右边缘被向左移动了一个编号占用的距离
* 不要参与编号的话就在环境名称结尾加一个星号`*`, begin 和 end 都要加
1. align
2. equation   : 最基础
3. eqnarray
4. gather

### 2.3.1. equation

* equation 是最一般的公式环境, 表示一个公式, 默认表示一个单行的公式
* 可以通过内嵌其他环境进行拓展, 例如对齐环境
$$\begin{equation*}
	\begin{split}
	\cos 2x &= \cos^2 x - \sin^2 x\\
	\end{split}
\end{equation*}
$$

$$
\begin{equation*}
	D(x) = \begin{cases}
            0, &\text{如果} x \in \mathbb{R} \setminus \mathbb{Q} \\
            1, an	
         \end{cases}
         %\text是为了在数学公式中处理中文
\end{equation*}$$
```tex
\begin{equation}
	\begin{split}
	\cos 2x &= \cos^2 x - \sin^2 x\\
	\end{split}
\end{equation}


\begin{equation}
	D(x) = \begin{cases}
            0, &\text{如果} x \in \mathbb{R}\setminus\mathbb{Q}	
         \end{cases}   
         %\text是为了在数学公式中处理中文
\end{equation}

```

### 2.3.2. align

* align 是最基本的对齐环境, 而不能被称作标准公式环境, 因此可以说是公式环境的基础
* align 和表格环境一样, 使用 `&` 分割单元, `\\` 换行
* 每一行都是独立的公式, 都会独立的编号
* `&` 分割出来的单元为单位进行对齐, 成为组, 每个组都可以指定特定排版, 相当于表格的列

```tex
\begin{align*}
    f(x)  &= (x+a)(x+b)         \\
          &= x^2 + (a+b)x + ab
\end{align*}

```

### 2.3.3.  amsmath - 矩阵专用

amsmath 中有一些很方便的矩阵环境, 用于便捷的书写矩阵  

语法都为 : `\begin{环境名} &用来分元素 \\用来分行 \end{环境名}`
* matrix : Plain 矩阵, 没有括号
* pmatrix : Parenthesess, round brackets . 圆括号
* bmatrix : Brackets, square brackets . 方括号
* Bmatrix : Braces, curly brackets. 花括号
* vmatrix : Pipes, 直线, 即用于表示行列式的情况
* Vmatrix : Double pipes, 双直线
* cases : 单边花括号

$$
 	\begin{matrix}
1 & 2 & 3\\
a & b & c
\end{matrix}
$$

$$
\begin{pmatrix}
1 & 2 & 3\\
a & b & c
\end{pmatrix}
$$


## 2.4. latex 矢量图



# 3. 文字格式

文字格式可以被分成四个模块
1. family  字体                默认: serif family
2. series  细, medium, 加粗    默认: medium
3. shape   斜体等              默认: upright
4. size    字号                默认: 10pt

标准库中的格式
* family
  * Serif 默认   
  * Sans serif
  * Typewriter
* series
  * Medium
  * Boldface series
* Shape
  * Upright
  * Italic
  * Slanted
  * Caps & Small cap
  * emphasized
* size

## 3.1. size 字号设置

部分改变字号, 可以使用环境, 直接将字号代码写在 begin end 命令里


| 名称   | pt      | mm       | 代码          |
| ------ | ------- | -------- | ------------- |
| 七号   | 5.25pt  | 1.845mm  | \tiny         |
| 六号   | 7.875pt | 2.768mm  | \scriptsize   |
| 小五号 | 9pt     | 3.163mm  | \footnotesize |
| 五号   | 10.5pt  | 3.69mm   | \small        |
| 小四号 | 12pt    | 4.2175mm | \normalsize   |
| 四号   | 13.75pt | 4.83mm   | \large        |
| 三号   | 15.75pt | 5.53mm   | \Large        |
| 二号   | 21pt    | 7.38mm   | \LARGE        |
| 一号   | 27.5pt  | 9.48mm   | \huge         |
| 小初号 | 36pt    | 12.65mm  | \Huge         |


## 3.2. 注释 comment

comment 包

https://ftp.yz.yamagata-u.ac.jp/pub/CTAN/macros/latex/contrib/comment/comment.pdf

# 4. 文章 layout / Style

首先设置各种页面布局 (layout) 和风格  

1 pt ≈ 0.3515 mm  
1 英尺= 25.4 mm

## 4.1. Page Layout

用于定义物理上的纸张大小和各种页边距

* `\documentclass[ 可选设置指定纸张大小 ]{ 必须参数指定文章类型 }`
* 该命令应该放在源文件一开始
* 纸张大小会影响对应边界设置等命令的效果 (因为命令的参数单位是 mm)
* 除此之外 \paperheight and \paperwidth 用于指定特殊的纸张大小

纸张大小的表

| 纸张大小              | 具体 Size     | 具体 PT          |
| --------------------- | ------------- | ---------------- |
| a4paper               | 210 mm×297 mm | 595.35 x 841.995 |
| a5paper               | 148 mm×210 mm | 419.58 x 595.35  |
| b5paper               | 176 mm×250mm  |
| letterpaper (default) | 216 mm×279 mm |
| legalpaper            | 216 mm×356 mm |
| executivepaper        | 184 mm×267 mm |




* 注意对纸张来说, 横纵各有1英尺的基础偏移 (25.4 mm)
* \hoffset 和 \voffset 是控制各种命令的参考点的位置, 默认都为0, 定义了纸张坐上点为各种模块的参考点
* 以此为基准来控制各种长度的表如下:
* 该表中的命令为 Mathematical expression
* 对应的命令 command 语句为  `\setlength{ \textwidth }{ 数值 }`

| 模块           | 命令            | 功能                                                                                 |
| -------------- | --------------- | ------------------------------------------------------------------------------------ |
| 文本           | \textheight     | Height of main texts without header and footer                                       |
| 文本           | \textwidth      | Width of main texts without marginal notes                                           |
| 文本           | \oddsidemargin  | Blank space on the left margin of odd-numbered pages, if both-side printing is opted |
| 文本           | \evensidemargin | Blank space on the left margin of even-numbered page                                 |
| 列属性         | \columnsep      | Gap between two columns in multi-column mode                                         |
| 列属性         | \columnseprule  | 在两列模式下中间加一个线用于区分两边, 默认是0代表不存在线                            |
| 列属性         | \columnwidth    | Width of a column in multi-column mode ( 默认根据 \textwidth and \columnsep 来计算). |
| 列属性         | \linewidth      | Width of the lines of texts (usually equal to \columnwidth)                          |
| Header         | \headheight     | Height of the header                                                                 |
| Header         | \headsep        | Vertical gap between the header and the first line of the main texts                 |
| Header         | \topmargin      | Extra vertical space above the header.                                               |
| Header         | \headrulewidth  | 定义 header 下方线的粗细                                                             |
| Footer         | \footrulewidth  | 定义 footer 上方线的粗细                                                             |
| Footer         | \footskip       | Vertical gap between the last line of the main texts and the footer.                 |
| Marginal notes | \marginparwidth | Width of marginal notes.                                                             |
| Marginal notes | \marginparsep   | Horizontal gap between the main texts and marginal notes                             |
| Marginal notes | \marginparpush  | Vertical space between two successive marginal notes.                                |

![效果图片](./latex_page_layout.png)


## 4.2. Page Style

本质上, Page Style 指代的是每一页的 header, footer, page number 等正文以外的修饰信息  

* `\pagestyle{ }`     command implements the chosen page style on the current page as well as on the succeeding page
* `\thispagestyle{ }` works locally on the current page only
  * `\thispagestyle{empty}` may be used in the title page of a book or a report.
  * `\thispagestyle{plain}` is issued by the:
    * document-classes of article, book, and report to the `\maketitle` command
    * first page of major sectioning commands like `\part{ }` or `\chapter{ }`
    * 为了阻止这个被自动调用的 plain 导致一些章标题页被加了页码, 需要在每个`\maketitle`, `\part{ }` or `\chapter{ }` 命令后 手动加上 `\thispagestype{} e.g. empty`

| Page Stype | 功能                                                      | 存在的包      |
| ---------- | --------------------------------------------------------- | ------------- |
| empty      | head 和 foot 都为空, 也没有页码                           |
| plain      | 没有 head, foot 只有页码, 是 article 和 report 的默认格式 |
| headings   | 没有 foot, head 有页码和该页的标题                        |
| myheadings | 同 headings, 但是 页标题需要用别的命令提供格式            |
| fancy      | 完整用户自定义 header 和 footers                          | fancyheadings |
| fancy      | 完整用户自定义 header 和 footers                          | fancyhdr      |


* headings : 默认的带页标题的格式, 依据 documentclass 和是否是 twoside 文档来定义章标题的格式和页码的位置


### 4.2.1. myheadings Style

The only difference is that the style of the header under `headings` is predefined, while it is user-defined under `myheadings`.

* 关于 header 中页标题的相关信息需要通过 marker 命令来设置
* `\markboth{ aeven }{ aodd }` 只在 twoside 生效, 用于指定 左右页的不同的 style
  * 页码在左页的时候会放在左边界
  * 页码在右页的时候会放在右边界
* `\markright{ acont }`  用于指定 oneside 时候的 style, 会作用在所有 page
* 有时候左右页想显示的层级不同, 需要不同的信息, 通过以下两个命令的结合即可实现
  *  `\markboth{aeven}{ }`
  *  `\markright {aodd}`


| 关联的命令       | marker 命令     |
| ---------------- | --------------- |
| `\chapter{ }`    | \chaptermark    |
| `\section{ }`    | \sectionmark    |
| `\subsection{ }` | \subsectionmark |

* 由于 myheadings 是具有 predefined 的风格, 因此自定义风格的时候需要使用 `\renewcommand{ }[ ]{ }` 命令
  * 可以在可选参数 `[]` 中设置为 `[1]`, 在命令定义中使用 `#1` 来导入 unit 的对应名字
* 可以用到的相关 label 命令:
  * `\chaptername` : generates the label-word Chapter
  * `\thechapter`, `\thesection`, `\thesubsection` : generate the serial numbers of the current unit

```latex
% 用于 twoside 的风格定义 book / report

% Chapter 1. Headers and Footers
\renewcommand{\chaptermark}[1]{\markboth{\chaptername∼\thechapter. #1}{ }}

% 1.2. Generating Headers
\renewcommand{\sectionmark}[1]{\markright{\thesection. #1}}

% 用于 oneside 的风格, article

\renewcommand{\sectionmark}[1]{\markboth{\thesection. #1}{ }}
\renewcommand{\subsectionmark}[1]{\markright{\thesubsection. #1}}
```

### 4.2.2. fancyheadings Style

定义在 fancyheadings 包里的 fancy Style 是完全自定义的

给上下标画线
* \headrulewidth  默认是 0.4 pt
* \footrulewidth  默认是 0 pt

fancyheadings 包的 Style 定义命令:
* peven 选项只有 twoside 的时候才生效

| Commands for headers  | Commands for footers  | Alignment |
| --------------------- | --------------------- | --------- |
| `\lhead[peven]{podd}` | `\lfoot[peven]{podd}` | Left      |
| `\chead[peven]{podd}` | `\cfoot[peven]{podd}` | Center    |
| `\rhead[peven]{podd}` | `\rfoot[peven]{podd}` | Right     |


```latex
% 导入包. 使用包
\usepackage{fancyheadings}
\pagestyle{fancy}


% 先重定义 marker
\renewcommand{\chaptermark}[1]{\markboth{\thechapter. #1}{}}
\renewcommand{\sectionmark}[1]{\markright{\thesection. #1}}

\lhead[\textbf{\thepage}]{\textbf{\rightmark}}
\rhead[\textbf{\leftmark}]{\textbf{\thepage}}



```

# 5. 文章管理

* `\documentclass{}`  定义在一开始, 说明文档的类型
* `begin{document}` `\end{documnet}`  内容的开始与结束

```latex
\documentclass[]{article}


\begin{document}
First document. 
\end{document}
```


## 5.1. preamble

定义在 ` \begin{document}` 叫做 latex 的preamble, 一般包含了:
1. the language
2. load extra packages you will need
3. set several parameters. 

文章介绍,  在这里定义不会直接显示在文档中
```latex
\title{First document}
\author{Hubert Farnsworth 
\thanks{funded by the Overleaf team}}
\date{February 2014}
```

### 5.1.1. 显示文档标题

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
## 5.2. Formatting

文章的 formatting 最主要的就是分章分节  

### 5.2.1. Sectional Units

* 分段分章是文档的必备属性. 在不同类型的文章中有不同类型的编号体系  
* 一般来说编号只编三级
* 使用加星号的命令可以阻止编号 `\chapter*{}`   

book/report:
1. `\chapter{}`     1
2. `\section{}`     1.1
3. `\subsection{}`  1.1.1

article
1. `\section{}`       1
2. `\subsection{}`    1.1
3. `\subsubsection{}` 1.1.1

### 5.2.2. Label and Referring

可以在任何位置分配label, 并使用 `\ref` 进行页内跳转
1. `\label{key}`  分配名称
2. `\ref{key}`    进行跳转
* 大部分对象都可以分配label, 命令连起来即可, 即`\section{a}\label{sec-a}`
* `ref{key}` 会显示成 label 章节的编号, 不带有文字, 即 `2.1`


可以分配label的, 会自动编号的项目:
* table
* figure
* equation
* `\item`

其他的referance:
* `\pageref{}` 显示成label所在的页码
* 带有 `v` 的command,`\vref` 定义在了  varioref 包中

### 5.2.3. 行与段落 lines and paragraphs

在源文件中换行并不会使输出文档换行, 而是被识别成一个空格,  
(因此源文件中为了书写整洁而换行时, 不需要在换行前加额外的空格)  
必须通过命令或者特定字符才能创建新行, 新行有两种类型, `分段`和`断行`

* 分段:
  * 会将两个行认作是两个段, 段首缩进会分别应用在两个部分
  * `\par`
  * 两个换行符, 
* 断行:
  * 一般来说文字超过纸张宽度后会自动换行, 这种方法属于手动换行
  * 因此不会被人做是两个段, 没有段首缩进
  * `\\` 建立新行, 可以指定空几毫米 `\\[2mm]`
  * `\newline` 建立新行
  * `\linebreak` 建立新行, 并强制回车前的最后一行拉伸填满页面的宽
  * `\\\\` 建立两个新行, 即空一行
* 阻止断行
  * 在某些情况下希望某个带空格的单词不被断行分开. 但这有可能导致文字超过行的宽度
  * `mbox{}`
  * 使用 ` ~ ` 或者 `\,` 过于手动不推荐
    * `10\,inch` `Dilip~Datta`



相比于行, latex更推荐通过段落命令来控制文章
* `\par` 无参数的, 使用默认首行缩进
  * 可以有附加命令跟在 `\par \附加命令` 后面用来指定别的格式
  * (全局)`\parindent = ?mm` 手动指定首行缩进 , 会作用于该命令之后的全部段落
  * (全局)`\parskip ?mm`  段上方的空白距离    , 会作用于该命令之后的全部段落
  * `\noindent` 无首行缩进
* `\paragraph{} \subparagraph{}`
  * 段首会有加粗的字
  * subparagraph 会有更多的段首缩进 
  * 在某些 cls 下会分配编号

## 5.3. titlesec 更改章节标题格式

* `\usepackage[center]{titlesec}`
  * 可选参数 center 可使标题居中, raggedleft 默认居左, raggedright 居右
* `\titleformat{command}[shape]{format}{label}{sep}{before}[after]`
  * command : 要重新定义的标题命令, \part, \chapter, \section, \s section, \s s section, \paragraph, \s paragraph
  * shape   : 设定段落形状, hang, block, display 等
  * format  : 定义标题外观, 字体加粗等
  * label   : 定义标题的标签, 即 标签内容前面的标号
  * sep     : 定义标题的标签与标题内容之间的间隔距离
  * before  : 在标题内容前加的内容
  * after   : 在标题内容后加的内容

一般使用的时候, 只会定义 format 和 label

# 6. latex Syntax 部件语法

一个tex文件可以简单的分解成2部分
* preamble  : 保存了全局的处理参数 `documentclass{}`
* body      : 文档的内容        `\begin {document}`

编译一个 tex 文档会有几个步骤
* 会生成 `.aux .log .dvi` 几个文件 `.dvi` 是最终输出的可视化文件
* `.dvi` 文件可以被转化成 `.ps .pdf` 文件


## 6.1. 基础class

Latex语法包含了两大类别:
* latex command
* latex environment
* 定义在别的文件中的不属于标准文档类的 command 或者 environment 称为packages

### 6.1.1. 基础字符

以下字符可以直接被打印到文档中
1. 英文字母和数字
2. 两种括号 `[] ()`
3. 5个数学符号 `+-*/=`
4. 断句符号 `,:;!.?`
5. 引号 ` " ' 
6. at @

其他的所有符号需要转义 `\verb""  或者 \verb!!` 写在两个引号or感叹号中间


横线有三种长度:
* `-` 用于链接两个单词 multi-language
* `--` 用于指定范围 A--B
* `---` 用于补充说明

### 6.1.2. commands

latex command 的属性可以表示成:
* 一般以 `\` 开始的一个指令
* 指令一般都是以英文字母组成
* 空参数后要接空格 `\command 字符` 或者`\command\ 字符`

### 6.1.3. environment

用于实现特殊功能 插入公式
* ename作为一个环境名称, 开启一个环境用 `\begin{ename} \end{ename}`
* 环境可以嵌套(必然)
* 环境也是有参数的 `\begin{ename}{p1}{p2} \end{ename}`
* 环境也是有可选参数的 `\begin{ename}[op]`

### 6.1.4. packages

packages:
* 在文档的 preamble 里载入, 即 `\documentclass{}` 和 `\begin{document}` 的中间载入
* 载入包的代码是 `\usepackage{pname}`
* 加载一个包也有可选参数 `\usepackage[p1]{pname}`
* 包的参数定义只对包中的feature生效, 而`\documentclass`是对整个文档生效, 包括加载的包

`\usepackage[utf8]{inputenc}`
指定要在该文档中使用的包  
解包 utf8 编码, 一般都会用该编码, 基本都有这一句  


### 6.1.5. documentclass

作为一开始的语句, `\documentclass[]{}` 具有设置该文档种类的功能, latex最基础的几大class是  
* letter
* article
* report
* book

1. 每一个文档类都有不同的可选参数, 以及对应的标准 commands 来生成该类文档的不同部分  
2. 每一个文档类都是一个 `.cls` 的文件, `\documentclass{article}`代表引入了`article.cls`


这些都是标准库的commands, 并不是必须要使用, 只是可以被使用
1. `\address{地址}` 送信人的地址, 会放到 top-right
2. `\signature{签名}` 送信人的签名, bottom-centre
3. `\begin{letter}{收信人地址}` 收信人地址会放到正文的左上
4. `\opening{问候}` 信件开头的问候
5. `\closing{Best regards,}` 信件结尾的问候
6. `\cc{copy}` send copy
7. `\encl{Enclosure}` 信件的附件

* 地址和签名都不属于 letter 环境中的内容 
* 信件类型的文档会默认自动插入日期, 可以通过 `\date{29/02/2016}` 命令更改

```tex
%File Name: myletter.tex
\documentclass[a4paper,12pt]{letter}

\begin{document}

\address{Sender’s Address}
\signature{Sender’s Name}

\begin{letter}{Recipient’s Address}

\opening{Dear Sir,}Contents of the letter...

\closing{Best regards,}
\cc{1. Secretary\\2. Coordinator}
\encl{1. Letter from CEO.\\2. Letter from MD.}
\end{letter}
\end{document}
```

## 6.2. 界面宏

LaTex有一些方便布置表格和图片的界面宏, 用于快速指定宽度 

| 宏             | 功能               |
| -------------- | ------------------ |
| `\linewidth`   | 当前行的宽度       |
| `\columnwidth` | 当前分栏的宽度     |
| `\textwidth`   | 整个页面版芯的宽度 |
| `\paperwidth`  | 整个页面纸张的宽度 |



# 7. 图表环境


## 7.1. Table

latex中处理表格的环境
* 传统表格, 不能分配序列号和标题, 文本为基础的独立对象, 容易在创建超大表格的时候出问题
  * tabular
  * tabularx
* longtable
* 其他相关环境
  * table
  * wraptable
  * sidewaystable

### 7.1.1. tabular

基础表格环境 tabular

* 环境指令 `\begin{tabular}[]{acols}`
  * `{acols} eg. {|l|c|r|}`参数用来指定每列的字母用 `lcr`, 并包括在`|` 中, 分别代表左中右对齐
  * `|` 其实代表每列用竖线分隔开
  * 注意 tabular的特点是所有列等宽, 宽度以所有列的最宽数据为基准
* 在环境中
  * 同一行的不同元素用 `&` 隔开
  * 结束表格的一行用 `\\`
  * `\hline` 用于在表格上画横线用于分隔行, 在内容的上下都应该加上
```latex
\begin{tabular}{|l|c|c|c|c|}
  \hline Name & Math & Phy & Chem & English\\
  \hline Robin & 80 & 68 & 60 & 57\\
  \hline Julie & 72 & 62 & 66 & 63\\
  \hline Robert & 75 & 70 & 71 & 69\\
  \hline
\end{tabular}
```

### 7.1.2. tabularx

加强版的表格环境, 解决了
1. tabular 的所有列等宽的问题, 通过自动计算所有列宽, 来减少表格越界的问题
2. 可以指定表格的最终宽度

用法
* 环境指令 `\begin{tabularx}{awidth}{acols}`
  * `{awidth}` : 指定表格的最终水平宽度
  * `{acols}`  : 指定表格的列. 和 tabular 一样的参数
    * `lcr`, 分别代表左中右对齐, 固定宽度的列
    * `X`  , 自动计算宽度, 所有的`X`列等宽, 但是宽度是根据 `{awidth}` 来计算
    * `X` , 列的文字是完全对齐, 容易不美观, 可以附加其他的对齐方式
    * `>{\raggedright\arraybackslash}`, `>{\centering\arraybackslash}`, or `>{\raggedleft\arraybackslash}` before `X`
    * `\arraybackslash` 是一个修正代码, 如果不使用的话, 可能导致表格中的 `\\`换行在某些情况下失效
* 宏数据
  * `\linewidth` 用来返回当前的页面宽度, 前面接倍数可以方便的指定表格宽度

```latex
\begin{tabularx}{0.8\linewidth}{|X|c|>{\raggedleft\arraybackslash}X|}
  \hline {\bf Name} & {\bf Sex} & {\bf Points}\\
  \hline  Milan & M & 1,500\\
          Julie & F & 1,325\\
          Sekhar & M & 922\\
          Dipen & M & 598\\
          Rubi & F & 99\\
  \hline
\end{tabularx}

```

### 7.1.3. table

封装的较常用的表格环境, 基础的 tabular 会嵌套在该环境中, 主要用来为表格
1. 将表格创建在分离的一段
2. 为表格加入标题和编号
3. 指定表格在页面的垂直位置

* 指令 `\begin{table}[!hbt]`
  * 可选参数 `eg. [!hbt]`, 用来指定表格的在页面的垂直位置, 该参数的正式名称是 `[avp]`
  * `htb` 代表3个不同的表格位置, 三个位置可以任意单独或者组合输入, 组合输入时效果是无关顺序的
  * `!`, 感叹号参数具有最高优先级, 如果使用了, 代表表格可以忽略一些限制来确保表格能够被正确放到对应位置上
  * `h`, 代表 here, 也是默认参数, 放在当前位置, 如果当前位置没空间且位置参数只有 `h`, 则尝试放在下一页的 `t` , 当前页的剩余空白则填入源文件中表格后面的文字
  * `t`, 代表 top, 表格尝试放在当前页的最上方
  * `b`, 代表 bottom , 当前页的最下方
  * 除了 `!htb` 以外, 在 `float` 包中还有一个独立的参数 `H`, 代表原本 `h`中, 表格没空间时, 当前页的剩余空白就那么空白着, 不填入表格后面的文字
* 内容
  * 对齐方法
    * `\centering` : 表格居中
    * `\flushleft \flushright` : 表格左右对齐
  * 标题
    * `\caption{}` 赋予序列号和标题, 标题是括号里的内容, 序列号则是`Table`后面跟自动的编号 `Table 1`
      * 表的标题一般写在表内容的前面
    * `\label{}`   一般用来给表格加入一个 label, 方便其他地方引用该表格 `\ref{}`
      * 一般写在表格标题后面, 因为使用 `\ref` 的时候表格必须要有编号
      * 没有 `\caption` 的话 `\label` 没有任何作用
  * 双栏归一栏
    * 双栏情况下, 当表格宽度过长时, 会导致表格覆盖到另一栏的文字
    * 只需要将表格中 `table` 更改为 `table*` 即可
    * 

用法
```latex
\begin{table}[!hbt]
\centering
\caption{Obtained marks.}
\label{tab-marks}
    此处嵌入 tabular 表格环境, 写入表格内容
\end{table}

```

## 7.2. Figure 插图

* 往 Latex 文件中插图
* 根据图片的格式, 需要使用不同的编译器, 尽量确保所有插图都是统一的格式
  * latex : eps ps
  * pdflatex: pdf jpeg tiff png
* 通过使用较新的 graphicx 包可以方便的插入任何格式的图片

### 7.2.1. epsfig

eps 格式的图片可以使用专有的命令 epsfig, 该命令定义在 `epsfig` 的包中

命令 ` \epsfig{file=fname,[其他属性]} 用于插入一个图片`
* file=fname :用于指定插图的名称, 可以省略后缀名, 需要确保该图片是 eps 格式
* width=  height= : 用于指定插图的显示大小, 省略则是原大小, 只指定一个则是按比例缩放, 指定两个则会相应的拉伸
* angle= : 用于给定一个逆时针旋转的角度, 



### 7.2.2. figure

同表格一样, 也是图片专用的环境, 使用 `begin{figure}[!hbt]` 进入环境来达成:
1. 给予一个编号和标题
2. 设定 label, 来方便文中其他地方的引用
3. 设定图片在一页中的显示位置, 设定方法同上关于表格的说明一样

side-by-side图片
* 要想插入 side-by-side 的图片, 只需要连续写两个插入图片命令即可 (`\includegraphics`)
* 命令之间不能有空行
* 使用 `\hfill` 来将两张图的中间尽可能填入空白 (图片被放到最左端和最右端)
* 两张图会被赋予同一个编号, 若想分别编号, 需要使用 `minipage` 环境

```latex
\begin{figure}[!hbt]
\centering
\epsfig{file=girl.eps, width=2.0cm, height=2.0cm}
% 表的标题一般写在表内容的后面
\caption{A girl.}  
% label 只有标题存在的时候才能生效
\label{girl1}
\end{figure}

```

### 7.2.3. subfigure

有时候需要将图片分组, 每个组有一个大标题, 然后图片有自己的小标题, 此时可以使用 subfigure
* `subfigure[标题]{内容}`
* 定义在 subfigure 包中, 使用 `\usepackage[tight]{subfigure}` 来导入
* 同样的 `\hfill` 可以用来填充空白, `\\` 可以用来强制换行

```latex
\begin{figure}[!htb]
\centering
\subfigure[A girl.]
{ \includegraphics[width=2.0cm]{girl}
\label{girl}
}\hfill
\subfigure[A flower.]
{ \includegraphics[width=2.0cm]{flower}
\label{flower}
}

```
## 7.3. includegraphics graphicx

更加通用的插图命令, 可以插入任何格式的图片, 定义在 `graphicx` 包中
* graphics 是标准的包
* graphicx 是基于以上进行拓展的图片包, 二者在使用时的参数格式有区别
* 以下以 graphicx 为主进行学习


命令 `\includegraphics*[aopt]{fname}`
* 如果命令带有星号*, 则图片会被裁剪的插入, 如果没有星号, 则图片超出边界的部分会和文字重叠 
* fname: 图片的名称, 不带后缀名
* aopt : 图片的属性, graphicx 的命令格式是所有命令都写在一个方括号内, 用逗号隔开


参数说明
* Bounding Box 设置, 包括两种兼容性的命令格式
  * `bb=x y h w` 用空格隔开的用于设置边界的参数
  * bblx, bbly, bburx, bbury. 分别等用于上面的 bb中的依次参数
  * natwidth, natheight . 等同于设置成 `bb=0 0 natheight natwidth`
* pagebox, 因为 pdf 文件没有 BoundingBox, 所以通过别的方法设置边界, 可以选择下面的任意一项
  * me


# 8. 引用 Reference

文献引用有专门的环境, 以及多种不同的包对应引用, 最常用的有两种  
1. thebibliography   环境    
2. bibtex 引用数据库

## 8.1. cite 语句

* 不管用哪一种, 在文章中使用引用都是 `\cite{ckey}`
* `\cite{}` 的作用类似于 `\ref{}`
* `\cite[note]{}` 的额外可选参数, 用于对引用进行额外的补充说明, 一般是 `cite[pages 45-46]` 这样的页码说明
  * 在编译结果中显示为 `[25, pages 45–46]`

可以引用的包
1. overcite
   * 使用该包将会自动把所有引用作为上标显示, 不需要更改文档中的任何内容, 只需要 `\usepackage{overcite}`
   * 这个包不能和 `\cite[note]{}` 的可选参数 note 一起使用, 因为 note 不能放到上标处


## 8.2. thebibliographic  环境

* 文献引用有专门的环境 `thebibliography` , 放在 `\end{document}` 之前即可
  * 较为原始, 不能区分 book 和 article
  * 不管有没有最终被引用, 都会出现在文章末尾的 Reference
* 每一个引用实例需要包含两个必须 parts
  * a user-defined unique citation key
  * detail of the reference
  * 不方便的地方就是所有引用细节必须一个一个填进去


command
* `\begin{thebibliography}{}`
  * 一般整个引用可以写在单独的另一个文件中, 当作数据库, 并通过 `\input{}` 直接导入
  * 开启环境需要传入一个参数用于指定 每个索引的 identifiers 所占的空格
  * `\begin{thebibliography}{0000}` 即给定4columns 用于打印 identifiers
    * 一般传入最长的那个 identifier 的长度就可以
  * 开启环境会在文章中自动生成对应的标题
    * 在 `article` 中是 `References`
    * 在 `book report` 中是 `Bibliography`
* `\bibitem[ident]{ckey}` , 新建一个引用
  * ident 是可选参数, 代表引用时显示的标号, 不是传入的时候就是 数字包括在方括号中 `[1]`
  * key 就是具体给这个引用分配的索引名称
* `\newblock `
  * 定义一个新的区域
  * 用于在 `\bibitem` 后传入每一个引用的不同部分, 作者, 文章名 等等
  * 这个方法的缺点就体现在这里, 不能自动区分

```tex
\begin{thebibliography}{00}
  \bibitem{Beven-2000}
  \newblock Beven, K.
  \newblock {\em Rainfall-Runoff Modelling,The Primer.}  % 斜体
  \newblockJohn Wiley \& Sons, Chichester; 2000.
\end{thebibliography}
```

## 8.3. BibTex 

Latex 文档引用的参考文献管理库, 克服了 thebiblography 的所有缺点  
同样是传入 bib 数据库, 但是只会显示文章中有 `cite` 的引用文献  

一个 bibtex 引用包含了三个必须参数
1. type of the reference
   * eg. book, article, proceedings
2. user-defined citation key
3. detail
   * 根据type的不同, detail 里的必须参数也随之改变
   * 各个 fields 可以任意顺序输入, 并会根据打印的 style 被自动 arrange
* 插入一条引用就是 使用`@`+`type of the reference`+`{ckey, 其他参数}`
  * eg. `@article{myckey, auther= {}, year={} }`
  * 除了 key, 各个小项目都用大括号括起来

使用方法
1. 在latex工作目录下新建一个 `.bib` 的文本文件
2. 在谷歌学术等网站上直接复制 `BibTex` 的参考文献格式并粘贴进去
3. 在 latex 文档中引入 `cite` 包 `\usepackage{cite}`
4. 在文章中使用引用`cite{name}`
5. 在结尾处设置参考文件显示格式
  * `\bibliographystyle{plain}`    指定引用的显示格式
6. 在结尾处设置参考文献
  * `\bibliography{reffile}`   传入 `.bib` 文件的名称, 不需要后缀名
  * 使用 `\nocite{key}` 可以在文末的引用里加入没被在文章中具体引用的文献, 用于补充
  * 在文章最后会自动生成 reference


### 8.3.1. Bibliographic Styles 

引用的显示格式举例:  
* `plain`, 按字母的顺序排列，比较次序为作者、年度和标题；
* `unsrt`, 样式同plain，只是按照引用的先后排序；
* `alpha`, 用作者名首字母+年份后两位作标号，以字母顺序排序；
* `abbrv`, 类似plain，将月份全拼改为缩写，更显紧凑；
* `ieeetr`, 国际电气电子工程师协会期刊样式；
* `acm`, 美国计算机学会期刊样式；
* `siam`, 美国工业和应用数学学会期刊样式；
* `apalike`, 美国心理学学会期刊样式；

在需要添加引用的部分使用 `\cite{name}`


# 9. Define Macros

* command 和 environment 都属于 Macros 的一种, 都可以被用户自定义
* Latex 预定义的内置宏也可以被用户复写
* 宏的最初的目的就是为了减少重复使用的超长命令
* 用户对宏的操作应该在 preamble 或者额外的 `.cls` 文件中



## 9.1. newcommand renewcommand

定义一个新的 command `\newcommand{newc}[num]{aval}` `\providecommand{newc}{aval}`
* newc 是新定义的命令的名字
* aval 是命令的参数
* 如果 newc 是一个已存在的命令 
  * `\newcommand` 会报错
  * `\providecommand` 会保留原本的命令, 且不会有任何提示, 所以不应该被使用
* `\renewcommand` 用于修改一个命令的定义, 语法相同


### 9.1.1. 定义无参数命令

无参数命令通常被用来简化输入, 直接替代成另一块字符

| 命令                                           | 效果                             |
| ---------------------------------------------- | -------------------------------- |
| `\newcommand{\bs}{$\backslash$}`               | ‘\bs’ to print ‘\’               |
| `\newcommand{\xv}{\mbox{\boldmath$x$}}`        | ‘\xv’to print‘x’                 |
| `\newcommand{\veps}{\ensuremath{\varepsilon}}` | ‘\veps’to print‘ε’               |
| `\newcommand{\cg}{\it Center of Gravity\/}`    | ‘\cg’to print‘Center of Gravity’ |

命令详解:
* `\backslash`  打印`\`
* 可以用 newcommand 来快速输入一串字符
* 用户定义的命令字符末尾需要加上 `\`, 来保护在文本模式下紧挨着的空格生效

### 9.1.2. 定义必须参数命令

`\newcommand{}[]{}` as `\newcommand{newc}[n]{..{#1}..{#2}..{#n}..}`
* 方括号的 n 代表必须参数个数
* 访问每个参数通过加大括号的  `{#1}` 来访问

### 9.1.3. 定义可选参数命令

`\newcommand{}[][]{}` as `\newcommand{newc}[n][farg]{..{#1}..{#2}..{#n}..}` 
* frag 是必须参数的默认值, 添加了默认值后该参数及变成可选参数
* frag 会顺序赋值给 `#1 ,#2`

## 9.2. def

同 newcommand 相比, `\def` 是 Tex原生的低级命令
* 功能同 newcommand 等价
* 不会检查命令是否存在


删除一个宏:
* `\let\mymacro\undefined`
* `\let\mymacro\donothing`

## 9.3. 定义 environment

# 10. documentclass 介绍  