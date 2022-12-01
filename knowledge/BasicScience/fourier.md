# 1. 基于傅里叶变化的信号分析  



## 1.1. 正交变换 正交分解

正交变换: 是信号变化的一系列统称, 广泛用于 简易处理, 特征抽取, 数据压缩

在线性代数中, 向量的正交可以表示为 $\vec{a}\cdot\vec{b}=0$

对于两个正交的向量, 可以离散的表示成:
$$\sum^n_{i=0}a_i*b_i=0$$

在信号处理中, 信号其实是连续的, 需要用连续的函数来表示, 对于函数的正交性形式, 则需要引入定义:
- 如果在区间 $(t_1,t_2)$ 上, 函数 $f_1(t)$和$f_2(t)$互不含有对方的分量, 则称 $f_1(t)$与$f_2(t)$在 $(t_1,t_2)$ 上正交
$$<f_1,f_2>=\int_{t_1}^{t_2}f_1(t)f_2(t)dt=0$$

正交分解的定理即为: 对于任意函数 $f(t)$, 在 $(t_1,t_2)$上可以表示成 正交函数集内函数的线性组合, 即

$$f(t) \approx \sum^N_{n=1}c_ng_n(t)$$



用于正交变换的方法也多种多样: 
* 傅里叶变换 Fourier Transform
* 正余弦变换 Sin Cosine Transform
* 沃尔希-哈德玛变换 Walsh-Hadamard Transform
* 斜变换 Slant Transform
* 哈尔变换 Haar Transform
* 离散小波变换 Discrete Wavelet Transform
* 离散K-L变换 Discrete Karhunen-Leave Transform
* 奇异值分解SVD变换 Singular-Value Decomposition
* Z变换
* 多尺度几何分析（超小波）
  
在数字处理中, 由于值是离散的, 因此各个变换的离散版本(Discrete) 也是很常用的  


## 1.2. Fourier Transform  傅里叶变换

傅里叶分析法是将一个函数表示成 周期分量之和的方法 

### 1.2.1. 傅里叶级数
傅里叶级数: 任何周期函数都可以用正弦函数和余弦函数构成的无穷级数来表示
* 前提是 : 周期函数
* 将连续信号分解为多个不同频率下的互相正交的信号
* 把信号由时域转移到频域
$$f(t)=\frac{a_0}{2}+\sum^{\infty}_{n=1}[a_n\cos(nw_0t)+b_n\sin(nw_0t)]$$

当信号的周期为无限大 $T\rarr\infty$ 即可引出 傅里叶变换, 可以应用在非周期的函数上, 即任意信号


傅里叶变换的局限性:
* 全时行  : $f(t)\rarr g(w), g(w)=\int^{+\infty}_{-\infty}f(t)e^{-wt}dt$ 
  * 因为需要在无穷上进行积分, 在不存在极限的情况下会有冗杂  
  * 需要对过去进行变换, 在某些情况下如果无法收集或者不存在过去信号的情况下

### 1.2.2. Discrete Fourier Transforms DFT 离散傅里叶变换

以一维离散数列来定义傅里叶变换:
* 对于长度为 M 的数列 `x[n]` 进行离散傅里叶变换 (无论正反)
* 可以得到的长度为 N 的频域数列 `X[k]`
* 这里的 N 被称为变换区间长度, 必须有 $N\ge M$ 
* 通常情况下只考虑 M=N, 因此很多时候 M N 的区别被模糊了, 没有原数列长度 M 的定义
* 倘若 N>M 的时候, 在计算中自动在 `x[n]` 后补零  


#### DFT 公式
正变换用 $DFT[x[n]]_N$ 来表示
$$X[k]=\sum^{N-1}_{n=0}e^{-2\pi j\frac{kn}{N}}x[n]$$


反变换可以写成 $IDFT[X[k]]_N$
$$x[n]=\frac{1}{N}\sum^{N-1}_{k=0}e^{2\pi j\frac{kn}{N}}X[k]$$


完整的DFT正反变换公式:
$$X[k]=\sum^{N-1}_{n=0}e^{-2\pi j\frac{kn}{N}}x[n]\rightleftharpoons x[n]=\frac{1}{N}\sum^{N-1}_{k=0}e^{2\pi j\frac{kn}{N}}X[k]$$

为了完善对称性, 进行了系数修正的版本如下  
$$X[k]=\frac{1}{\sqrt N}\sum^{N-1}_{n=0}e^{-2\pi j\frac{kn}{N}}x[n]\rightleftharpoons x[n]=\frac{1}{\sqrt N}\sum^{N-1}_{k=0}e^{2\pi j\frac{kn}{N}}X[k]$$

#### DFT 的应用

The values in the result follow so-called “standard” order
* DFT 的默认输出被称为标准排列, 在 `A=FFT(a,n)`的情况下
* `A[0]` 包含了 0频率的分量, 也是 `sum of the signal`

### 1.2.3. 二维离散傅里叶变换




### 1.2.4. 正余弦变换  Sine Cosine Transform

是傅里叶变换的一个变种, 类似于离散傅里叶变换, 但是去除了虚数部分  
* 相当于一个长度是它的两倍的对实偶函数进行的离散傅里叶变换  
* DCT 有8种标准类型
* 有两种相关变换  离散正弦变换, 改进的离散余弦变换
* 用途: 对数字信号 (信号, 图像) 进行有损压缩, e.g. JPEG

形式化定义: 线性的可逆函数 $F: R^n \rarr R^n$, 把 n 个实数变换到另外 n 个实数的操作  


本质上, DCT 和 DST 都来源于离散傅里叶变换, 属于特殊形式的DFT    
离散正余弦变换都各有 8 种形式, 被广泛用于图像视频压缩:
* 通过牺牲部分高频信息来压缩视频  
* DCT-II 被用的最为广泛, 在 H.264算法中被大量使用
* 近年来新的算法也逐渐开始使用其他形式的DCT, 例如 DCT-VIII 被加入到 H.266 的候选变换类型中


#### 1.2.4.1. DCT-II 

$$f_m=\sum^{n-1}_{k=0}x_kcos[\frac{\pi}{n}m(k+\frac{1}{2})]$$

最常用的一种 DCT, 通常被直接称作 DCT    

#### 1.2.4.2. DCT-III  

DCT-II 的逆变换, 通常称为 逆离散余弦变换 



### 1.2.5. Gabor transform / windowed Forier transform

为了解决傅里叶变换需要参考无限的问题, 限制傅里叶变换的参考范围
* 引入了 时窗 的概念 : 在某一段区域内, 希望 $f'(t)$ 能够无限趋近于 $f(t)$, 其他的区域则不关心 (置0)
* 定义时窗函数为 $g(t-u)$, 是一个衰减函数, 在 $t=u$ 的时候取到极值, 时窗函数的定义并不固定, 只需要 $g(0)$ 不为0, 且在其他位置快速衰减即可  
* 在具体的 Gabor transform 中, 是定义为了二元函数 $G(w,u)$, 完整的变换需要遍历所有的 u 的取值  
* 缺点: 由于单次变换中函数g是人为选定的是固定的, 导致分辨率单一
  * 根据 时窗函数 g 的选择, 会有不同的分辨率, 即每次考察的时间窗口长度 
  * 假如 g 的时窗非常小, 会导致频率上的分辨率降低
  * 加入 g 的时窗非常大, 会导致频率上的分辨率过高, 极限情况下就等同于 傅里叶变换

$$G(w,u)=\int^{+\infty}_{-\infty}f(t)e^{-wt}g(t-u)dt$$


### 1.2.6. Wavelet Transform 小波变换 (W.T.)

针对 Gabor transform 的单一分辨率的缺点, 希望能够通过参数来控制时窗的长度以及频域上的分辨率  
* 将时窗函数理解成投影 $\int f(x)g(x)dx$ 可以理解成 f 在 g 上的投影
* 我们希望得到一个足够好的投影函数 $\Psi(a,b,t)$ 或者写成 $\Psi_{a,b}(t)$:  
  * a,b 是类似于 Gabor 变换, 用于控制窗口长度以及中心点的参数
  * 一个通用的小波变换投影函数可以写成 $\Psi_{a,b}(t)=\frac{1}{\sqrt a}\psi\frac{t-b}{a}$
  * 定义 a 为 放缩尺度因子 $a>0$
  * 定义 b 为 平移因子    $-\infty <b<\infty$

定义小波变换为, 难度主要为寻找一个合理的投影函数, 使得 $\triangle t \triangle w$ 自由可控:
$$W_{a,b}f(t)=\int^{+\infty}_{-\infty}f(t)\psi(a,b,t)dt$$

定义小波逆变换为:
$$f(t)=\frac{1}{c}\int_0^\infin\int_\infin^\infin a^{-2}W_f(a,b)\psi_{a,b}(t)dbda$$


墨西哥草帽式:
$$\psi(t)=\frac{2}{\sqrt{3}}\pi^{-\frac{1}{4}}(1-t^2)e^{-t^2/2}$$

Morlet小波:
$$\psi(t)=e^{-t^2/2}e^{iw_0t}$$

还有很多无法写出来数学式的基函数  