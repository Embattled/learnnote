# 1. 数学基础

字母加上小斜杠的字符一般被用来表示数域  

* $\mathbb{Z}$ 整数域
* $\mathbb{Q}$ 有理数
* $\mathbb{R}$ 实数域
* $\mathbb{C}$ 复数域


## 1.1. 复数域

复数域补足了实数域的不完备的地方
* 负数可以开根
* 实数域需要引入特殊的无穷 $\infty$ 数字, 从数学上来说不完备, 复数域上无穷则是一个普通的数

## 1.2. 复变函数  

复变函数的定义:
* $z=x+iy$ 称 x 是 real, y 是 imaginary 虚部
* $f(z)=u+iv=u(x,y)+iv(x,y)\in \mathbb{C}$
  * $x,y,u,v \in \mathbb{R}$

复变函数的导数:
* 标准导数定义:
$$f'(z_o)=\lim_{z\rarr z_0}\frac{f(z)-f(z_0)}{z-z_0}$$
* 若复变函数 $f(z)=u+iv$ 可导, 则 u,v 存在牵连关系
  * 至少沿着 x,y (实轴和虚轴)的方向趋近得到的表达式相同
  * 沿着实轴: 
    * $z-z_0=\triangle x$
    * $f'(z_o)=\lim_{\triangle x \rarr 0}\frac{\triangle u + i\triangle v}{\triangle x}=\frac{\partial u}{\partial x}+i\frac{\partial v}{\partial x}$
  * 沿着虚轴, 同理:
    * $z-z_0=i\triangle y$
    * $f'(z_0)=\lim_{\triangle x \rarr 0}\frac{\triangle u + i\triangle v}{i\triangle y}=-i\frac{\partial u}{\partial y}+\frac{\partial v}{\partial y}$
  * 由此, 沿着 x 或 y 得到的极限应该相同, 即分别的实部和虚部都应该分别相同, 可得到
    * 实部 $\frac{\partial u}{\partial x}=\frac{\partial v}{\partial y}$
    * 虚部 $\frac{\partial v}{\partial x}=-\frac{\partial u}{\partial y}$
    * 事实上, 所有初等函数都满足该条件
* 复变函数保留全套求导公式

## 1.3. 柯西黎曼关系

对于一个在复数域中的复变函数 $f(z)=u(x,y)+iv(x,y)$  
* 若 $f(z)$ 在区域 $\sigma$ 里满足柯西黎曼关系, 则由 $\sigma$ 的边界函数 $C$ 可以推导出 区域 $\sigma$
内任意一点的函数值
* 此时称 $f(z)$ 在区域 $\sigma$ 内是解析函数  


## Total Variation 总变差

数学中, 总变差 指的是一个函数其数值变化的差的总和  
* 对于实值函数 f 在区间 a,b 上的总变差 即一维参数曲线 f(x) 在 x in (a,b) 上的弧长
* 求连续可微函数的总变差可以通过积分求出 $V_a^b(f)=\int_a^b|f'(x)|dx$
* 求离散函数的总变差即求各个相邻元素差的和 $_a^b(f)=\sum_{i=0}^{n_P-1}|f(x_{i+1})-f(x_i)|$



# 2. 微积分


## 2.1. 格林公式 Green formula 

格林公式描述了: 平面上沿着闭曲线 L 对坐标的曲线积分与 该闭曲线 L 围成的闭区域 D 上的二重积分直接的关系  

* 当xOy平面上的曲线起点与终点重合时, 则称曲线为闭曲线
* 正方向: 规定一个人沿着闭曲线 L 环形时, 若区域 D 总是位于该人的左侧, 则称前进方向为正方向, 反之则为负方向

$$\iint_D(\frac{\partial Q}{\partial x}-\frac{\partial P}{\partial y})dxdy=\oint_L Pdx+Qdx$$

# 3. 级数



在信号领域, 信号分解的方法多种多样, 可将信号分解为直流分量+交流分量, 偶分量+奇分量, 实部分量+虚部分量, 脉冲分量, 正交分量等多种形式  




# 4. 概率论



## 4.1. 分布

### 4.1.1. 二项分布 Binomial Distribution



### 4.1.2. 泊松分布 Poisson Distribution

是一种统计与概率学里常见到的离散机率分布 (discrete probability distribution)

$$P(X=k)=\frac{\lambda^k}{k!}, k=0,1,...$$

泊松分布的特征
* 方差和数学期望均为 $\lambda$
* $\lambda$ 是单位内随机事件的平局发生次数
* 泊松分布适合描述单位时间内随机事件的发生次数


### 4.1.3. Variance-Stabilizing Transformation VST

方差稳定化转换 (VST) 是一类统计学上的 数据变形 操作, 用于将 参数分布 的数据方差与数据均值关联的特性减轻, 来方便建模或者其他操作  
* 代表性的分布有 泊松分布 和 二项分布  

即使是很简单的 $y=\sqrt{(x)}$ 对于降低泊松分布的方差期望相关性都有很好的效果  

该类转换的代表性研究是 Anscombe transform  

一般来说, 如果某个分布的 方差和期望的关系已经被函数化建模  $var(X)=h(\mu)$, 那么 一个具有可接受偏差的 VST 可以表示为, 根据具体实际情况还会加上标量偏移以及 缩放因子    
$$y\propto \int^x \frac{1}{\sqrt{h(\mu)}}d\mu$$


例1 : $h(\mu)=s^2\mu^2$, 即标准差和期望成比例, 则有 VST  
$$y=\int^x\frac{1}{\sqrt{s^2\mu^2}}d\mu=\frac{1}{s}\ln(x) \propto \log(x)$$


例2 : $h(\mu)=\sigma^2+s^2\mu^2$, 即在期望较小的时候, 有一个基础方差 sigma, 此时有 VST  
$$y=\int^x\frac{1}{\sqrt{\sigma^2+s^2\mu^2}}d\mu=\frac{1}{s}asinh\frac{x}{\sigma/s} \propto asinh\frac{x}{\lambda}$$

### 4.1.4. Delta Method 

the approximate probability distribution for a function of an asymptotically normal statistical estimator from knowledge of the limiting variance of that estimator. 


# 5. Hidden Markov Model HMM

隐马尔可夫模型, 是一个在当前世界应用极其广泛的数学模型  

一个 HMM 模型有三个要素, 可以表示成 $\lambda = (A,B,\Pi)$
* A : 状态转移概率矩阵, 假设共有 N 种不可见的状态, 状态间的互相转移的概率为 $A=[a_{ij}]_{N\times N}$
* B : 观测概率矩阵, N 种不可见的状态共计可以有 M 种观测, 则处在 j 状态下得到观测 k 的概率为 $B=[b_j(k)]_{N\times M}$, 又称发射概率
* $\Pi$ : 初始状态概率向量, 即初始时间时每个状态的概率

一个 HMM 的完整五元组, 还要再加上
* Q : 所有可能的状态集合
* V : 所有可能的观测集合

HMM 是一个关于 `时序` 的 `概率模型`, 一个不可观测的状态随机序列 S (state sequence) 是一个隐藏的马尔科夫链, 每个状态随机生成一个观测, 由此产生的观测的随机序列 O (observation sequence).

* 观测序列 $O=O_1,O_2,...,O_T$
* 状态序列 $S=S_1,S_2,...,S_T$

三个基本问题:
1. 观测问题 : 给定模型 $\lambda = (A,B,\Pi)$ , 求一个观测序列 O 出现的概率
2. 学习问题 : 给定观测序列 O, 估计模型的参数, 使得该观测序列 O 的概率最大
3. 预测问题(解码问题) (词性标注问题): 给定模型和观测序列O, 求观测序列 O 所对应的概率最大的状态序列 I 
* 目前 概率计算问题和解码问题都有最优解, 而学习问题则是最复杂的问题, 需要解决预测子问题 

## 5.1. 前向后向算法 解决 观测问题

因为序列和模型概率都全部已知, 理论上可以通过暴力求解对应概率, 而实际上由于隐状态的个数一般非常大, 计算全部路径的计算量是 N^T , 不可能应用在实际中


流行解法: 前向后向算法 (基于动态规划)

前向算法:
* 定义二维 DP 表ti , 记录到沿着给定观测 O 的 t 时刻时状态为 i 的概率
* 初值: `DP[1][i]` = $\pi_i b_i(o_1)$  
* `DP[t+1][i]` = $(\sum^N_{j=1}DP_{[t][j]}a_{ji}) b_i(o_{t+1})$
* 求结果 $P(O|\lambda)=\sum_{i=1}^NDP_{[T][i]}$
* 复杂度 O(TN^2)

后向算法: 仅仅只是更改了计算方向
* 初值: `DP[T][i]=1`
* `DP[t][i]`=$\sum^N_{j=1}DP_{[t+1][j]}a_{ij}b_j(o_{t+1})$
* 求结果  $P(O|\lambda)=\sum_{i=1}^NDP_{[T][i]}\Pi_ibi(o_1)$
* 感觉从计算上来说多了一个乘法, 发射概率需要在 sum 里的每个单位都乘进去

## 5.2. 维比特算法 

维特比算法之所以重要, 是因为凡是使用隐含马尔可夫模型描述的问题都可以用它来解码: 
* 即, 该算法是应用于解码问题的一个算法
* 词性标注问题的定义:
  * 有一句已经分词好的句子 n 个单词, 有定义好的词性字典 m 种词性
  * 求每一个单词的词性
  * 共有n^m 种可能
* 算法构成:
  * 输入 : 模型 $\lambda=(A,B,\Pi)$ 观测$O=(o_1, o_2,...,o_T)$
  * 输出 : 最优(解码/词性/状态)序列 $I=(i_1,i_2,...,i_T)$

算法流程
* 初值: 
  * `DP[1][i]` = $\pi_i b_i(o_1)$  同前向算法一样 , t=1时 各个状态 满足 o1 的概率
  * `DP2[1][i]=0`  置全0
* 递推:
  * `DP[t][i]` = $max_{j\in[1,N]}(DP[t-1][j]a_{ji}) b_i(o_t)$, 解释: t-1时刻的dp里转移到状态 i 的最大概率, 然后发射出 ot 的概率
  * `DP2[t][i]` = $argmax_{j\in[1,N]}(DP[t-1][j]a_{ji})$ , 解释: 仅仅只是记录对应转移到 $i_t$ 的最大概率的 $i_{t-1}$
* 中止:
  * $i_T=argmax_{j\in[1,N]}(DP[T][j])$ 最终时刻里概率最大的状态即为最优解里的 $i_T$
* 回溯:
  * $i_{t-1}=DP2[t][i_t]$, 一个 O(n) 的查表而已, 得到最终的完整 $I$ 序列