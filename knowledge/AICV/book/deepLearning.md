# 1. Deep Learning 

`Ian Goodfellow` is a Research Scientist at Google.
`Yoshua Bengio` is Professor of Computer Science at the Université de Montréal.
`Aaron Courville` is Assistant Professor of Computer Science at the Université de Montréal.

2016 年的书

An introduction to a broad range of topics in deep learning, covering mathematical and conceptual background, deep learning techniques used in industry, and research perspectives.


# 2. 深度模型中的优化

## 2.1. 基本算法 - 基本优化算法


### 2.1.1. 基本算法 SGD

* SGD stochastic gradient descent - 随机梯度下降
  * 相比于正规的梯度下降需要计算整个训练集上的平均损失, 然后计算梯度
  * 最早的 SGD 是对每一个样本单独计算梯度并立即更新
  * 后来提出了 minibatch 的概念
  * 现在通用的基本上指代 应用 minibatch 的SGD, 表示仅使用 minibatch 的训练样本即可更新一次参数
* 在实际使用中
  * 我们一般会设置线性衰减学习率 $\epsilon_k = (1-\alpha)\epsilon_0+\alpha\epsilon_\tau$
  * 即在 $\tau$ 次迭代后, 学习率保持常数不再下降

算法过程
* 参数: 学习率 $\epsilon_k$ , 初始化的网络参数 $\theta$
* while 循环条件 do
* ---- 采集 minibatch 数据 $\{x^{(1)},...,x^{(m)}\}$
* ---- 计算梯度 $\hat{g} \leftarrow +\frac{1}{m}\nabla_\theta \sum_iL(f(x^{(i)};\theta),y^{(i)})$
* ---- 更新参数 $\theta \leftarrow \theta - \epsilon \hat{g}$
* end while

### 2.1.2. 动量 momentum

动量的概念是为了加速学习
* 动量会积累 : 之前迭代梯度的 指数级衰减的 移动平均
* 动量的表示 : $\upsilon$ 代表速度, 表示参数空间的 梯度方向和速率
* 动量在大部分情况都设置成 0.9

带梯度的 SGD算法过程
* 参数: 学习率 $\epsilon_k$, 动量参数 $\alpha$, 初始化的网络参数 $\theta$, 初始速度 $v$
* while 循环条件 do
* ---- 采集 minibatch 数据 $\{x^{(1)},...,x^{(m)}\}$
* ---- 计算梯度 $g \leftarrow +\frac{1}{m}\nabla_\theta \sum_iL(f(x^{(i)};\theta),y^{(i)})$
* ---- 计算速度 $\upsilon \leftarrow \alpha\upsilon-\epsilon g$
* ---- 更新参数 $\theta \leftarrow \theta + \upsilon$
* end while
* 拆开来看 带动量就是更新参数的时候既用梯度又用速度 
  * $\theta \leftarrow \theta -\epsilon g+ \alpha\upsilon$
  * $\upsilon \leftarrow \alpha\upsilon-\epsilon g$



Nesterov accelerated gradient(NAG): 带有预知的动量
* Nesterov 动量对于RNN的学习有很大的帮助, 和普通动量一样常常和其他优化算法进行结合
* 参数: 学习率 $\epsilon_k$, 动量参数 $\alpha$, 初始化的网络参数 $\theta$, 初始速度 $v$
* while 循环条件 do
* ---- 采集 minibatch 数据 $\{x^{(1)},...,x^{(m)}\}$
* ---- 根据动量计算临时更新后的参数 $\tilde{\theta}\leftarrow \theta + \alpha v$
* ---- 在临时点计算梯度 $g\leftarrow +\frac{1}{m}\nabla_{\tilde{\theta}}\sum_iL(f(x^{(i)};\tilde{\theta}),y^{(i)})$
* ---- 计算速度 $\upsilon \leftarrow \alpha\upsilon-\epsilon g$
* ---- 正式更新参数 $\theta \leftarrow \theta + \upsilon$



## 2.2. 自适应学习率算法



基于 minibatch 的自适应学习率算法是目前应用最广的算法

* 自适应算法的核心是不再拥有全局统一的学习率, 而是根据每一个单独参数的梯度历史来动态调整对应参数自己的学习率
* AdaGrad 几乎不需要手动调整学习速率, 因为真正的学习速率都是每次迭代重新计算的, 用默认的 0.01即可, 缺点就是容易学习率降低太快
* Adadelta 和 RMSProp 是独立被研究的, 都是为了解决 AdaGrad学习率降速的问题
  * RMSProp 推荐 学习率 0.001, 梯度累计衰减率 0.9
  * Adadelta 在 RMSProp 的基础上再加入了参数更新量的累计, 直接省略了初始学习率的超参数

### 2.2.1. AdaGrad

AdaGrad : (2011)有较好的理论性质, Ian 的书中表示经验上 AdaGrad 实际上会导致 有效学习率过早和过量的减少(一直在累加梯度)
* 适合应用在稀疏数据上
* 对非频繁的参数应用大更新, 对频繁的参数应用小更新, 具体表现在对于参数 $\theta$ , 不再使用统一的学习率 $\epsilon$ , 而是对于每一个单独参数在每一次参数迭代应用单独的学习率
* 参数: 全局学习率 $\epsilon_k$, 初始化的网络参数 $\theta$
* 用于除数非零的一个常数 $\delta =10^{-7}$
* 梯度累计变量 $r=0$
* while 循环条件 do
* ---- 采集 minibatch 数据 $\{x^{(1)},...,x^{(m)}\}$
* ---- 计算梯度 $g \leftarrow +\frac{1}{m}\nabla_\theta \sum_iL(f(x^{(i)};\theta),y^{(i)})$
* ---- 累计梯度的平方 $r\leftarrow r+g\odot g$
* ---- 计算更新 $\Delta\theta\leftarrow -\frac{\epsilon}{\delta+\sqrt{r}}\odot g$
* ---- 应用更新 $\theta \leftarrow \theta+\Delta\theta$

### 2.2.2. RMSProp 和 Adadelta

RMSProp : (2012)基于 AdaGrad 算法, 相比于一直累计梯度, RMSProp 指数级丢弃遥远梯度, 有效最常见算法之一
* 参数: 全局学习率 $\epsilon_k$, 初始化的网络参数 $\theta$
* 用于除数非零的一个常数 $\delta =10^{-6}$
* 梯度累计变量 $r=0$ , 梯度累计的舍弃速度 $\rho =0.9$
* while 循环条件 do
* ---- 采集 minibatch 数据 $\{x^{(1)},...,x^{(m)}\}$
* ---- 计算梯度 $g \leftarrow +\frac{1}{m}\nabla_\theta \sum_iL(f(x^{(i)};\theta),y^{(i)})$
* ---- 累计梯度的平方 $r\leftarrow \rho r+(1-\rho)g\odot g$
* ---- 计算更新 $\Delta\theta\leftarrow -\frac{\epsilon}{\delta+\sqrt{r}}\odot g$
* ---- 应用更新 $\theta \leftarrow \theta+\Delta\theta$

可以看出相比于 AdaGrad, RMSProp 只进行了很小的改变, 就是对梯度的累计进行了指数衰减

Adadelta : (2012) 同样是为了解决 AdaGrad 莽撞的降低学习速率的设定, 在累积了历史梯度的同时, 还附加累计了参数的历史更新量
* 参数: 初始化的网络参数 $\theta$, 完全不用设置初始学习速率
* 用于除数非零的一个常数 $\delta =10^{-6}$
* 梯度累计变量 $r=0$, 参数更新量的累计变量 $t=0$ , 两个累计的舍弃速度 $\rho =0.9$
* while 循环条件 do
* ---- 采集 minibatch 数据 $\{x^{(1)},...,x^{(m)}\}$
* ---- 计算梯度 $g \leftarrow +\frac{1}{m}\nabla_\theta \sum_iL(f(x^{(i)};\theta),y^{(i)})$
* ---- 累计梯度的平方 $r\leftarrow \rho r+(1-\rho)g\odot g$
* ---- 用新的梯度累计和旧的参数更新累计来计算参数更新 $\Delta\theta\leftarrow -\frac{\delta+\sqrt{t}}{\delta+\sqrt{r}}\odot g$
* ---- 累计更新量的平方 $t=\rho t +(1-\rho)\Delta\theta^2$
* ---- 应用更新 $\theta \leftarrow \theta+\Delta\theta$

### 2.2.3. Adam

Adaptive Moment Estimation(Adam) : (2014)引入梯度的一二阶矩, 对超参数的设置非常鲁棒  
Adam 用另一种方式实现了动量, 即将动量表示成梯度的一阶矩累计(相当于学习率=1 的动量)  

* 参数: 全局学习率 $\epsilon_k=0.001$, 初始化的网络参数 $\theta$
* 用于除数非零的一个常数 $\delta =10^{-8}$
* 两个梯度累计变量 $r=0 s=0$
* 两个梯度累计的舍弃速度 $\rho_1 =0.9, \rho_2=0.999$
* 时间戳 $t$
* while 循环条件 do
* ---- 采集 minibatch 数据 $\{x^{(1)},...,x^{(m)}\}$
* ---- 计算梯度 $g \leftarrow +\frac{1}{m}\nabla_\theta \sum_iL(f(x^{(i)};\theta),y^{(i)})$
* ---- 时间戳 $t\leftarrow t+1$
* ---- 累计梯度的一阶 $s\leftarrow \rho_1 s+(1-\rho_1)g$
* ---- 修正一阶累计值 $\hat{s}\leftarrow\frac{s}{1-\rho_1^t}$
* ---- 累计梯度的二阶 $r\leftarrow \rho_2 r+(1-\rho_2)g\odot g$
* ---- 修正二阶累计值 $\hat{r}\leftarrow\frac{r}{1-\rho_2^t}$
* ---- 根据修正值$\hat{s}\hat{r}$计算更新 $\Delta\theta\leftarrow -\frac{\epsilon}{\delta+\sqrt{\hat{r}}}\odot \hat{s}$
* ---- 应用更新 $\theta \leftarrow \theta+\Delta\theta$ 


# 卷积网络  

# 序列建模: 循环和递归网络
