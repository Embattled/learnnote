# 统计学

统计学是通过搜索, 整理, 分析, 描述`数据`等手段, 以达到推断所测对象的本质, 甚至`预测对象未来`的一门综合性科学。统计学用到了大量的数学及其它学科的专业知识, 其应用范围几乎覆盖了社会科学和自然科学的各个领域


## Bayesian Information Criterions 贝叶斯信息准则 BIC

主观贝叶斯派归纳理论的重要组成部分:
* 在不完全情报下, 对部分未知的状态用主观概率估计
* 然后用贝叶斯公式对发生概率进行修正
* 最后再利用期望值和修正概率做出最优决策



## Stein's Unbiased Risk Estimate (SURE)

于 1981 年由  Charles Stein 提出, 发表在论文 `Estimation of the Mean of a Multivariate Normal Distribution`  中  

在实际应用中, 被用来选取最优的 小波降噪的参数

对于一个满足高斯分布(正态分布) 的随机变量 可以定义其观测(噪声信号)为 $x$ , 无噪信号 $\theta$
* 那么有 噪音 $n = x-\theta = N(0,\sigma^2)$
* 观测信号 $x = N(\theta,\sigma^2)$
* 高斯分布的密度函数可以写为 $\phi(x)=\frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{(x-\theta)^2}{2\sigma^2}}$

高斯公式的转换
* $\frac{\partial\phi(x)}{\partial x} = \phi(x)\frac{\partial(-\frac{(x-\theta)^2}{2\sigma^2})}{\partial x}=\phi(x)\frac{x-\theta}{\sigma^2}$
* 因此对于高斯密度公式, 满足:
* $\sigma^2\frac{\partial\phi(x)}{\partial x}=\phi(x)(\theta-x)$


对于任意一个连续随机变量x 的数学期望 $E(x)$ 可以写成如下
* $E(x)=\int_{-\infin}^{+\infin}x\phi(x) dx$  

对于任意一个可微的函数 $f(x)$ 其导数 $\frac{\partial f(x)}{\partial x}$ 也是关于 x 的一个函数, 那么有
* $E(\frac{\partial f(x)}{\partial x}) = \int_{-\infin}^{+\infin} \frac{\partial f(x)}{\partial x}\phi(x)dx$


联系上面高斯公式的转换公式, 有
* $\sigma^2 E(\frac{\partial f(x)}{\partial x}) = \sigma^2 \int_{-\infin}^{+\infin} \frac{\partial f(x)}{\partial x}\phi(x)dx$
* $= \sigma^2[f(x)\phi(x)]_{-\infin}^{+\infin} -\int_{-\infin}^{+\infin}\sigma^2f(x)\frac{\partial\phi(x)}{\partial x}dx$
* $=\int_{-\infin}^{+\infin}f(x)\phi(x)(x-\theta)dx$
* $=E(f(x)(x-\theta))$
* 这部分被称为 Stein 引理


对于向量的随机变量来说, 有观测值 $x\in \mathbb{R}^M$, 真值同样 $\theta\in \mathbb{R}^M$, 其中 $M$ 是维度数, 
* 对于一个从观测值 $x$ 推导真实值$\theta$ 的推到函数 $h(x)=\hat{\theta}$ 进行 MSE评价  
* $MSE = ||h(x)-\theta||^2_2=||\hat{\theta}-\theta||^2_2=\sum_i^M(\hat{\theta_i}-\theta_i)^2$


以此作为基础, 对于无噪信号 $\theta$ (不可知), 观测 $x$ (有噪信号), 希望得到一个最优的推算函数 $h(x)=\hat{\theta}$ 使得 MSE 风险最小
* 因为是随机变量, 所以实际上的目标是求 $E_x(MSE)=E_x(||\hat{\theta}-\theta||^2_2)$
* 将观测变量带入 MSE 公式
* 有 $MSE = ||(\hat{\theta}-x)+(x-\theta)||^2_2$ , 展开
* 有$E_x(MSE) =  E_x||\hat{\theta}-x||^2_2+E_x||x-\theta||^2_2+2E_x<(\hat{\theta}-x),(x-\theta)>$  
* 首先有 1. 可以通过直接计算得到 
* $E_x||\hat{\theta}-x||^2_2=E_x||h(x)-x||$  
* 其次 2. $E_x||x-\theta||^2_2$ 可以通过对数据集全局进行分析得到估算量, 例如和图像传感器相关的整体噪声方差期望 $\sigma^2$, 对于M维向量, 有$\sigma^2=M\sigma^2_e$
* 要利用 Stein 引理 进行代入的就是最后的部分 
  * $2E_x<(\hat{\theta}-x)(x-\theta)>$
  * $=2\sum_i^ME_x((\hat{\theta_i}-x_i)(x_i-\theta_i))$ 
  * $=2\sum_i^ME_x(f(x_i)(x_i-\theta_i)), f(x_i) = \hat{\theta_i}-x_i$
  * $=2\sum_i^M\sigma_e^2E_x(\frac{\partial f(x_i)}{\partial x_i})$
  * $=2\sum_i^M\sigma_e^2E_x(\frac{\partial h(x_i)}{\partial x_i}-1)$
* 在同一个应用场景下, 式子 2: $E_x||x-\theta||^2_2$ 和式子3 代入的 $2\sigma_e^2E_x(\frac{\partial f(x)}{\partial x})$ 都是指的是同一个随机变量 x 的方差  $\sigma$ 因此可以对结果进行最后的整理
* $E_x(MSE)=E_x||\hat{\theta}-x||^2_2+M\sigma^2_e+2\sum_i^M\sigma_e^2E_x(\frac{\partial h(x_i)}{\partial x_i}-1)$ 

$$=E_x(SURE(\hat{\theta},x))=E_x||\hat{\theta}-x||^2_2+2\sigma^2\sum_i^ME_x(\frac{\partial h(x_i)}{\partial x_i}) - M\sigma^2_e$$



在实际计算中, 有
* $\sum_i^M\frac{\partial h(x_i)}{\partial x_i} = div_x\hat{\theta}$
* 因此有完整的SURE 简易写法
$$ E_x(SURE(\hat{\theta},x))=E_x||\hat{\theta}-x||^2_2+2\sigma^2div_x\hat{\theta} - M\sigma^2_e$$
$$E_x(MSE)=E_x(SURE(\hat{\theta},x))=E_x||\hat{\theta}-x||^2_2$$

总结:
* SURE 是一种无偏估计方法
* 是一种 `几乎任意的非线性有偏估计量` 的均方误差(MSE)的无偏估计量
* 提供了给定 Estimator 准确性的指示
* 这很重要, 因为 Estimator 的真实 MSE 是待估计的未知参数的函数, 因此无法准确观测 

简而言之: 对于未知随机变量, 因为无法得知其真实值, 所以无法计算 均值, 方差等  
那么, 通过带入 SURE 公式, 即可在完全不知道随机变量均值的情况下, 直接评估对于一个真实值推测器的 MSE 性能 ,实现对 estimator 的最优化  


补充  
* 前面定义了真实值进行推导的函数为 $h(x)=\hat{\theta}$
* 相应的, 可以定义对观测值进行修正的函数 $g(x)+x=\hat{\theta}=h(x)$
* 那么则有 SURE 的另一种写法

$$E_x(SURE(\hat{\theta},x))=E_x||h(x)-x||^2_2+2\sigma^2\sum_i^ME_x(\frac{\partial h(x_i)}{\partial x_i}) - M\sigma^2_e$$
$$E_x(SURE(\hat{\theta},x))=E_x||g(x)||^2_2+2\sigma^2\sum_i^ME_x(\frac{\partial g(x_i)}{\partial x_i}) + M\sigma^2_e$$

