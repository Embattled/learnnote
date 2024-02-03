# 1. 概率论 Probability Theory 

是研究随机现象数量规律的数学分支




### 1.0.1. 零均值随机变量的幂的数学期望

一般的, 对于一个随机变量的方差有 $D(X)=E(X^2)-[E(X)^2]$

因此若知道了一个随机变量的平方的期望对于了解该随机变量的方差十分有用, 但是这并不是一件容易的事情, 这里只讨论零均值且y轴对称的随机分布, 因为
* 均值为0可以避免很多低阶项
* y轴对称则奇数幂的数学期望为0  


首先, 任意方差的高斯分布都可以转换成标准正态分布  
$$X\sim \Nu(0,\sigma^2)\Rightarrow Y=\frac{X}{\sigma}\sim \Nu(0,1) $$

因此对于任意的高斯分布X的幂期望, 都有
$$E(X^n)=E((\sigma Y)^n)=\sigma^nE(Y^n)$$

推导需要使用 Gamma 函数
$$\Gamma(x)=\int_0^{+\infin}t^{x-1}e^{-t}dt$$

进行推导需要展开正态分布的公式
* $Y=\frac{1}{\sqrt{2\pi}}e^{-\frac{y^2}{2}}$
* $E(Y^n)=\int (Y)^n d(y^n)$
* $=\int y^n\frac{1}{\sqrt{2\pi}}e^{-\frac{y^2}{2}}dy$
* $=\frac{1}{\sqrt{2\pi}} \int 2^{(n-1)/2}(\frac{y^2}{2})^{(n-1)/2}e^{-\frac{y^2}{2}}d\frac{y^2}{2}$
* $=\frac{2^{n/2-1}}{\sqrt{\pi}}\int t^{(n-1)/2}e^{-t}dt$
* $=\frac{2^{n/2}}{\sqrt{\pi}}\int_0^{+\infin} t^{(n-1)/2}e^{-t}dt$
* $=\frac{2^{n/2}}{\sqrt{\pi}}\int_0^{+\infin} t^{(n+1)/2-1}e^{-t}dt$
* $=\frac{2^{n/2}}{\sqrt{\pi}}\Gamma(\frac{n+1}{2})$

根据Gamma函数的性质
$$\Gamma(x+1)=x\Gamma(x), \Gamma(\frac{1}{2})=\sqrt{\pi}$$

最终有 
$$E(Y^n)=\frac{2^{n/2}}{\sqrt{\pi}}(\frac{n-1}{2})(\frac{n-3}{2})...(\frac{1}{2})\Gamma(\frac{1}{2}), n= odd$$

$$E(X^n)=\sigma^n((n-1)(n-3)...(1))$$

### 1.0.2. deviation and 离差 与变异 Variation

* 一个特定数值 对于其平均值的偏移, 称为离差 deviation
* 一个变量的各个数值, 对于其平均值的偏移, 称为 变异


### 1.0.3. coefficient of determination 可决系数 (可能属于概率论)

又称: 测定系数, 决定系数, 可决指数

表示一个随机变量于多个随机变量关系的数学特征, 用来反映回归模式说明因变量变化可靠程度的一个统计指标, 与 `负复相关系数` 类似    

定义: 已被模式中 -> 全部自变量说明的自变量的变差 对 自变量总变差 的比值

# 随机时间与概率  
