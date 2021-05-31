# 1. Data Science Solutions 

数据科学中实现一个解决方案的大致流程  

## 1.1. Workflow

工作目标：
* Classifying : 明确 samples 的类别, 可能从中得到classes与目标解决方案的隐含关联
* Correlating : 明确 features 之间, features 和 goal 之间的相关性, 为后面的处理做帮助
* Converting  : 根据选择的模型对数据进行预处理
* Completing  : 在 Converting 时可能需要对数据的 Missing Value 进行处理
* Correcting  : 检查 Training set 的 outliers 删掉无用的 Feature
* Creating    : 创建新 Feature, 基于 Correlation, Conversion, Completeness
* Charting    : 可视化数据.


### 1.1.1. 分析数据

1. 确认dataset 中的可用 feature
2. 确认features的类型, categorical or numerical or mixed 
   * 确认 distribution
   * 确认和目标 label 的 correlation, 统计或者作图
3. 修复数据
   * 确认哪些 features 可能会有 errors or `typos` 打印错误
   * 确认是否有 missing value

# 2. Machine Learning


Q: What is Cross-Validation?

A technique used to assess how well a model performs on a new independent dataset.  
Simpliest example of cross-validation is split data into two groups: training data and testing data.  
Training data to build the model and testing data to test the model.

Q: How to define/select metrics.

No one-sizes-fits-all metric.  
The metric chosen to evaluate a model depends on various factors:
- Is it a regression or classification task?
- What is the business objective? Eg. precision vs recall.
- What is the distribution of the target variable?   

Metrics: adjusted r-squared, MAE, MSE, accuracy, recall, precision, f1 score.

Q: 

# 3. 特征工程 - 抽出问题的特征向量

## 3.1. 文本表示模型

* 文本, 一种非结构化数据, 非常重要
* 文本数据的特征表示是非常重要的
* 在英文中, 同一个词有多种词形变化
  * 一般会对单词进行词干抽取 (Word Stemming)


### 3.1.1. TF-IDF

* TF-IDF 一般用来计算单词在文章中的权重
* $TF-IDF(t,d)=TF(t,d)\times IDF(t)$
* TF(t,d) 表示单词 t 在文档 d , 中出现的频率
* IDF(t)是逆文档频率, 用来衡量单词 t 对表达语义所起的重要性
  * $IDF(t)=log \dfrac{文章总数}{包含单词t的文章总数+1}$
  * 即, 如果一个单词在各种文章中都出现, 那么该单词可能没什么特殊意思, 通用词汇
  * 但是如果一个单词出现的非常少, 说明该单词非常专门化

### 3.1.2. Bag of Words 词袋模型

* 最基础的文本表示模型
* 将文章看成一袋子的单词
* 将一篇文章用一个长向量来表示
  * 每一个维代表字典中的一个单词
  * 维度的权重用TF-IDF来计算
* 缺点:
  * 将文章进行单词级别的划分, 但是没有考虑单词之间的组合含义
  * 容易丢失信息
  
### 3.1.3. N-gram

* 对词袋模型的改进
* 将连续出现的 $n\quad(n\le N)$ 个单词组成的词组也作为一个单独的特征放到向量表中

# 4. 模型评估 - 多种指标用于评估模型的性能

在模型应用中, 评估指标的选择问题是最容易被发现的问题, 也是最可能影响效果的因素.  
除了模型的评估指标之外:  
- 模型过拟合, 欠拟合
- 测试集和训练集的划分不合理
- 线下评估与线上测试的样本分布存在差异


Binary Classifier 是机器学习领域中最广泛的分类器.  
Precision, recall, F1 score, P-R 曲线都是评价二分类的指标.
- Accuracy 准确率
- Precision 精确率
- Recall 召回率
- RMSE Root Mean Square Error 均方根误差
- ROC  AOC  
  

## 4.1. 准确率 Accuracy

$Accuracy=n_{correct}/n_{total}$  
准确率属于分类模型的指标, 表示分类正确的样本占总样本个数的比例  

缺陷: 对样本比例的均衡有要求. Eg. 负样本占99%, 全部预测成负样本也有99%的准确度  

## 4.2. 精确率 Precision 和 召回率 Recall 和 F1 score

常被用做排序问题的评价指标: 没有一个确定的阈值来把得到的结果判定为正或者负, 而是采用 Top N 返回的结果来进行性能评估.  
- 精确率: 分类正确的正样本个数占分类器判定为正样本个数的比例
- 召回率: 分类正确的正样本个数占真正的正样本个数的比例
- 精确率和召回率存在矛盾与统一  
- 在实际的应用中, P-R 曲线应用的更多

1. 提高精确率会让模型变得更谨慎, 只会标记出可信度更高的样本, 而对于可信度相对不高的正样本容易漏掉导致 Recall 降低
2. 对于小众音乐/视频的搜索问题, 用户往往会寻找排在较靠后位置的数据, 因此对于 Recall 的要求更高
3. 即使 Precision@5 达到了100%, 对于上面的应用场景, Recall@5 也只有5%.

**P-R曲线**:
1. 横轴是Recall, 纵轴是 Precision
2. 对于 P-R 曲线上的一点, 代表了在某一阈值下, 模型将大于该与之的结果判定为正样本, 反之为负样本, 此时结果对应的 Recall 和 Precision.
3. 整条曲线是通过将阈值从高到低移动而生成的.
4. 原点代表啊当阈值最大时候的精确率和召回率.
5. Recall=0时曲线上的点代表召回率接近0的时候, 模型的精确率, 反之亦然


**F1 score**:
- F1 score 是由精准率和召回率派生而来的指标, 是精准率和召回率的调和平均值
- $F1=\frac{2*precision*recall}{precision+recall}$

## 4.3. RMSE 指标

Root Mean Squre Error  

一般情况下, RMSE可以很好的反应回归模型预测值与真实值的偏离程度    
$RMSE=\sqrt{\sum^n_{i=1}(y_i-\hat{y_i}^2)/n}$  

缺点: 在实际问题中, 如果存在个别偏离非常大的 Outlier, 即使离群点的数量非常少, 也会让 RMSE的指标变得非常差  
模型预测优秀但是RMSE指标高的解决方法:
1. 如果认为离群点是噪声的话, 就应该在数据预处理的时候将这些点过滤掉
2. 如果不认为离群点是噪声的话, 应该进一步提高模型的预测能力, 使得对离群点的预测也包含进去
3. 使用更合适的评价指标, Eg. MAPE ( Mean Absolute Percent Error )

$MAPE=\sum_{i=1}^n|\frac{y_i-\hat{y_i}}{y_i}|*\frac{100}{n}$

## 4.4. ROC曲线 Receiver Operating Characteristic Curve

中文名 受试者工作特征曲线  
* 横坐标: 假阳性率 False Positive Rate FPR=FP/N  N是负样本的个数
* 纵坐标: 真阳性率 True Positive Rate TPR=TP/P   P是正样本的个数
* 曲线从 原点出发, 最终到达 (1,1)点
* ROC曲线的绘制方法: 通过不断调整分类器的阈值, 来绘制ROC曲线
* 另一种画法是: 
  * 将横坐标以 1/N, 纵坐标以 1/P 为刻度, 从0,0 出发, 样本的预测概率从高到低进行排序, 依次遍历样本
  * 遇到一个正样本就沿纵轴方向绘制, 遇到一个负样本就沿横轴绘制, 直到 1,1 点

另一个指标 AUC ( Aera Under Curve )即为 ROC曲线下的面积大小, 可以反应模型性能  
* ROC 曲线一般都处于 y=x 的上方, 否则的话将预测输出改为 1-p 既可以得到一个更好的模型
* AUC的取值因此一般在 0.5~1, 越高代表模型越好  


ROC对比 P-R 曲线:
1. ROC在正负样本的分布发生变化的时候, 曲线的形状能基本保持不变, 而 P-R 曲线会发生较大变化. 因此ROC能够降低测试集改变带来的干扰  
2. ROC更加适用于 排序, 广告, 推荐等领域, 因为负样本数量更多
3. 对于研究者, P-R能够更清晰的看到模型在特定数据集上的表现, 更能反映其性能  

## 余弦距离

用于评估样本间的距离, 分析两个特征向量之间的相似性, 取值范围是 -1~1

# 5. 降维

* 对原始数据提取的特征进行降维
  * 减少冗余和噪声
  * 提高特征的表达能力
  * 降低训练复杂度

## 5.1. PCA Principal Components Analysis

* 如何定义主成分
  * 三位空间中的点分布在同一个平面上, 用三维坐标会有冗余
  * 找出平面, 然后用该平面上的二维坐标表示点, 即完成了数据降维
* 主成分分析
  * 特点: 线性, 非监督, 全局
  * 目标: 最大化投影方差, 让数据在主轴上投影的方差最大
  * 加入核映射, 得到核主成分分析(KPCA)
  * 加入流形映射的降维方法, eg. 等距映射, 局部线性嵌入, 拉普拉斯特征映射
  
### 5.1.1. 最大方差理论

* 根据信息理论, 信号具有较大方差, 噪声具有较小方差, 信号与噪声的比例成为信噪比
* PCA 的目标即最大化投影方差, 让数据在主轴上的投影的方差最大

推导过程:
1. 给定一组数据向量 $\{v_1,v_2,...,v_n \}$, 先将其中心化得到 $\{x_1,x_2,...,x_n \}$  
2. 向量内积表示为第一个向量投影到第二个向量上的长度, 对于单位向量 $\omega$, 投影可以表示为 $(x_i,\omega)=x_i^T\omega$  
3. 找到一个投影方向$\omega$, 使得$\{x_1,x_2,...,x_n \}$ 在该方向上的投影方差尽可能大. 且易知中心化后的向量投影后的均值也为0

- 投影后的方差为: $D(x)=\frac{1}{n}\sum_{i=1}^n(x_i^T\omega)^2=\omega^T(\frac{1}{n}\sum_{i=1}^nx_ix_i^T)\omega$  

4. 可以发现$(\frac{1}{n}\sum_{i=1}^nx_ix_i^T)$ 就是样本协方差矩阵
5. 将其化简为Σ后, 可以将PCA目标转化为 $max(\omega^T\sum\omega) s.t. (\omega^t\omega=1)$
6. 引入拉格朗日乘子, 求导后可推出 $\sum\omega=\lambda\omega\quad\rightarrow D(x)=\lambda$
- 即投影后的方差就是协方差矩阵的特征值, 目标的最大方差也就是最大特征值, 而目标最佳投影方向也就是最大特征值的特征向量, 

求解方法:
1. 对样本数据进行中心化处理
2. 求样本的协方差矩阵
3. 特征值分解, 并按大小排列
4. 降维目标是d维的话, 就选择前 d个特征值的特征向量, 然后进行映射, 方差较小的特征(噪声)自动被舍弃  
- $x_i'^T=[\omega_1^Tx_1,\omega_2^Tx_2,...,\omega_d^Tx_d]$
- 容易得知降维后得信息占比: 



### 5.1.2. 最小平方误差理论

# 6. 优化算法

* 机器学习算法 = 模型表征 + 模型评估 + 优化算法

## 6.1. 监督学习的损失函数

* 损失函数定义了模型的评估指标
* 没有损失函数就无法求解模型参数
* 不同损失函数有不同的优化难度
* 损失函数 $L(\cdot,\cdot)$
  * $L(f(x_i,\theta),y_i)$ 越小, 代表模型对该样本的匹配度越好


**对于二分类问题 $Y={-1,1}$:**   
- 目标: $signf(x_i,\theta)=y_i$
- `0-1损失`: 
  * $L_{0-1}(f,y)=1_{fy\le0}$
  * 当 fy 小于等于0的时候, loss=1, 否则等于0
  * 能够直观刻画分类的错误率
  * 由于其非凸且非光滑的特点, 很难对其进行优化  
- `Hinge损失`:
  * $L_{hinge}(f,y)=max{0,1-fy}$
  * $fy\ge1$的时候因为没有惩罚, loss=0 , 因此在 fy=1 处函数不可导
  * 不能使用梯度下降算法
  * 可以用次梯度下降法 Subgradient Descent Method  
- `Logistic损失`:
  * $L_{logistic}(f,y)=log_2(1+exp(-fy))$
  * 该函数处处光滑, 可以使用梯度下降法
  * 该函数对所有样本点都有惩罚, 即恒有 loss>0
  * 对异常值更敏感  
- `Cross Entropy`:
  * 交叉熵也是0-1损失的光滑凸上界
  * $L_{cross}(f,y)=-log_2\frac{1+fy}{2}$
  * $fy\ge1$的时候也没有惩罚, loss=0


**对于回归问题 $Y=R$:**   
- 目标: $f(x_i,\theta)\approx y_i$
- `平方损失`:
  * $L_{square}(f,y)=(f-y)^2$
  * 该函数光滑, 可以用梯度下降
  * 因为是平方, 对于异常点, 预测值偏离较远时, 容易惩罚过大
- `绝对损失`:
  * $L_{absolute}(f,y)=|f-y|$
  * 对于异常点相对更鲁棒一些
  * 在 f=y 处无法求导
- `Huber损失`:
  * 综合了平方损失和绝对损失
  * 在偏差较小时为平方损失, 偏差较大时为绝对损失
  * $$L_{Huber}(f,y)=\left\{\begin{aligned}(f-y)^2, |f-y|\le\delta\\2\delta|f-y|-\delta^2,|f-y|>\delta\end{aligned}\right.$$

## 6.2. 机器学习算法优化基础

* 机器学习模型的参数都可以写成优化问题
* 模型不同, 损失函数也不同, 对应的优化问题也不同
  * 可以分为 凸优化, 非凸优化
* 对于一道无约束优化问题 $min_\theta L(\theta)$, 已知 L 光滑
* 根据解决这个问题可以将经典的优化算法分为直接法和迭代法
  * 直接法: 利用数学直接求解出最优解
  * 迭代法: 迭代地修正对最优解的估计

### 6.2.1. 凸优化

* 对于函数 $L(\cdot)$ , 函数图形为向下凹进去的
* 对于任意两点 x,y和实数 lambda=(0~1)
  * $L(\lambda x+(1-\lambda)y\le \lambda L(x)+(1-\lambda)L(y))$
  * 函数曲面上的任意两点连线, 线上的任意一点都不会处于函数图形的下方 (都在上方或者等于)
* 凸优化问题的局部极小值就是全局极小值, 因此比较容易求解

- 逻辑回归对应的优化就是凸优化
  * $L_i(\theta)=log(1+exp(-y_i\theta^Tx_i))$
  * 目标为求出对所有样本, Loss的和最小的参数 theta
  * 对Loss求导: $\triangledown L_i(\theta)=\dfrac{-y_ix_i}{1+exp(y_i\theta^Tx_i)}$
  * 继续求二阶导 $\triangledown^2L_i(\theta)=\dfrac{exp(y_i\theta^Tx_i)}{(1+exp(y_i\theta^Tx_i))^2}x_ix_i^T$
  * 容易看出 $\triangledown^2L_i(\theta) \ge 0$, 因此原函数为凸函数


### 6.2.2. 非凸优化

* 一般来说非凸优化被认为比较难求解, 但PCA是一个特例
* 借助 SVD 可以直接得到主成分分析的全局极小值

- 主成分分析的优化函数
  * $min_{VV^T=I_k}L(V)=||X-V^TVX||_F^2$
  * 根据凸函数的定义, 只有一个全局最小值
  * 根据该函数的定义, 假设$V^*$是全局最小值, 那么$-V^*$ 也是全局最小值
  * 代入凸函数定理 
    * $L(0.5V^*+0.5(-V^*))=L(0)=||X||_F^2$
    * $0.5L(V^*)+0.5L((-V^*))=||X-V^{*T}V^*X||_F^2$
    * 易知式一大于式二, 两点连线上的值`小于`函数上的对应点

### 6.2.3. 直接法求解最优解

* 直接法对于问题有严苛的两个限制, 因此应用场景非常小
  * 需要问题 $L(\cdot)$是凸函数, 即求导后=0即为最优解
  * 需要梯度为零0的式子可解 $\triangledown L(\theta^*)=0$ 有闭式解

- 岭回归问题 (Ridge Regression)
  * 岭回归问题是对最小二乘法的一个补充, 损失了无偏性, 但是得到了高数值稳定性
  * 目标函数 $L(\theta)=||X\theta -y||_2^2+\lambda||\theta||_2^2$