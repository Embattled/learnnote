- [1. Data Science Solutions](#1-data-science-solutions)
  - [1.1. Workflow](#11-workflow)
  - [1.2. 面试QA](#12-面试qa)
- [2. Machine Learning 机器学习基础](#2-machine-learning-机器学习基础)
  - [2.1. 学习算法](#21-学习算法)
  - [2.2. 拟合](#22-拟合)
- [3. 特征工程 - 抽出问题的特征向量](#3-特征工程---抽出问题的特征向量)
  - [3.1. 通用特征处理](#31-通用特征处理)
    - [3.1.1. 特征归一化](#311-特征归一化)
    - [3.1.2. 类别型特征(数据)](#312-类别型特征数据)
    - [3.1.3. 组合特征与降维处理](#313-组合特征与降维处理)
  - [3.2. 文本表示模型](#32-文本表示模型)
    - [3.2.1. TF-IDF 单词权重计算](#321-tf-idf-单词权重计算)
    - [3.2.2. Bag of Words 词袋模型](#322-bag-of-words-词袋模型)
    - [3.2.3. N-gram](#323-n-gram)
    - [3.2.4. Word2Vec 谷歌2013提出的词嵌入模型](#324-word2vec-谷歌2013提出的词嵌入模型)
  - [3.3. 图像数据增强 过拟合防止](#33-图像数据增强-过拟合防止)
- [4. 模型评估 - 多种指标用于评估模型的性能](#4-模型评估---多种指标用于评估模型的性能)
  - [4.1. 准确率 Accuracy](#41-准确率-accuracy)
  - [4.2. 精确率 Precision 和 召回率 Recall 和 F1 score](#42-精确率-precision-和-召回率-recall-和-f1-score)
  - [4.3. RMSE 指标](#43-rmse-指标)
  - [4.4. ROC曲线 Receiver Operating Characteristic Curve](#44-roc曲线-receiver-operating-characteristic-curve)
  - [4.5. 余弦距离](#45-余弦距离)
  - [4.6. A/B 测试](#46-ab-测试)
  - [4.7. 模型评估的抽样方法](#47-模型评估的抽样方法)
- [5. 降维](#5-降维)
  - [5.1. PCA Principal Components Analysis](#51-pca-principal-components-analysis)
    - [5.1.1. 最大方差理论](#511-最大方差理论)
    - [5.1.2. 最小平方误差理论](#512-最小平方误差理论)
  - [5.2. LDA Linear Discriminant Analysis](#52-lda-linear-discriminant-analysis)
- [6. 监督学习基础算法 Supervised Learning](#6-监督学习基础算法-supervised-learning)
  - [6.1. Support Vector Machines](#61-support-vector-machines)
- [7. 非监督学习基础算法 Unsupervised Learning](#7-非监督学习基础算法-unsupervised-learning)
  - [7.1. 聚类算法](#71-聚类算法)
    - [7.1.1. K-means](#711-k-means)
    - [7.1.2. x-means](#712-x-means)
    - [7.1.3. Gap Statistic](#713-gap-statistic)
    - [7.1.4. ISODATA 迭代自组织数据分析法](#714-isodata-迭代自组织数据分析法)
    - [7.1.5. Self-Organizing Map SOM 自组织映射神经网络](#715-self-organizing-map-som-自组织映射神经网络)
- [8. 优化算法](#8-优化算法)
  - [8.1. 监督学习的损失函数](#81-监督学习的损失函数)
  - [8.2. 机器学习算法优化基础](#82-机器学习算法优化基础)
    - [8.2.1. 凸优化](#821-凸优化)
    - [8.2.2. 非凸优化](#822-非凸优化)
    - [8.2.3. 直接法求解最优解](#823-直接法求解最优解)
- [RANSAC Random Sample Consensus](#ransac-random-sample-consensus)

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


分析数据的过程：
1. 确认dataset 中的可用 feature
2. 确认features的类型, categorical or numerical or mixed 
   * 确认 distribution
   * 确认和目标 label 的 correlation, 统计或者作图
3. 修复数据
   * 确认哪些 features 可能会有 errors or `typos` 打印错误
   * 确认是否有 missing value
## 1.2. 面试QA

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



# 2. Machine Learning 机器学习基础

机器学习算法即 - 能够从数据中学习的算法  
* 机器学习本质上属于应用统计学
  * 统计学的两种主要方法
    * 频率派估计
    * 贝叶斯推理
  * 大部分算法都具有超参数 (在算法之外定义)
* 机器学习算法可以分成
  * 监督学习    : 回归, 分类, 等结构化输出问题
  * 无监督学习  : 密度估计等
  * 半监督学习
    * 一部分样本有监督目标, 另一部分没有
    * 整个样本集合被标记为 正负, 但是单独的样本例没有标记

## 2.1. 学习算法

* 给定经验 E , 应用于任务 T, 获得性能度量 P 的提升
* 样本 (example) 是已经从事件中收集来的量化后的 特征(feature) 集合
* 当前机器学习任务举例
  * 分类
    * 输入缺失分类
  * 回归
  * 结构化输出
    * 转录 - OCR, NLP
    * 机器翻译
    * 图片标识 , segmentation, 文字表述图片

机器学习的性能
* 错误率 (errorrate) : 模型的 0-1 损失的期望


## 2.2. 拟合

* 模型拟合各种函数的能力被称为 模型的容量 capacity
  * 容量小的模型容易 欠拟合
  * 容量大的模型容易 过拟合


**欠拟合避免**:
* 添加新特征
* 增加模型复杂度
* 减少正则化系数

**过拟合避免**:

1. 数据方面:  获取更多的训练数据, 数据增强
2. 降低模型复杂度
3. 正则化方法: 对于传统的函数拟合, 将模型的权值大小加入到损失函数中 L1, L2
4. 集成学习


# 3. 特征工程 - 抽出问题的特征向量

特征工程: 对原始数据进行加工, 提炼为特征
* 特征工程是一个表示和展现数据的过程
* 旨在取出原始数据中的杂质和冗余


数据的两种常用类型
* 结构化数据: 关系型数据库的表
* 非结构化数据: 图像, 音频, 视频

## 3.1. 通用特征处理

对于表结构数据或者已经提取成向量的特征所进行的一系列预处理

### 3.1.1. 特征归一化

消除特征之间的量纲差别: 使得所有特征都统一到一个大致相同的数值区间  
* 对于迭代类型的算法模型比较重要, 对于决策树模型则不太适用
* 决策树主要依据特征x的信息增益来决定是否进行分裂  



* 线性函数归一化 (Min-Max Scaling) 线性归一到 0~1 区间, 等比缩放
  * $$X_{norm}=\frac{X-X{min}}{X_{max}-X_{min}}$$ 
* 零均值归一化 (Z-Score Normalization) 将原始数据映射到 均值0, 标准差为 1 的分布上
  * 需要计算原始特征的均值 $\mu$, 标准差 $\sigma$
  * 定义归一化公式 $Z=\frac{x-\mu}{\sigma}$
  * 对于梯度下降来说, 可以让迭代空间呈现圆形, 降低到最优解的距离

### 3.1.2. 类别型特征(数据)

* 离散的特征(数据形式): 只能在有限选项内取值的特征
* 除了决策树等少量算法, 大部分模型只支持数值类型的特征, 需要进行转化


1. 序号编码: 用于处理类别间具有大小关系的数据 (高中低), 在保留大小关系的情况下赋予数值
2. 独热编码(One-Hot): 类别间无任何关系, 根据选项的可能数n, 将该单个特征转换成n维的01稀疏向量 
   * 视情况而定使用稀疏向量的特殊表示方法来节省空间
   * 不要让该编码导致特征向量维度过高, 适度进行特征筛选很有必要
3. 二进制编码: 相当是对One-Hot的改进, 将各个种类编码成 $log_2n$ 维度的二进制特征

### 3.1.3. 组合特征与降维处理

对于类别型特征(离散特征)
* 为了提高复杂关系的拟合能力, 
* 最简单的两两组合, 两个特征组合后的特征选项个数是 m*n
* 如果某一方的特征是 用户id 等具有特大数量级种类的离散特征, 将会导致组合特征不能被应用
* 较为复杂的组合方法: 决策树构建组合特征


对于直接学习 m*n 规模特征很困难的应用场景, 可使用类似于矩阵分解的降维思路
* 将两个特征映射到低维k维的特征
* 要学习的特征变为两个映射参数
* $m\times k_1+n\times k_2$


## 3.2. 文本表示模型

* 文本, 一种非结构化数据, 非常重要
* 文本数据的特征表示是非常重要的
* 在英文中, 同一个词有多种词形变化
  * 一般会对单词进行词干抽取 (Word Stemming)

### 3.2.1. TF-IDF 单词权重计算

TF-IDF 一般用来计算单词在文章中的权重

* $`TF-IDF'(t,d)=TF(t,d)\times IDF(t)$
* TF(t,d) 表示单词 t 在文档 d , 中出现的频率
* IDF(t)是逆文档频率, 用来衡量单词 t 对表达语义所起的重要性
  * $IDF(t)=log \dfrac{文章总数}{包含单词t的文章总数+1}$
  * 即, 如果一个单词在各种文章中都出现, 那么该单词可能没什么特殊意思, 通用词汇
  * 但是如果一个单词出现的非常少, 说明该单词非常专门化


### 3.2.2. Bag of Words 词袋模型

* 最基础的文本表示模型
* 将文章看成一袋子的单词
* 将一篇文章用一个长向量来表示
  * 每一个维代表字典中的一个单词
  * 维度的权重用TF-IDF来计算
* 缺点:
  * 将文章进行单词级别的划分, 但是没有考虑单词之间的组合含义
  * 容易丢失信息


### 3.2.3. N-gram

* 对词袋模型的改进
* 将连续出现的 $n\quad(n\le N)$ 个单词组成的词组也作为一个单独的特征放到向量表中

### 3.2.4. Word2Vec 谷歌2013提出的词嵌入模型

词嵌入: 将词向量化的一类模型的统称
* 将每个词都映射成低维空间 K (50~300维) 上的一个稠密向量
* 低维空间的每一个维度可以看作一个隐含的主题
* 对于有 N 个单词的文档, 可以用 N*K 的矩阵来直接表示该文章 

Word2Vec是谷歌提出的词嵌入模型, 是一个浅层的神经网络模型, 拥有两种网络结构
* CBOW (Continues Bag of Words): 根据上下文出现的词语来预测当前词的生成概率
* Skip-gram: 根据当前词来预测上下文中各词的生成概率
* 输入的每个词用 One-Hot 编码表示, 特征维度为N(单词字典的单词数)
* 


## 3.3. 图像数据增强 过拟合防止

此处为简单介绍, 在数据不足的情况下尽可能提升模型的效果


基于模型的方法: 降低过拟合的一系列 trick
* 简化模型 e.g. 非线性模型转为线性模型
* 添加约束项
* 集成学习
* Droupout

基于数据的方法: Data Augmentation (需要各种先验知识)
* 一定程度的随机几何变换, 裁剪, 填充
* 噪声扰动
* 颜色变换, RGB空间主成分分析
* 亮度, 清晰度, 对比度, 锐度


其他:
* 生成式对抗网络得到合成数据
* 迁移学习

# 4. 模型评估 - 多种指标用于评估模型的性能

在模型应用中, 评估指标的选择问题是最容易被发现的问题, 也是最可能影响效果的因素.  
除了模型的评估指标之外:  
- 模型过拟合, 欠拟合
- 测试集和训练集的划分不合理
- 线下评估与线上测试的样本分布存在差异


Binary Classifier 是机器学习领域中最广泛的分类器.  
Precision, recall, F1 score, P-R 曲线最早都是评价二分类的指标.
- Accuracy 准确率
- Precision 精确率
- Recall 召回率
- RMSE Root Mean Square Error 均方根误差
- ROC  AOC  
  

## 4.1. 准确率 Accuracy

$Accuracy=n_{correct}/n_{total}$  
准确率属于分类模型的指标, 表示分类正确的样本占总样本个数的比例  

* 缺陷: 对样本比例的均衡有要求. 
* Eg. 负样本占99%, 全部预测成负样本也有99%的准确度  

## 4.2. 精确率 Precision 和 召回率 Recall 和 F1 score

常被用做排序问题的评价指标: 没有一个确定的阈值来把得到的结果判定为正或者负, 而是采用 Top N 返回的结果来进行性能评估.  
- 精确率 Precision  : 分类正确的正样本个数占分类器判定为正样本个数的比例
- 召回率 Recall     : 分类正确的正样本个数占真正的正样本个数的比例
- 精确率和召回率存在矛盾与统一  
- 在实际的应用中, P-R 曲线应用的更多

1. 提高精确率会让模型变得更谨慎, 只会标记出可信度更高的样本, 而对于可信度相对不高的正样本容易漏掉导致 Recall 降低
2. 对于小众音乐/视频的搜索问题, 用户往往会寻找排在较靠后位置的数据, 因此对于 Recall 的要求更高
3. 即使 Precision 达到了100%, 对于上面的应用场景, Recall 也只有5%.

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
3. 使用更合适的评价指标, Eg. 平均绝对百分比误差 MAPE ( Mean Absolute Percent Error )

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

## 4.5. 余弦距离

用于评估样本间的距离, 分析两个特征向量之间的相似性, 
* 余弦相似度  : 取值范围是 -1~1, 相同向量的相似度是1
* 余弦距离, 将 1 减去余弦相似度, 取值为 0~2, 相同向量的距离是0


余弦相似度: $cos(A,B)=\frac{A\cdot B}{||A||_2||B||_2}$  
欧氏距离: $d(A,B)=\left|| A-B \right||_2$
* 欧氏距离体现数值上的绝对差异
* 余弦相似度体现方向上的相对差异, 对于维度的长度差距不敏感
* 对于相同内容的长短文本, 如果使用词频或者词向量作为特征
  * 容易导致欧氏距离过高
  * 对于余弦相似度则能仍版保持正交是为0, 相同时为1
* 需要进行相似性判断的时候适合使用余弦相似度
* 需要进行量化特征比较的时候适合欧氏距离


**余弦距离是否是严格定义的距离**
* 距离: 在一个集合中, 每一对元素均可以唯一确定一个实数
  * 正定性: d(x,y)=0 当且仅当 x=y
  * 对称性: d(x,y)=d(y,x)  
  * 三角不等式: 任意两边之和大于第三边
* 余弦距离不满足三角不等式, 不属于标准距离
* 在机器学习领域不满足于标准距离定义的其他量化还有很多 e.g. KL距离 (相对熵)

## 4.6. A/B 测试

A/B 测试是验证新模块的有效性, 在机器学习中是验证模型最终效果的主要手段
* 相比于模型开发适合的离线评测, A/B 测试属于线上评测
* 对用户进行分桶, 分成实验组和对照组
* 实验组使用新模型, 同时保证样本的独立性和采样方式的无偏性
* 直接比较新模型的 点击率, 存留时长等市场价值


## 4.7. 模型评估的抽样方法

样本分组为训练集和测试集, 存在多种分组方法  

* Holdout 检验: 朴素的随机分组, 因此最终的评估指标数据也带有随机性
* 交叉检验 : 重复进行多次训练和评估, 最终取平均
  * k-fold  : 随机分成 k 个子集, 进行 k 次, 每次用除了第 i 个子集以外的子集作为训练集
  * 留一验证 : k-fold 的特殊情况, k = n (样本总数)
* 自助法  : 针对样本总数及其小的情况
  * 从包含 n 个样本的数据集中, 进行 n 次有放回的随机抽样
  * 包含重复的 n 个样本作为训练集
  * 从没有被抽出过的样本作为验证集
  * 极限情况, 当 n = 无穷, 有大约 1/e 的样本没被抽样过



# 5. 降维

* 对原始数据提取的特征进行降维
  * 减少冗余和噪声
  * 提高特征的表达能力
  * 降低训练复杂度
* PCA 主成分分析法  : 非监督, 将原始数据映射到一些方差较大的方向上
* LDA 线性判别分析  : 有监督的降维方法, 会考虑标签. 最大化类间距 最小化类内距
* 基本应用原则: 对无监督任务使用PCA, 对有监督任务使用LDA
* PCA与LDA的联系
  * 求解过程相似, 都是计算矩阵的特征向量作为最佳投影方向
  * 语音识别
    * 先PCA降维, 过滤固定频率
    * 后LDA降维, 获取不同人的区分性特征

## 5.1. PCA Principal Components Analysis

主成分分析法

* 如何定义主成分
  * 三位空间中的点分布在同一个平面上, 用三维坐标会有冗余
  * 找出平面, 然后用该平面上的二维坐标表示点, 即完成了数据降维
* 主成分分析
  * 特点: 线性, 非监督, 全局
  * 目标: 最大化投影方差, 让数据在主轴上投影的方差最大
  * 加入核映射, 得到核主成分分析(KPCA)
  * 加入流形映射的降维方法, eg. 等距映射, 局部线性嵌入, 拉普拉斯特征映射
* 不论是最大方差还是最小化平方误差, 都能推导出相同的公式
  
### 5.1.1. 最大方差理论

* 根据信息理论, 信号具有较大方差, 噪声具有较小方差, 信号与噪声的比例成为信噪比
* PCA 的目标即最大化投影方差, 让数据在主轴上的投影的方差最大

推导过程:
1. 给定一组数据向量 $\{v_1,v_2,...,v_n \}$, 先将其中心化得到 $\{x_1,x_2,...,x_n \}$  
2. 向量内积表示为第一个向量投影到第二个向量上的长度, 对于单位向量 $\omega$, 投影可以表示为 $(x_i,\omega)=x_i^T\omega$  
3. 找到一个投影方向$\omega$, 使得$\{x_1,x_2,...,x_n \}$ 在该方向上的投影方差尽可能大. 且因为进行了中心化, 易知中心化后的向量投影后的均值也为0

- 投影后的方差为: $D(x)=\frac{1}{n}\sum_{i=1}^n(x_i^T\omega)^2=\omega^T(\frac{1}{n}\sum_{i=1}^nx_ix_i^T)\omega$  

4. 可以发现$(\frac{1}{n}\sum_{i=1}^nx_ix_i^T)$ 就是样本协方差矩阵
5. 将其化简为Σ后, 可以将PCA目标转化为 $max(\omega^T\sum\omega) s.t. (\omega^t\omega=1)$
6. 引入拉格朗日乘子, 求导后可推出 $\sum\omega=\lambda\omega\quad\rightarrow D(x)=\lambda$
- 即投影后的方差就是协方差矩阵的特征值, 目标的最大方差也就是最大特征值, 而目标最佳投影方向也就是最大特征值的特征向量, 

求解方法:
1. 对样本数据进行中心化处理
2. 求样本的协方差矩阵
3. 特征值分解, 并按大小排列
4. 降维目标是d维的话, 就选择前 d个特征值的特征向量作为投影方向, 然后进行映射得到d维特征值, 方差较小的特征(噪声)自动被舍弃  
- $x_i'^T=[\omega_1^Tx_1,\omega_2^Tx_2,...,\omega_d^Tx_d]$
- 容易得知降维后得信息占比: 



### 5.1.2. 最小平方误差理论

PCA求解的最佳投影方向, 就是一个 d 维的平面
* 若 d=1 就是求解一个最佳直线, 所有点到直线的距离平方和最小
* 可以类比成线性回归问题

计算过程
* 对于某个d维超平面D, 可用 d 个标准正交准基 $W={\omega_1,\omega_2,...,\omega_d}$构成
* 对于一个数据点$x_k$, 用$\tilde{x_k}$来表示到超平面 D 的投影
* 根据线性代数理论, 可用基线性表示 $\tilde{x_k}=\sum_{i=1}^d(\omega_i^Tx_k)\omega_i$
* 即, $\tilde{x_k}$就是$x_k$在W这组正交基下的坐标
* 证明暂略



## 5.2. LDA Linear Discriminant Analysis

线性判别分析: 既是一种有监督学习算法, 同时也可以用来对数据进行降维  
* 从出发点来看, LDA 是为了分类服务的
* 找到一个投影方向, 使得投影后的样本尽可能按照原始类被分开

从二分类问题:
* 设样本类为 $C_1,C_2$, 两类样本的均值为$\mu_1,\mu_2$
* 希望投影之后两类之间的距离尽可能大:
  * 有两类的中心在$\omega$方向上的投影向量 $\tilde{\mu_1}=\omega^T\mu_1, \tilde{\mu_2}=\omega^T\mu_2$
  * 有距离 $D(C_1,C_2)=||\tilde{\mu_1}-\tilde{\mu_2}||_2^2$
  * 问题可以化为 $D(C_1,C_2)=max_\omega ||\omega^T(\mu_1-\mu_2)||_2^2$
  * 可发现当 $\omega$ 与 $(\mu_1-\mu_2)$ 方向一致时, 距离可达到最大值
  * 单纯要求类中心距离大容易导致样本特征重叠, 因此可引出第二个优化要求
* 最小化类内距离
  * 定义$D_1,D_2$为投影后的类内方差, 有 $D=\sum(\omega^Tx-\omega^T\mu_1)^2=\sum\omega^T(x-\mu_1)(x-\mu_1)^T\omega$
  * 定义类间散度矩阵$S_B$, 类内散度矩阵 $S_w$
  * 定义优化目标: $max_\omega J(\omega)=\frac{D(C_1,C_2)}{D_1+D_2}=\frac{\omega^TS_B\omega}{\omega^TS_w\omega}$
  * 求解只需对 $\omega$求导, 令导数为0
* 最大化的目标对应了一个矩阵的特征值, 因此LDA 的计算实际上就是求特征向量
* 因为计算投影方向$\omega$ 不需要长度
* 有 $\omega=S_w^{-1}(\mu_1-\mu_2)$ 即只需要样本均值和类内方差, 立刻可以得到最佳投影方向


# 6. 监督学习基础算法 Supervised Learning

## 6.1. Support Vector Machines

* SVMs are a set of supervised learning methods used for classification, regression and outliers detection.
* 优点
  * Effective in high dimensional spaces.
  * Still effective in cases where number of dimensions is greater than the number of samples.
  * Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.
  * Versatile: different Kernel functions can be specified for the decision function. 
* 缺点
  * If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.
  * SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation.

# 7. 非监督学习基础算法 Unsupervised Learning

非监督学习的主要任务:
* 推荐算法

非监督学习主要包含两大类学习方法:
* 数据聚类      : 通过多次迭代找到数据的最优分割
* 特征变量关联  : 各种相关性分析找到变量之间的关系

## 7.1. 聚类算法

通过多次迭代来找到数据的最优分割

### 7.1.1. K-means

通过迭代方式寻找 K 个簇, 使得代价函数最小:
* 代价函数 : 各个样本距离所属簇中心点的误差平方和
* c: 每个数据x所属的簇, u: 簇对应的中心点
* $J(c,u)=\sum^M_{i=1}||x_i-\mu_{ci}||^2$

优点:
* 适合于大数据集, 计算复杂度是 O(NKt), N: 数据量 K: 簇数 t: 迭代数
* 算法可伸缩
缺点:
* 受初值和离群点的影响大, 每次结果不稳定
* 难以面对簇分布差别比较大的情况
* 结果属于局部最优解, 无法得到全局最优解
* 必须手动选择 K 值
* 样本只能被划分到一个簇内, 无法对应多簇情况


K-means的优化:
* 数据归一化和离散点处理: 由于 K-means 是基于欧氏距离的算法
* 合理选择 K 值: 基于多次实验的手肘法 (elbow)
  * 尝试不同的 K 值做成图标, 选择梯度下降速度的拐点作为 K
* 使用核函数:
  * 通过非线性映射, 使得数据点映射到高维的特征空间, 并在新空间中进行聚类

K-means 算法流程:
* 预处理: 归一化, 离群点
* 初值 : 随机选取 k 个簇中心点
* 定义迭代步数 t, 直至损失函数收敛
  * 对每一个样本, 计算其最近的簇并划分到该簇
  * 对于每一个簇, 根据簇内的点重新计算簇中心

K-means ++
* 改进了初始值的生成方法:
* 第一个簇中心通过随机方法
* 选取第 n+1 个中心点时, 距离当前 n 个聚类中心越远的点会有更高的概率被选为聚类中心
### 7.1.2. x-means

一种基于 K-means 的改进算法:
* 完善了
  * k-means 的计算规模受限, 每次迭代需要大量的计算
  * 必须手动指定 K 值
* 但是保留了 : 局部最小值的缺点  

如果确定 K 值:
* 确定一个 K 值下限
* 在此基础上向上搜索, 找到最好的 `BIC` 得分 (Bayesian Information Criterion)




### 7.1.3. Gap Statistic

自动化确定 K 值的基于蒙特卡洛模拟的方法:
* 记 Dk 为 k 簇 时候的损失函数
* 在样本空间按照均匀分布的随机产生和原始数据一样多的随机样本, 并对这个随机样本进行 k-means
  * 重复多次得到对应随机情况 Dk 的期望 
  * 对一定范围内的 k 都进行蒙特卡洛模拟
* 定义 Gam Statistic $Gap(k)=E(logD_k)-logD_k$
* 当取得最佳 k 的时候, Gap(k) 也应取得最小

### 7.1.4. ISODATA 迭代自组织数据分析法

基于 k-means, 加入了对簇个数的自动化增删
* 如果某个类别的样本数过少, 则删去该簇
* 如果某个类别的样本数过多, 分散较大, 则将该类分割成两个子类

缺点: 需要手动指定较多的 参数
* 预期的聚类中心数 Ko: 尽管不用手动指定最终的K, 但仍然需要一个大致的范围, 一般最终结果是 `Ko/2 ~ 2Ko`
* 最少样本数目: Nmin, 每个类别所需要的最少数目
* 最大方差 Sigma: 控制类的分散程度, 如果超过该阈值则会分裂
* 聚类中心的最小聚氯 Dmin : 如果两个簇中心点过近, 则会合并


### 7.1.5. Self-Organizing Map SOM 自组织映射神经网络

...

# 8. 优化算法

* 机器学习算法 = 模型表征 + 模型评估 + 优化算法

## 8.1. 监督学习的损失函数

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

## 8.2. 机器学习算法优化基础

* 机器学习模型的参数都可以写成优化问题
* 模型不同, 损失函数也不同, 对应的优化问题也不同
  * 可以分为 凸优化, 非凸优化
* 对于一道无约束优化问题 $min_\theta L(\theta)$, 已知 L 光滑
* 根据解决这个问题可以将经典的优化算法分为直接法和迭代法
  * 直接法: 利用数学直接求解出最优解
  * 迭代法: 迭代地修正对最优解的估计

### 8.2.1. 凸优化

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


### 8.2.2. 非凸优化

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

### 8.2.3. 直接法求解最优解

* 直接法对于问题有严苛的两个限制, 因此应用场景非常小
  * 需要问题 $L(\cdot)$是凸函数, 即求导后=0即为最优解
  * 需要梯度为零0的式子可解 $\triangledown L(\theta^*)=0$ 有闭式解

- 岭回归问题 (Ridge Regression)
  * 岭回归问题是对最小二乘法的一个补充, 损失了无偏性, 但是得到了高数值稳定性
  * 目标函数 $L(\theta)=||X\theta -y||_2^2+\lambda||\theta||_2^2$


# RANSAC Random Sample Consensus

于 1981 年由 Fischler 和 Bolles 最早提出  

根据一组包含异常数据的样本数据集, 计算出数据的数学模型参数, 从而反过来得到有效样本数据的算法, 该算法假设  
* 数据集中有 inliers 和 outliers, 即存在噪声
* 根据数据集中的一小组 inliers 是可以计算出匹配所有 inliers 的模型的正确参数的  

基本思想: 样本集 P
1. 考虑一个最小抽样集 S, |S|=n , 从 P 中随机抽取 n 个样本初始化模型 M
2. 余集(P-S) 中, 与初始化模型 M 的误差小于一定阈值 t 的样本 与 S 一起构成了 S* , 此时假设认为 S* 中的点都是 inliers
3. 若 |S*|> 一定的数量, 则认为该模型合理, 利用 S* 重新计算新的模型 M*, 再重新抽取 S, 重复该过程
4. 若重复一定次数都未找到 Consensus Set, 则算法失败, 否则根据所找到的样本数最多的 S* 作为结论的 inliers


* 优点: 鲁棒的估计模型参数
* 缺点: 不存在迭代上限, 不保证得到正确结果, 需要正确选择超参数, 只能得到一个模型, 如果数据存在两个合理模型, 则无法找到    