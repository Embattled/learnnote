- [1. scikit-learn introduction](#1-scikit-learn-introduction)
  - [1.1. Model Persistence](#11-model-persistence)
  - [1.2. Model Operation](#12-model-operation)
- [2. Supervised learning 模型](#2-supervised-learning-模型)
  - [2.1. Linear Models 线性模型](#21-linear-models-线性模型)
    - [2.1.1. Logistic regression](#211-logistic-regression)
    - [2.1.2. SGD - Stochastic Gradient Descent](#212-sgd---stochastic-gradient-descent)
  - [2.2. Support Vector Machines](#22-support-vector-machines)
  - [2.3. Nearest Neighbors 最近邻](#23-nearest-neighbors-最近邻)
    - [2.3.1. NN Classification](#231-nn-classification)
    - [2.3.2. NN Regression](#232-nn-regression)
  - [2.4. Decision Trees](#24-decision-trees)
  - [2.5. Ensemble Methods](#25-ensemble-methods)
    - [2.5.1. Voting Classifier](#251-voting-classifier)
    - [2.5.2. Voting Regressor](#252-voting-regressor)
  - [2.6. 监督学习的 Neural Network models](#26-监督学习的-neural-network-models)
- [3. Unsupervised learning 模型](#3-unsupervised-learning-模型)
  - [3.1. cluster](#31-cluster)
    - [3.1.1. DBSCAN](#311-dbscan)
    - [3.1.2. KMeans](#312-kmeans)
- [4. Model selection and evaluation](#4-model-selection-and-evaluation)
- [5. Dataset loading utilities](#5-dataset-loading-utilities)
  - [5.1. Loading other datasets](#51-loading-other-datasets)
  - [5.2. Generated datasets](#52-generated-datasets)
  - [5.3. Toy dataset](#53-toy-dataset)
  - [5.4. Real world datasets](#54-real-world-datasets)
- [6. decomposition](#6-decomposition)
  - [6.1. FactorAnalysis](#61-factoranalysis)
  - [6.2. PCA](#62-pca)
- [7. manifold](#7-manifold)
    - [7.0.1. TSNE](#701-tsne)
- [8. preprocessing](#8-preprocessing)
  - [8.1. scale](#81-scale)
- [9. collections](#9-collections)
# 1. scikit-learn introduction

* 目前非常有名的开源机器学习库
* 导入 : `import sklearn`

## 1.1. Model Persistence

如何去保存一个 scikit-learn 模型  

1. 使用 python 自带的 pickle 包
2. `Open Neural Network Exchange` 或者 `Predictive Model Markup Language (PMML)`

## 1.2. Model Operation

记录一下可能是所有模型类通用的方法
  



# 2. Supervised learning 模型

监督性学习是最基础的模型, 也是一个用最多的模型  
在该大分类下没有特定的库头  

## 2.1. Linear Models 线性模型

`sklearn.linear_model.*`  

* sckikit 提供了一系列线性模型用于处理回归任务 regression
* 即目标值可以表示为线性特征的组合 $\hat{y}(w,x)=w_0+w_1x_1+...+w_px_p$  
* 在 scikit 中, 定义了权值 vector $w=(w_1,...,w_p)$ 为 `coef_`, 而偏移 $w_0$ 为 `interept_`   

要使用 线性模型进行分类任务的话, 使用 Logistic regression.  

### 2.1.1. Logistic regression

A linear model for classification rather than regression.  
该模型还有其他名字:
* logit regression
* maximum-entropy classification (MaxEnt)
* log-linear classifier

`sklearn.linear_model.LogisticRegression()`:  
- 可以进行 binary, One-vs-Rest, multinomial
- 可以使用的 regularization 包括 $l_1,l_2, Elastic-Net$ , 默认使用   
- 不使用 regulatization 相当于将 entropy 的权重 C 设置的非常高  

该函数实现的 solvers:
- liblinear
- libfgs
- newton-cg
- sag
- saga

### 2.1.2. SGD - Stochastic Gradient Descent

最简单的拟合模型的方法  


## 2.2. Support Vector Machines

`from sklearn import svm`  

总共提供了4中不同的SVM算法以及对应的7个类  
| 类名            | 功能                                  |
| --------------- | ------------------------------------- |
| svm.LinearSVC   | Linear Support Vector Classification. |
| svm.LinearSVR   | Linear Support Vector Regression.     |
| svm.NuSVC       | Nu-Support Vector Classification.     |
| svm.NuSVR       | Nu Support Vector Regression.         |
| svm.SVC         | C-Support Vector Classification.      |
| svm.SVR         | Epsilon-Support Vector Regression.    |
| svm.OneClassSVM | Unsupervised Outlier Detection.       |

通用类初始话参数：
* `decision_function_shape`: {'ovo','ovr'} default= 'ovr'
  * ovr = one versus rest , 即在多类别分类时会产生 n 个分类器
  * ovo = one versus one  , 产生 n*(n-1)/2 个分类器

## 2.3. Nearest Neighbors 最近邻

`sklearn.neighbors`  

最近邻也是经典方法, 虽然在教程索引中被包含在了监督学习中, 但事实上提供了 最近邻基础的监督学习 和 非监督学习的方法. 
* 最近邻也是无参数学习 non-parametric 
* 非监督性的最近邻是其他学习方法的基础 eg. notably manifold learning, spectral clustering.
* 监督性的最近邻主要有两个方向 : classification 用于离散label, regression 用于 连续 label

最近邻的原理:
1. Find a predefined number of training samples closest in distance to the new point, then predict the label from these.
2. The number of samples can be a user-defined constant (kNN).
3. Or vary based on the local density of points (radius-based neighbor learning).
4. Distance can be any metric measure
   * standard Euclidean distance is the most common choice.
* 最近邻的一大特性 : non-generalizing, 因为只是简单的记录训练数据
* 最近邻可以转化成其他的基于索引 index 的结构 `Ball Tree, KD Tree`

sklearn.neighbors 的函数可以接受 numpy arrays 或者 scipy.sparse 的matrices 作为训练输入  

### 2.3.1. NN Classification  

* Neighbors-based classification 是一种 instance-based learning or non-generalizing learning.
* Not attempt to construct a general internal model, but simply stores instances of training data.

scikit-learn 有两种 NN 分类器
1. `KNeighborsClassifier` 实现了 KNN.
   * K 的选择很大程度上取决于数据
   * 高的 k 可以提高对噪声的抑制能力, 但会让分类边界变得模糊
2. `RadiusNeighborsClassifier` 实现了 fixed radius r 的最近邻.
   * 更多的用在数据采样不均匀, 特征分布不均匀的数据集上
   * 不适合用在高维特征上
3. 基础的 NNC 使用均匀权重, 即query结果是 simple majority vote of the nearest neighbors.
   * 在一些场景中更适合使用与距离成反比的权重
   * 设置方法的参数 ` weight = 'uniform', weight = 'distance'`

示例代码:  
```py
# 对 iris 数据集进行拟合
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

# 定义分类器, 选择合适的 k 和权重类型
clf = neighbors.KNeighborsClassifier( k , weights = ?)  
clf.fit(X, y)

# 做 2-D 点图画图
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                    np.arange(y_min, y_max, h))
# 预测图上的每个点
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
```

完整函数原型：
```py
 class sklearn.neighbors.KNeighborsClassifier(
   n_neighbors=5, 
   *, weights='uniform', 
   algorithm='auto', 
   leaf_size=30, 
   p=2, 
   metric='minkowski', 
   metric_params=None
   , n_jobs=None, 
   **kwargs)
```

### 2.3.2. NN Regression



## 2.4. Decision Trees

* Decision Trees (DTs) 也是一种 非参数监督学习 ( non-parametric supervised learning method ).
* 



## 2.5. Ensemble Methods 

Ensemble 是组个几个基础模型的预测, 提高算法的泛化性和鲁棒性  
* `import sklearn.ensemble`
* Ensemble 的两个主要分类
  * averaging method : 降低了模型的 variance. 建立多个独立的模型并取之平均
  * Boosting method : 降低了模型的 bias, 基础模型按顺序排列.

### 2.5.1. Voting Classifier

`sklearn.ensemble.VotingClassifier`  
* 组合不同的模型, 使用多数表决进行分类. majority vote or sote vote (average predicted probabilities).
* 用于一组同等程度表现的模型, 用于平衡他们的弱点.

```py
class sklearn.ensemble.VotingClassifier(
  estimators, 
  *, 
  voting='hard',
  weights=None, 
  n_jobs=None, 
  flatten_transform=True, 
  verbose=False)
```

构造函数参数详解:  
* estimators        : 最主要的传入参数, list of (str, estimator) tuples, 注意传入的元素是元组
  * 单个分类器 `('lr', clf1)` 其中 lr 是赋予的名字, 
  * 使用 VC对象的`.fit` 会对所有成员分类器`self.estimators_` 同时进行 `fit`
  * 单个 estimator 可以被设置成 `drop` 代表禁用
* `voting='hard'`   : 投票模式, `hard` or `soft`
  * hard  : 就是多数表决
  * soft  : 将所有分类器得到的所有类的 probability 相加得到最高的值为最终结果
* `flatten_transform=True` : 进一步设置 `soft` 投票模式下 `transform` 方法的返回格式
  * flatten_transform=True, returns matrix with shape (n_classifiers, n_samples * n_classes).
  * flatten_transform=False, it returns (n_classifiers, n_samples, n_classes)
* `n_jobs` : int, default=None
  * 用于加速程序运行, 是否并列晕眩
  * None 代表单处理器
  * -1 代表全速运行, 所有处理器都用
* `weights` : 和分类器个数一样长的 list
  * 用于指明不同分类器的权重
  * 默认为空代表均一

方法详解:  
* `fit(X, y, sample_weight=None)` 拟合数据
  * 可以传入 和 n_samples 相同长度的列表, 用来指明每个样本的权重, 否则均一
* `predict(X)` 预测, 返回和样本长度一样的 array-like
* `transform(X)` 预测, 但是返回的是每个分类器对每个类的预测概率 `probabilities`  

### 2.5.2. Voting Regressor






## 2.6. 监督学习的 Neural Network models


# 3. Unsupervised learning 模型

## 3.1. cluster

### 3.1.1. DBSCAN



### 3.1.2. KMeans

```py
X = digits.values
ground_truth = truth.values

import imp
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

pca = PCA(n_components=30)
Cx = pca.fit_transform(scale(X))

from sklearn.cluster import KMeans
clustering = KMeans(n_clusters=10, n_init=10, random_state=1)

clustering.fit(Cx)

import numpy as np
ms = np.column_stack((ground_truth,clustering.labels_))
df =  pd.DataFrame(ms, columns = ['Ground truth', 'Clusters'])
ctab = pd.crosstab(df['Ground truth'], df['Clusters'], margins=True)





from sklearn.cluster import KMeans
import numpy as np
inertia = list()
for k in range(1, 21):
  clustering = KMeans(n_clusters=k,
                      n_init=10, random_state=1)
  clustering.fit(Cx)
  inertia.append(clustering.inertia_)
delta_inertia = np.diff(inertia) * (-1)

```



# 4. Model selection and evaluation

# 5. Dataset loading utilities

`sklearn.datasets`  
* scikit 也有自己的数据装载包以及样例数据集, 可以快速的测试模型以及方便的从超大型数据库中加载数据.  
* scikit 还可以进行 synthetic 数据的生成



## 5.1. Loading other datasets

提供了一些 miscellaneous tools, 用来加载杂七杂八类型的数据集  

## 5.2. Generated datasets

The dataset generation functions can be used to generate controlled synthetic datasets.  

## 5.3. Toy dataset

* 无需下载的用于快速测试程序的小型数据库
* 不具备什么现实泛化意义

* `return_X_y = False` 如果为真, data 和 target 将会分别返回, 而不是以 `bunch` 形式
* `as_frame` 如果为真, 数据返回的形式将是 pandas.DataFrame 

```py
load_boston(*[, return_X_y])
load_iris(*[, return_X_y, as_frame])
load_diabetes(*[, return_X_y, as_frame])
load_digits(*[, n_class, return_X_y, as_frame])
load_linnerud(*[, return_X_y, as_frame])
load_wine(*[, return_X_y, as_frame])
load_breast_cancer(*[, return_X_y, as_frame])
```

## 5.4. Real world datasets




# 6. decomposition

## 6.1. FactorAnalysis

```py
from sklearn.decomposition import FactorAnalysis


iris = pd.read_csv('iris_dataset.csv')
X = iris.values
cols = iris.columns.tolist()

factor = FactorAnalysis(n_components=4).fit(X)
factor_comp = np.round(factor.components_,3)

print(pd.DataFrame(factor_comp,columns=cols))

print(f'Explained variance by each component:\n {evr}')

print(pd.DataFrame(pca_comp,columns=cols))

```

## 6.2. PCA


```py
homes=pd.read_csv("homes.csv")
X = homes.values
cols = homes.columns.tolist()

import imp
from sklearn.decomposition import PCA

pca = PCA().fit(X)

# 获取每个数据的有效占比  
evr = pca.explained_variance_ratio_

pca_comp = np.round(pca.components_,3)

print("This is the result of the PCA on homes.csv:")
print(pd.DataFrame(pca_comp,columns=cols))

```



# 7. manifold

### 7.0.1. TSNE
```py
import imp
from sklearn.manifold import TSNE
tsne = TSNE(init='pca',
            # Larger datasets usually require a larger perplexity. Consider selecting a value between 5 and 50.
            perplexity=50, 
            # For larger values, the space between natural clusters will be larger in the embedded space. Again, the choice of this parameter is not very critical.
            early_exaggeration=25, 
            # Maximum number of iterations for the optimization. Should be at least 250.
            n_iter=300 
            )

Tx = tsne.fit_transform(X)




import numpy as np
import matplotlib.pyplot as plt
plt.xticks([], [])
plt.yticks([], [])
for target in np.unique(ground_truth):
  selection = ground_truth==target
  X1, X2 = Tx[selection, 0], Tx[selection, 1]
  plt.plot(X1, X2, 'o', ms=3)
  c1, c2 = np.median(X1), np.median(X2)
  plt.text(c1, c2, target, fontsize=18, fontweight='bold')

plt.show()

```

# 8. preprocessing

用于各种预处理数据, 各个方法之间比较独立

includes scaling, centering, normalization, binarization methods



## 8.1. scale

```py
digits = pd.read_csv('digits.csv')
truth = pd.read_csv('ground_truth.csv')

X = digits.values
ground_truth = truth.values

import imp
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

pca = PCA(n_components=30)
Cx = pca.fit_transform(scale(X))
evr = pca.explained_variance_ratio_

print(f'Explained variance {evr}')

```



# 9. collections

