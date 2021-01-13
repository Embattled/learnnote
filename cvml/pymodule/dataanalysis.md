# 1. seaborn 数据可视化包

## 1.1. density plot AND Histogram 


A density plot is a smoothed, continuous version of a histogram estimated from the data.   


```py

sns.distplot(flights['arr_delay'], hist=True, kde=False, bins=int(180/5), color = 'blue',hist_kws={'edgecolor':'black'})
sns.distplot(flights['arr_delay'], hist=True, kde=True, bins=int(180/5), color = 'darkblue', hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 4})

```

## 1.2. swarmplot 

Draw a categorical scatterplot with non-overlapping points.  
This function is similar to stripplot(), but the points are adjusted (only along the categorical axis) so that they don’t overlap.   
This gives a better representation of the `distribution of values` , but it does `not scale well to large numbers of observations` .   
This style of plot is sometimes called a “beeswarm”.  


# 2. scipy

Scipy是一个用于数学、科学、工程领域的常用软件包，可以处理插值、积分、优化、图像处理、常微分方程数值解的求解、信号处理等问题。  
它用于有效计算Numpy矩阵，使Numpy和Scipy协同工作，高效解决问题。  

| 模块名            | 应用领域           |
| ----------------- | ------------------ |
| scipy.cluster     | 向量计算/Kmeans    |
| scipy.constants   | 物理和数学常量     |
| scipy.fftpack     | 傅立叶变换         |
| scipy.integrate   | 积分程序           |
| scipy.interpolate | 插值               |
| scipy.io          | 数据输入输出       |
| scipy.linalg      | 线性代数程序       |
| scipy.ndimage     | n维图像包          |
| scipy.odr         | 正交距离回归       |
| scipy.optimize    | 优化               |
| scipy.signal      | 信号处理           |
| scipy.sparse      | 稀疏矩阵           |
| scipy.spatial     | 空间数据结构和算法 |
| scipy.special     | 一些特殊的数学函数 |
| scipy.stats       | 统计               |


## 2.1. scipy.stats  统计包
 
### 2.1.1. 偏度（skewness） 和 峰度（peakedness；kurtosis）又称峰态系数
* 偏度（skewness）
  * 是统计数据分布偏斜方向和程度的度量，是统计数据分布非对称程度的数字特征。  
  * 偏度定义中包括正态分布（偏度=0），右偏分布（也叫正偏分布，其偏度>0），左偏分布（也叫负偏分布，其偏度<0）
* 峰度（peakedness；kurtosis）又称峰态系数
  * 表征概率密度分布曲线在平均值处峰值高低的特征数。
  * 直观看来，峰度反映了峰部的尖度。随机变量的峰度计算方法为：随机变量的四阶中心矩与方差平方的比值。
  *  If the kurtosis is greater than 3, then the dataset has heavier tails than a normal distribution (more in the tails).
  *  If the kurtosis is less than 3, then the dataset has lighter tails than a normal distribution (less in the tails). 

```py
variable = player_table['Height(inches)']

# 使用scipy计算偏度（skewness）
s = skew(variable)
print(f'Skewness {s}')

# 使用pandas计算
s=variable.skew()
print(f'Skewness {s}')

# 计算 Z 和 P
zscore, pvalue = skewtest(variable)
print(f'z-score {zscore}')
print(f'p-value {pvalue}')


# 使用pandas计算峰度
k = kurtosis(variable)
zscore, pvalue = kurtosistest(variable)

print(f'Kurtosis {k}')
print(f'z-score {zscore}')
print(f'p-value {pvalue}')

```


###  2.1.2. T-Test

```py

from scipy.stats import ttest_ind

group1 = player_table["Position"] == 'Catcher'
group2 = player_table["Position"] == 'Outfielder'
variable = player_table["Height(inches)"]

t, pvalue = ttest_ind(variable[group1], variable[group2],axis=0, equal_var=False)

print(f"t statistic {t}")
print(f"p-value {pvalue}")

```

### 2.1.3. ANOVA

```py
from scipy.stats import f_oneway

group1 = player_table["Position"] == 'Catcher'
group2 = player_table["Position"] == 'Outfielder'
group3 = player_table["Position"] == 'Shortstop'
group4 = player_table["Position"] == 'Starting Pitcher'

variable = player_table["Height(inches)"]

f, pvalue = f_oneway(variable[group1],variable[group2],variable[group3],variable[group4])

print(f"One-way ANOVA F-value {f}")
print(f"p-value {pvalue}")

```

### 2.1.4. from  chi-square statistic

scipy.stats import chi2_contingency

```py

pcts = [0, .25, .5, .75, 1]
players_binned = pd.concat(
  [pd.qcut(player_table.iloc[:,3], pcts, precision=1),
  pd.qcut(player_table.iloc[:,4], pcts, precision=1),
  pd.qcut(player_table.iloc[:,5], pcts, precision=1)],
  join='outer', axis=1)


from scipy.stats import chi2_contingency

table = pd.crosstab(player_table['Position'], players_binned['Height(inches)'])
print(table)

chi2, p, dof, expected = chi2_contingency(table.values)
print(f'Chi-square {chi2}   p-value {p}') 


```

### 2.1.5. pearson
a = player_table["Height(inches)"]
b = player_table["Weight(lbs)"]
#print(a)
rho_coef, rho_p = spearmanr(a, b)
r_coef, r_p = pearsonr(a, b)

print(f'Pearson r: {r_coef}')
print(f'Spearman r: {rho_coef}')


# 3. scikit-learn 

`import sklearn `

## 3.1. decomposition

### 3.1.1. FactorAnalysis

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

### 3.1.2. PCA


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



## 3.2. manifold

### 3.2.1. TSNE
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

## 3.3. preprocessing

### 3.3.1. scale

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

## 3.4. cluster

### 3.4.1. DBSCAN



### 3.4.2. KMeans

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


## 3.5. collections

