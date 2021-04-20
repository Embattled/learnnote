- [1. scikit-learn introduction](#1-scikit-learn-introduction)
- [2. Supervised learning 模型](#2-supervised-learning-模型)
  - [2.1. Ensemble Methods](#21-ensemble-methods)
- [3. Unsupervised learning 模型](#3-unsupervised-learning-模型)
- [4. Model selection and evaluation](#4-model-selection-and-evaluation)
- [5. decomposition](#5-decomposition)
  - [5.1. FactorAnalysis](#51-factoranalysis)
  - [5.2. PCA](#52-pca)
- [6. manifold](#6-manifold)
    - [6.0.1. TSNE](#601-tsne)
- [7. preprocessing](#7-preprocessing)
  - [7.1. scale](#71-scale)
- [8. cluster](#8-cluster)
  - [8.1. DBSCAN](#81-dbscan)
  - [8.2. KMeans](#82-kmeans)
- [9. collections](#9-collections)
# 1. scikit-learn introduction

* 目前非常有名的开业机器学习库
* 导入 : `import sklearn`


# 2. Supervised learning 模型

## 2.1. Ensemble Methods

* `import sklearn.ensemble`


# 3. Unsupervised learning 模型


# 4. Model selection and evaluation
# 5. decomposition

## 5.1. FactorAnalysis

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

## 5.2. PCA


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



# 6. manifold

### 6.0.1. TSNE
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

# 7. preprocessing

## 7.1. scale

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

# 8. cluster

## 8.1. DBSCAN



## 8.2. KMeans

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


# 9. collections

