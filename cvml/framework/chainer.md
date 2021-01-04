# 1. chainer

Chainer is a rapidly growing neural network platform.

## 1.1. Trainer

训练器用来设置神经网络以及训练数据, 多级结构如下:

* Trainer
  * Updater
    * Iterator
      * Dataset
    * Optimizer
      * Model
  * Extensions


## 1.2. 模型搭建的步骤

根据Trainer的结构, 从底层开始向上构建

### 1.2.1. Dataset

绑定数据集

```py
# chainer.datasets
from chainer import datasets


# 特征
X = data_array[:, 1:].astype(np.float32)
# 标签
Y = data_array[:, 0].astype(np.int32)[:, None]
# 绑定到datasets.TupleDataset 并分割成 train test
train, test = datasets.split_dataset_random(
    datasets.TupleDataset(X, Y), int(data_array.shape[0] * .7))
```

### 1.2.2. Iterator

```py
# 参数是 数据集和 Batch Size
train_iter = ch.iterators.SerialIterator(train, 100)

# test 迭代器 不重复 不打乱
test_iter = ch.iterators.SerialIterator(
    test, 100, repeat=False, shuffle=False)
```

### 1.2.3. Model

```py
# Network definition
# 用一个函数来生成 MLP model 指定输入维度和输出维度  
def MLP(n_units, n_out):
    layer = ch.Sequential(L.Linear(n_units), F.relu)
    model = layer.repeat(2)
    model.append(L.Linear(n_out))

    return model

# 建立二元分类器
model = L.Classifier(
    MLP(44, 1), lossfun=F.sigmoid_cross_entropy, accfun=F.binary_accuracy)
```

### 1.2.4. Optimizer

将模型封装到优化器  

```py
# Setup an optimizer
optimizer = ch.optimizers.SGD().setup(model)
```

### 1.2.5. Updater

* Setting the device=-1 sets the device as the CPU. 
* To GPU, set device equal to the number of the GPU, usually device=0

将 Iterator 和 Optimizer 绑定起来
```py
# Create the updater, using the optimizer
updater = training.StandardUpdater(train_iter, optimizer, device=-1)

	
# 最终将 Updater 设置成 Trainer
# Set up a trainer
trainer = training.Trainer(updater, (50, 'epoch'), out='result')
```

### 1.2.6. Extensions

用于给 Trainer 添加附加操作  
通过方法 `trainer.extend()`  

```py
	
# Evaluate the model with the test dataset for each epoch
# 给 Trainer 附加每个 epoch 的测试
trainer.extend(extensions.Evaluator(test_iter, model, device=-1))
```

### 运行

模型搭建好后, 即可很轻松的运行程序  
```py
#  Run the training
trainer.run()


```