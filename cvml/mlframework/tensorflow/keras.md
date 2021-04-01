# Keras

* 高级的神经网络 API, 开发重点就是支持快速的实验
* 可以使用 TensorFlow, CNTK, 或者 Theano 作为后端运行



# model

* Keras的核心数据结构 
* 组织网络层的方式
  

## Sequential 顺序模型

* 顺序模型即 多个网络层线性堆叠
* 建立好对象后, 使用 `.add()` 来堆叠模型

```py
from keras.models import Sequential
model = Sequential()

from keras.layers import Dense

model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))
```


## 配置学习

在完成了模型的构建后, 可以使用 .compile() 来配置学习过程：
```py
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

# 还可以对优化器进行详细的配置
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))
```

## 迭代训练训练


```py
# x_train 和 y_train 是 Numpy 数组 -- 就像在 Scikit-Learn API 中一样。
model.fit(x_train, y_train, epochs=5, batch_size=32)

# 手动训练一个批次
model.train_on_batch(x_batch, y_batch)
```


## 评估与预测

```py
# 评估模型性能
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)

# 对新的数据生成预测
classes = model.predict(x_test, batch_size=128)
```