# Pytorch的建模流程

Pytorch实现神经网络模型的一般流程包括：
1. 准备数据
2. 定义模型
3. 训练模型
4. 评估模型
5. 使用模型
6. 保存模型

其中对于新手, 准备数据是最困难的过程  

实践中通常会遇到的数据类型包括结构化数据，图片数据，文本数据，时间序列数  

## 结构化数据建模流程范例


```py
import os
import datetime

#打印时间
def printbar():
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n"+"=========="*8 + "%s"%nowtime)

#mac系统上pytorch和matplotlib在jupyter中同时跑需要更改环境变量
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" 
```