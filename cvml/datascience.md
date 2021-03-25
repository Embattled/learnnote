# Data Science Solutions 

##　Workflow

工作目标：
* Classifying : 明确 samples 的类别, 可能从中得到classes与目标解决方案的隐含关联
* Correlating : 明确 features 之间, features 和 goal 之间的相关性, 为后面的处理做帮助
* Converting  : 根据选择的模型对数据进行预处理
* Completing  : 在 Converting 时可能需要对数据的 Missing Value 进行处理
* Correcting  : 检查 Training set 的 outliers 删掉无用的 Feature
* Creating    : 创建新 Feature, 基于 Correlation, Conversion, Completeness
* Charting    : 可视化数据.


### 分析数据

1. 确认dataset 中的可用 feature
2. 确认features的类型, categorical or numerical or mixed 
   * 确认 distribution
   * 确认和目标 label 的 correlation, 统计或者作图
3. 修复数据
   * 确认哪些 features 可能会有 errors or `typos` 打印错误
   * 确认是否有 missing value