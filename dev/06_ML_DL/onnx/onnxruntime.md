# onnxruntime

onnx runtime 提供了在 CPU 或者 GPU 上运行机器学习模型的简单方法, 而不依赖于训练框架.  
因为机器学习框架通常更多的针对 batch training 进行优化, 而不是 prediction.  

https://onnxruntime.ai/getting-started

ONNX Runtime 是由微软维护的一个跨平台机器学习推理加速器, 即实际进行推理的 推理引擎 
* ONNX Runtime 是直接对接 ONNX 的
* ONNX Runtime 可以直接读取并运行 .onnx 文件, 不需要另外的转换
* 对于 PyTorch - ONNX - ONNX Runtime 这条部署流水线, 得到 .onnx 文件 并通过  ONNX Runtime 上运行模型, 模型部署就算大功告成了

对于模型的推论:
* onnx.reference 是 onnx 内部自带的用于测试的推论接口
* onnxruntime 是一个独立的, 可以直接对接 onnx 的高速推论引擎
* 因为这二者的概念不同, 所以 onnxruntime 放于别的章节里
  * 官方将推理架构和推理运算执行器进行了分离, 称实际进行计算的设备为 Execution Providers
  * 若要进行 GPU 推理, 需要独立安装对应的 Providers `onnxruntime-gpu`
  * 同理还有 `onnxruntime-openvino` `onnxruntime-directml`



# python API

onnx implements a python runtime useful to help understand a model. It is not intended to be used for production and performance is not a goal.


## InferenceSession

