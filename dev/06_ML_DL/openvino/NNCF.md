# Neural Network Compression Framework (NNCF)

NNCF provides a suite of advanced algorithms for Neural Networks inference optimization in OpenVINO™ with minimal accuracy drop.
* 提供了一套   `post-training` 和 `training-time` 的算法, 用于 OpenVINO 中的神经网路推理优化
* 其中 OpenVINO 的 taining-time 优化必须要通过 NNCF 来实现.  

NNCF is designed to work with models from PyTorch, TensorFlow, ONNX and OpenVINO™.

整个组件被整合成了一个独立于 OpenVINO 环境 python 的包, 可以方便的为 Pytorch 和 TensorFlow 添加不同的压缩算法.  
注意, 通过 NNCF 进行优化的模型不再属于原本的框架, 而是属于 OpenVINO IR 的表达, 要进行推论的话则需要使用 OpenVINO Runtime.  

NNCF 的官方推荐工作流:
1. 优先进行  `Post-training quantization` , 并进行性能比较
2. 如果性能损失过大, 进行 `Quantization-aware training` 来提高精度同时保持优化后的推论速度
3. 如果量化后的模型仍然速度不够快, 进行 `Filter Pruning` 来进一步删减模型的参数来提高模型的速度.  

