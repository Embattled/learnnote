# OpenMMLab

上海人工智能实验室管理维护的 视觉 AI 开放库

是一个基于 Pytorch 的, 以快速上手和产品落地为核心目的 高级 API


系统阶层
* Deployment 产品部署         : MMDelopy
* Computer Vision Libraries: 30+ 计算机视觉库
* Foundational Libraries    : 基础架构库  MMCV  MMEngine
* Deep Learning Framework   : 整个系统基于 Pytorch 


# MMEngine  

基于 Pytorch 的封装 深度学习基础库

支持的功能有
* Integrate mainstream large-scale model training frameworks        : 集成主流的 大规模模型训练框架
  * ColossalAI
  * DeepSpeed
  * FSDP
* Supports a variety of training strategies : 多种训练策略
  * Mixed Precision Training
  * Gradient Accumulation
  * Gradient Checkpointing
* Provides a user-friendly configuration system
  * Pure Python-style configuration files, easy to navigate
  * Plain-text-style configuration files, supporting JSON and YAML
* Covers mainstream training monitoring platforms
  * TensorBoard | WandB | MLflow
  * ClearML | Neptune | DVCLive | Aim
