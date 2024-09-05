# PyTorch Recipes

Recipes are bite-sized, actionable examples of how to use specific PyTorch features, different from our full-length tutorials.

细碎的教学


# PyTorch Profiler

使用 `torch.profiler` 来对模型进行各种性能使用量分析

profile 通常按照 context manager 来进行使用, 参数有
* `activities` : 要记录的活动
  * ProfilerActivity.CPU : 除了 CUDA 以外的所有运算
  * ProfilerActivity.CUDA
* `record_shapes ` : 是否记录各种运算的输入 shape
* `profile_memory ` : 是否记录 model's Tensors 消耗的内存量
  


```py
from torch.profiler import profile, record_function, ProfilerActivity
model = models.resnet18()
inputs = torch.randn(5, 3, 224, 224)


with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("model_inference"):
        model(inputs)

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
```