
# 1. Torch Distributed

torch.distributed 的几个子分支
* torch.distributed
* torch.distributed.algorithms.join
* torch.distributed.elastic
* torch.distributed.fsdp
* torch.distributed.optim
* torch.distributed.tensor.parallel
* torch.distributed.checkpoint
* torch.distributions

## 1.1. Torch Distributed Elastic 

Torch 弹性分布式

分布式的, 具有弹性, 容错性的任务

主要用于大规模集群学习?

### 1.1.1. Torch Distributed Elastic Quickstart
<!-- 完 -->
启动一个 job, 在命令行直接执行命令    

```sh
# launch a fault-tolerant job
torchrun
   --nnodes=NUM_NODES
   --nproc-per-node=TRAINERS_PER_NODE
   --max-restarts=NUM_ALLOWED_FAILURES
   --rdzv-id=JOB_ID
   --rdzv-backend=c10d
   --rdzv-endpoint=HOST_NODE_ADDR
   YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)

# elastic job,   on at least MIN_SIZE nodes and at most MAX_SIZE nodes
torchrun
    --nnodes=MIN_SIZE:MAX_SIZE
    --nproc-per-node=TRAINERS_PER_NODE
    --max-restarts=NUM_ALLOWED_FAILURES_OR_MEMBERSHIP_CHANGES
    --rdzv-id=JOB_ID
    --rdzv-backend=c10d
    --rdzv-endpoint=HOST_NODE_ADDR
    YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)
```

参数说明:指定
* `--rdzv-endpoint=HOST_NODE_ADDR`      : 指定为 `<host>[:<port>]` 默认值 29400
  * 学习任务的后端实例化和托管的节点, 类似于学习任务的头目
  * 可以指定为集群中的任意节点, 但理想情况下使用 高带宽的节点
* `--standalone` : 启动带有 `a sidecar rendezvous backend` 的单任务节点
  * 使用群组训练接口, 但只在单设备上运行?
  * 这种情况下 不需要传递 --rdzv-id, --rdzv-endpoint, and --rdzv-backend


### 1.1.2. torch.distributed.elastic API


#### 1.1.2.1. torchrun (elastic launch)

弹性启动器

是一个 `torch.distributed.launch` 的超集 Superset, 提供了以下的额外功能:
* 通过重启所有的 workers 来处理 worker failures
* worker 的 RANK 和 WORLD_SIZE 会自动赋值
* 节点数可以弹性的改变  between minimum and maximum

torchrun 本身是一个控制台脚本, 源于 `torch.distributed.run`, 


