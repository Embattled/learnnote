# 1. Hign Performance Compute

# 2. MPI programming

Message Passing Interface (MPI): parallel programs with *message passing*  

* Multiple programs (MPI processes) run on a parallel computer.
* Each MPI process has its own memory space.
* MPI processes 互相之间可以传输数据
* MPI 只定义了接口, 没有实现


**Embarrassingly Parallel** : 完全并行问题, 子问题之间没有任何的冲突  

**Cyclic(or interleaved) allocation** : 隔行分配, 如果有p个处理器, 则任务k被分配到 (k mod p) 号处理器执行

**SPMD - Single Program Multiple Data**
* 一个程序通过多个MPI进程来执行, 每个MPI进程有自己的ID, 称作 `MPI rank`
  * `MPI rank` 用来操作不同的进程执行于不同的运算
  * `if (MyMPIrank==0)/* do something*/  else /*do something*/`
* MPI进程通过 `communicators` 来分组和进行数据传输
  * `MPI_COMM_WORLD` 就是一个全局的 communicator


## 2.1. MPI 的编译和执行

```shell

# 总体上, 编译命令 mpicc 和 gcc 基本类似
# -o 指定输出可执行文件的名称
mpicc –o sat1 sat1.c


# mpi 程序的执行也要通过相应的命令执行
# -np 指定执行时候的 node
mpirun -np 1 ./sat1
mpirun -np 4 ./sat1

```

## 2.2. MPI 库基础  

头文件是 mpi.h  
任何MPI关键字的格式都是 `MPI_AAA_BBB`  
任何MPI函数名的格式都是 `MPI_Aaa_Bbb`  


### 2.2.1. 基础操作

处理器数和各个进程的ID都是在程序外决定的, 在运行时候指定的命令决定的

```cpp
#include <mpi.h>
//-----main
int id,p;

int sum;

// 初始化
MPI_Init(&argc, &argv);

// 获取在一个 communicator 中有多少个处理器
// 注意 MPI_COMM_WORLD 是一个全局的 communicator , 管理了所有的 processes
MPI_Comm_size(MPI_COMM_WORLD, &p);

// 获取一个进程的 rank (即ID)
MPI_Comm_rank(MPI_COMM_WORLD, &id);

// 并行跳跃循环 interleaved 分配 task
for(int i=id;i< num_of_tasks ;i+=p)
{
    sum+=process();
    // taskprocess
}

// 此时算出的 sum 是各个进程的 subsum
fprintf(stderr, ”Process %d is done. Subtotal=%d¥n”, id, s);

// 结束清空 cleanup
MPI_Finalize();

```

### 2.2.2. 进程间通信 Reduce

用于将各个进程的数据整合到一起

```cpp

int gather;

for(int i=id;i< num_of_tasks ;i+=p)
{
    subsum+=process();
    // taskprocess
}

// MPI 数据 Reduce  用于收集所有进程的某一数据
// &subsum 本地数据,每个进程都有的数据
// &gather 要收集存储到的最终的变量
// &1      Data count
// MPI_INT Data type
// MPI_SUM 对各个进程的数据作何收集处理, 这里是相加
// 0      *重要, 指定gather的数据存储在哪一个进程里
// MPI_COMM_WORLD 指定reduce处理作用的组, 这里是全局全部线程
MPI_Reduce(&subsum,&gather,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
// 这种方法的通信次数是 p-1次, 可以通过划分组来减少总的通信次数




//只通过 rand=0 的进程输出信息
if(id==0)
    fprintf(stderr, ”Process %d is done. Total=%d¥n”, id, g);

MPI_Finalize();
```

数据关键字对照表   
| Name               | CType          |
| ------------------ | -------------- |
| MPI_CHAR           | signed char    |
| MPI_DOUBLE         | double         |
| MPI_FLOAT          | float          |
| MPI_INT            | int            |
| MPI_LONG           | long           |
| MPI_LONG_DOUBLE    | long double    |
| MPI_SHORT          | short          |
| MPI_USIGNED_CHAR   | unsigned char  |
| MPI_UNSIGNED       | unsignedint    |
| MPI_UNSIGNED_LONG  | unsigned long  |
| MPI_UNSIGNED_SHORT | unsigned short |


Reduction Operator
| Name       | Meaning                   |
| ---------- | ------------------------- |
| MPI_BAND   | Bitwise AND               |
| MPI_BOR    | Bitwise OR                |
| MPI_BXOR   | BitwiseeXclusiveOR (XOR)  |
| MPI_LAND   | LogicalAND                |
| MPI_LOR    | Logical OR                |
| MPI_LXOR   | Logical eXclusiveOR (XOR) |
| MPI_MAX    | Maximum                   |
| MPI_MAXLOC | Maximum and itslocation   |
| MPI_MIN    | Minimum                   |
| MPI_MINLOC | Minimum and its location  |
| MPI_PROD   | Product                   |
| MPI_SUM    | Sum                       |

### p2p 通信

最基础的线程对线程通信

阻塞通信函数, 当信息通信完成后函数返回
  * MPI_Send : 发送一个数据给另一个线程
  * MPI_Recv : 接受一个数据从另一个线程
* 非阻塞通信
  * MPI_Isend
  * MPI_Irecv

函数格式
```cpp

// &j : 用于接收数据的变量地址
// 1  : 传输的数据个数
// MPI_INT : 数据类型
// dst/src  : 发送方或者接收方进程的rank
// 0  : 标签, 用于在一次通信中传输多个数据的时候区分不同数据
// MPI_COMM_WORLD  : Communicator
// &status : 状态

// 注意编写程序的时候不要让所有进程都先执行Send
// 因为只有 接收方接受完数据后, Send 函数才会返回
MPI_Send(&j,1,MPI_INT,dst,0,MPI_COMM_WORLD,&status);

MPI_Recv(&j,1,MPI_INT,src,0,MPI_COMM_WORLD,&status);

```

### Benchmarking 函数

* MPI_Wtime   用于返回执行时间, 一般通过执行两次然后计算差来代表运行时间
* MPI_Wtick   返回 MPI_Wtick 的精度

```c
double etime;

// 获得第一次时间 直接用负号
etime=-MPI_Wtime(); 

// some task

// 计算差值
etime+=MPI_Wtime();

```
### MPI_Barrier 同步函数

Barrier synchronization  
只有同一组的所有进程都执行到该函数时, 才会返回, 用于同步进程的进度

```c
double etime;

// 先阻塞所有进程 ,确保同步开始计时
MPI_Barrier(MPI_COMM_WORLD);
etime=-MPI_Wtime(); 

// some task

// 同步后再计算差值, 确保获得最完整的计算时间
MPI_Barrier(MPI_COMM_WORLD);
etime+=MPI_Wtime();


```


### 其他的通信方法

* 将每个线程的相同名称的变量进行处理
  * Reduce      : A single process finally has the result of reduction op.
  * Allreduce   : Every process finally has the result of reduction op.


* 某个数据的各个部分分散在各个进程中, 将他们收集回来
  * Gather      : Global communication for a single process to collect data items distributed among others.
  * Allgather   : Global communication for every process to collect data items distributed among others.


* 发送数据给其他线程
  * Broadcast   : One MPI process sends the **same data** to the others.
  * Scatter     : One MPI process sends **different data** to each of the others.