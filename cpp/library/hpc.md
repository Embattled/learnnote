# 1. Hign Performance Compute



**Embarrassingly Parallel** : 完全并行问题, 子问题之间没有任何的冲突  

**Cyclic(or interleaved) allocation** : 隔行分配, 如果有p个处理器, 则任务k被分配到 (k mod p) 号处理器执行


# 2. MPI programming

Message Passing Interface (MPI): parallel programs with *message passing*  

* Multiple programs (MPI processes) run on a parallel computer.
* Each MPI process has its own memory space.
* MPI processes 互相之间可以传输数据
* MPI 只定义了接口, 没有实现



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

### 2.2.3. p2p 通信

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

### 2.2.4. Benchmarking 函数

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
### 2.2.5. MPI_Barrier 同步函数

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


### 2.2.6. 其他的通信方法

* 将每个线程的相同名称的变量进行处理
  * Reduce      : A single process finally has the result of reduction op.
  * Allreduce   : Every process finally has the result of reduction op.


* 某个数据的各个部分分散在各个进程中, 将他们收集回来
  * Gather      : Global communication for a single process to collect data items distributed among others.
  * Allgather   : Global communication for every process to collect data items distributed among others.


* 发送数据给其他线程
  * Broadcast   : One MPI process sends the **same data** to the others.
  * Scatter     : One MPI process sends **different data** to each of the others.


# 3. OpenMP programming

OpenMP is used to specify how to execute a block.   

* process : When a program is launched, OS reserves a set of some computing resources for the execution.
  * CPU time
  * Memory space
  * File descriptors
* thread  : A thread is created inside a process, an execution flow
  * CPU time is assigned to each thread
  * The other resources are shared with the other threads of the process.

## 3.1. OpenMP 编译和运行

* Environmental Variable OMP_NUM_THREADS
```shell

# configure the number of threads for running the program
# B shell
export OMP_NUM_THREADS=4
# C shell
set OMP_NUM_THREADS 4
```

compile:  
`g++ -fopenmp sample.c`   

## 3.2. scalability

* A small part of the code consumes a large part of the execution time.
* Ts: Single Execution Time
* Tp: Parallel Execution Time

ratio=Ts/Tp

## 3.3. OpenMP 编程基础

### 3.3.1. Directives 命令格式

Compiler Directive的基本格式如下：  

`#pragma omp directive-name  [clause [ [, clause]...]`


* 旧版的OpenMP中, directive-name共11个:
* 有 clause 选项的: `parallel, for, sections, single`
* 无 clause 选项的: `atomic, barrier, critical, flush, master, ordered, threadprivate`

* clause（子句） 相当于是Directive的修饰，定义一些Directive的参数什么的
  * `copyin(variable-list),`
  * `copyprivate(variable-list),`
  * `default(shared | none)`
  * `firstprivate(variable-list)`
  * `if(expression)`                          : 为 clause 的实效添加表达式
  * `lastprivate(variable-list)`
  * `nowait`
  * `num_threads(num)`
  * `ordered, private(variable-list)`
  * `reduction(operation: variable-list)`
  * `schedule(type[,size])`                   : 设置C++ for的多次迭代如何在多个线程间划分
  * `shared(variable-list)`

* 4.0 的OpenMP中的新 directive-name:
* 有 clause 的:
  * simd 三命令:
    * `simd`  : Applied to a loop to indicate that the loop 可以被转换成 simd loop
    * `declare simd` : 和simd 配套的 , 不懂
    * `loop simd` : 不懂
  * target 三命令:
    * `target` : 指定执行代码的目标设备
    * `target update`
    * `declare target`
  * `teams` : 创建一个线程 league
  * `distribute` 
  * 太多了不看了


### 3.3.2. GPU Programming with OpenMP


Target Directives
* `#pragma omp target`
* `#pragma omp target data` (Defining only data mapping)
  * New features of OpenMP 4.0 or later
  * The block specified by the target directive can be offloaded to another processor such as GPU.
  * A map clause is used to send/retrieve data to/from the GPU.
* `#pragma omp target teams distribute` : Create a league of teams
* `#pragma omp parallel`                : Creates a team of (synchronizable) threads.

```cpp

/* Aray Anew is created, and A is transferred from/to GPU */
#pragma omp target map(alloc:Anew) map(tofrom:A)
{
  #pragma omp parallel
  for(i=;i<N;i++){
    /* time-consuming data-parallel computation */
  }
}


#pragma omp target data map (alloc:Anew) map(tofrom:A)
{
  #pragma omp target teams distribute
  for(j=0;j<M;j++){
    #pragma omp parallel
    for(i=;i<N;i++){
      /* time-consuming data-parallel computation */
    }
  }
}
```








## 3.4. directive 

### 3.4.1. parallel 有 clause


OpenMP 以 block 为单位并行运算, 即每个Compiler Directive 只作用于其后的语句, 所以用{}来代表一个复合语句    

Work-sharing by inserting directives into a sequential code.  

`#pragma omp parallel` 表示其后语句将被多个线程并行执行，线程个数由系统预设,一般等于逻辑处理器个数;  
`#pragma omp parallel num_threads(4)`  指定线程数为4  

最基础的语句  


### 3.4.2. for  有 clause

C++ for循环需要一些限制从而能在执行C++ for之前确定循环次数，例如C++ for中不应含有break等。OpenMP for作用于其后的第一层C++ for循环  


```cpp
// C code
for(i=0;i<10;i++)
  a[i] = b[i]*f + c[i];


// OpenMP code

#pragma omp parallel
{
#pragma omp for
  for(i=0;i<10;i++)
    a[i] = b[i]*f + c[i];
  // 01234 在 cpu0 ，56789 在cpu1 , 顺序分配
}

// parallel region中只包含一个for directive作用的语句. 此时可以将parallel和for “缩写”
#pragma omp parallel for
for(int i=0; i<size; ++i)
    data[i] = 123;

```
### 3.4.3. section  有 clause

如果说for directive用作数据并行，那么sections directive用于任务并行，它指示后面的代码块包含将被多个线程并行执行的section块  

```cpp

#pragma omp parallel
{
  // 如果一个 block 中有许多个可以并行的区块
  #pragma omp sections
  {
    #pragma omp section
    {
      // 代码 A
    }
    #pragma omp section
    {
      // 代码 B
    }
  //  A 和 B 将会并行执行  
  }
}

```

### 3.4.4. single 有 clause

指示代码将仅被一个线程执行,即只执行一次, 具体是哪个线程不确定

```cpp
#pragma omp parallel num_threads(4)
{
    #pragma omp single
    std::cout << omp_get_thread_num();
    std::cout << "-";
}
// 输出 0---- 一个0 和四个- 
```

### 3.4.5. master

指示代码将仅被主线程执行，功能类似于single directive，但single directive时具体是哪个线程不确定（有可能是当时闲的那个）。

### 3.4.6. critical

定义一个临界区，保证同一时刻只有一个线程访问临界区  
```cpp

// Each thread holds a unique value in tmp.
#pragma omp parallel private (tmp)
{
  tmp = 0;
  #pragma omp for
  for(i=0;i<10;i++)
    if(tmp < a[i]) tmp = a[i];


  // Once a thread enters the critical section, the others cannot enter it.
  // 防止同时访问max
  #pragma omp critical
  {
    if(max<tmp) max = tmp;
  }
}
```

### 3.4.7. barrier 

定义一个同步，所有线程都执行到该行后，所有线程才继续执行后面的代码  


```cpp

#pragma omp parallel num_threads(3)
{
    #pragma omp critical
    std::cout << omp_get_thread_num() << " ";
    #pragma omp barrier
    #pragma omp critical
    std::cout << omp_get_thread_num()+10 << " ";

    // 输出 0 2 1 10 12 11   一位数数字打印完了才开始打印两位数数字
}

```
## 3.5. clauses 

### 3.5.1. if

可以用if clause条件地进行并行化，用num_threads clause覆盖默认线程数  


```cpp

int a = 0;
#pragma omp parallel if(a) num_threads(6)
{
    // 因为 a 为0 , 所以没有指定多线程
    std::cout << omp_get_thread_num();
}
// 输出为0



int a = 7;
#pragma omp parallel if(a) num_threads(6)
{
  // 6 个线程执行
    std::cout << omp_get_thread_num();
}
// 输出 0512346



```

### 3.5.2. reduction 

reduction clause用于归约，如下是一个并行求和的例子：

```cpp

int sum=0;
std::cout << omp_get_thread_num() << ":" << &sum << std::endl << std::endl;

#pragma omp parallel num_threads(8) reduction(+:sum)
{
    #pragma omp critical

    std::cout << omp_get_thread_num() << ":" << &sum << std::endl;

    #pragma omp for
    for(int i=1; i<=10000; ++i){
        // 每个线程用自己的sum求一部分和
        sum += i;
    }
}
// 最后将所有线程的私有sum加起来赋值给共享版本的sum

std::cout << "sum's valuse: " << sum << std::endl; 
// 50005000
```

### 3.5.3. Clauses and Critical Section

Threads share a single memory space.  All data are shared by threads by default.  

To create private variables, use clasuses
* shared (default)
  * The variables are shared by threads
* private
  * private/firstprivate/lastprivate/copyprivate
  * The variables can have different values for different threads.
  * The value can be copied from/to the outside of the block.

### nowait


* Each thread does not wait for the others at the end of the block.




### 3.5.4. 主要block命令总结

* `#pragma omp parallel`    : directs parallelizing the following block
* `#pragma omp for`         : directs work-sharing the following for loop
* `#pragma omp sections`    : directs the following block is a set of parallel sections
* `#pragma omp section`     : directs the following block is a section
* `#pragma omp critical`    : directs the following block is a critical section. Just one thread can enter it

