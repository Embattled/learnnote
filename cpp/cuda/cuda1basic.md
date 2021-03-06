# 1. cuda 的基础

1. 在GPU并行的术语,把一个计算任务以原来N倍速运行称为 Nx加速
2. cuda使用的是单指令多线程(SIMT)的并行模式
3. CUDA中,一个核包含一个逻辑计算单元(ALU)和一个浮点计算单元(FPU)。多个核集成在一起被称为多流处理器(SM)。
4. 将一个计算任务分解成多个子任务,成为线程。多个线程组织为线程块。线程块被分解为大小与一个SM中核数量相同(32)的线程束(warp)。多流处理器的控制单元指挥所有的核同时执行同一个指令,每个核使用由CUDA提供的单独的索引值来进行不同的运算。
5. 当一个线程束所需要分数据不可获得时,多流处理器会转向执行另一个可获得数据的线程束。这种方式叫隐藏延迟。所关注的是整体的运算吞吐量而不是单个核心的整体速度。
6. CUDA关键的软件结构是核函数(kernel),该函数产生大量的组织成可以分配给多流处理器的计算线程。CUDA的索引变量会替换串行代码里的循环索引。

## 在vs中使用cuda

打开普通项目: 项目->生成依赖项-> 生成自定义 对话框中选中对应的cuda版本

# 2. cuda API 和 C 语言扩展
## 1. 核函数的加载

* 加载核函数从函数名开始,比如 `aKernel`,以一个包含了以逗号分开的参数列表的括号结尾,同C语言
* `aKernel<<<Dg,Db>>>(args)` 在三个尖括号中间加入网格的维度(Dg),线程块数(Db),组成了加载核函数中的执行配置和维度的声明。
* 在函数前面加上声明执行位置的标识符
  * __global__ 可以在主机端调用并在设备端执行  注：在设备上调用__global__函数称为“动态并行”技术
  * __host__   在主机端调用在主机端执行   （默认限定符,通常省略）
  * __device__   在设备端调用并在设备端执行  （从核函数中调用的函数需要这个标识符）
  * __host__device__  这可以让系统同时编译两个版本

## 2. 核函数的定义
* 核函数不能有返回值,需要如此声明：  `_global_ void aKernel (typedArgs)`
* 核函数提供了对于`每一个线程块`和`线程的维度数`和`索引变量`。
  * 维度数目变量：   
	* gridDim     网格中的线程块数目
	* blockDim    每个线程块中的线程数目
  * 索引变量
	* blockIdx    这个线程块在网格中的索引
	* threadIdx   这个线程在线程块中的索引
* 在GPU上执行的函数通常不能访问在主机端CPU可以访问的内存中的数据。

## 3. cuda 的重要API

1. 数据传送API
	* `cudaMalloc()`    可以分配设备端内存
	* `cudaMemcpy()`    将数据传入或者传出设备
	* `cudaFree()`      释放掉设备中不再使用的内存
	
2. 核函数的并发执行放弃了对顺序执行的控制,因此需要为同步和并发执行提供相应函数。
	* _syncThreads()             在一个 **线程块** 中进行线程同步
	* cudaDeviceSynchronize()    可以有效的同步 **一个网格中的所有线程**
	* atomicAdd()                原子操作,防止多线程并发访问同一个变量时的冲突
	
3. 重点数据类型
	* Size_t   代表内存大小的专用变量类型
	* cudaError_t  错误处理专用变量
	
4. 向量类型：
	* CUDA将标准C的向量数据类型拓展到4维,独立的组件通过 .x .y .z .w进行访问
	* `unit3`变量时一个包含了三个整型数的向量,在`blockIdx和threadIdx`上使用
	* `dim3`变量与unit3变量一样,但将为声明的变量设置为1,在 `gridDim和blockDim`上使用

5. 向量的使用  
   * 使用**一维,二维,三维的索引**来标识线程,构成**一维,二维,三维的线程块**
   * dim3结构类型使用在**核函数**的调用 `<<< , >>>`中
   * 相关的内置变量
     * `threadIdx` 可以获得线程thread的ID索引,如果线程是一维的就使用 `threadIDx.x` 如果是二维的则可以使用`threadIDx.y`
     * `blockIdx` 线程块的ID索引,同样有`.x  .y  .z` 
     * `blockDim` 线程块的维度,同样有 `.x  .y  .z`
     * `gridDim` 线程格的维度,同样有 `.x  .y  .z`
   * 对于一位的block,`threadID==threadIdx.x`
   * 对于大小为`(blockDim.x, blockDim.y)`的 二维 block,线程的`threadID==threadIdx.x+threadIdx.y*blockDim.x`
   * 同理对于三位的 `threadID=threadIdx.x+threadIdx.y*blockDim.x+threadIdx.z*blockDim.x*blockDim.y。`

# 3. cuda 的网格

