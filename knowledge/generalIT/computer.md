# 1. Computer Hardware Fundamentals




# 2. CPU processor

## 2.1. varia

机器语言 : 命令 + 地址

CPU组成:
* 控制电路
* 计算电路, 算术计算 + 逻辑计算
  * ALU (Arithmetic Logic Unit)
  * FPU (Floating Point number processing Unit)
* 寄存器
  * 通用寄存器
  * 命令寄存器 : 存储要执行的机器语句
  * 程序计数器 : IP, 记录下一条程序的地址
  * 地址寄存器 : 要读取的数据的主内存地址
  * 数据寄存器 : 读取到的主内存数据

CPU的性能:
* 时钟数 : clocks, 一秒震动多少回, 1GHz
* CPI (Clock cycles Per Instruction): 一条命令需要几个时钟数, 一般来说根据命令的不同为 4~30 左右
* MIPS (Million Instructions Per Second) : 每秒能够执行多少条指令, 按照命令频度以及对应的CPI的平均来计算
* FLOPS (Floating-point Operation Per Second) : 每秒能够处理多少次浮点小数计算命令

CPU高速化技术: 
* RISC (Reduced Instruction Set Computer) : 精简指令集, 适合流水线化
* CISC (Complex Instruction Set Computer) : 复杂指令集, 拥有微程序架构, 不适合流水线化

超流水线
* 很可能会出现主频较高的CPU实际运算速度较低的现象

超标量 superscalar 
* 在单颗处理器内核中实行了 : 指令级并行, 即两条流水线

VLIW (Very Long Instruction Word)
* 将可以并行化的多条命令合成一条超长命令

# 3. 操作系统


操作系统的功能:
* 效率的使用硬件资源
* 提供用户的操作界面, 不用意识到具体的硬件构造
* 提供 API (Application Program Interface)


## 系统管理

操作系统的处理单位:
* Job : 具体的指令, 执行程序时即 job 的逐个执行 (job step), 对于 main frame 的OS还存在 JCL (Job Control Language)
* task (process) : 操作系统的资源分配单位, 一般以程序为单位

操作系统层级的并列处理
* spooling  : 为外部的低速设备加入缓冲池, CPU通过缓冲池来间接读写外部设备
* multi-programming: 为了缓解某些程序会进行IO导致CPU空闲的情况, 同时运行多个程序提高CPU使用率, 在目前的实现中基本以 multi-task 来称呼
* multi-task :  单颗CPU同时执行复数个 task
* multi-thread: 在 multi-task 基础上把 task 再次细分成 thread 来调配CPU资源, 和 task 之间不同, 同一个 task 的thread使用相同的内存空间, 仅仅区分了CPU使用时间

### 3.1.1. Job 管理

分两种
* job scheduler :
  * 通过解释 JCL (Job Control Language) 来实现 job 管理
  * job 的执行准备工作完成后, 到 job 执行完毕分为了4个阶段
  * 1 read: 将 job 登录到等待队列
  * 2 initiator : 为队列中最高优先级的job分配资源(内存, IO设备等), 读入程序
  * 3 terminator: job 执行结束, 释放占用的相关资源
  * 4 writer : 处理程序的输出数据, 传送给打印机等设备
* master scheduler :  
  * 直接使用相关的 interface 来实现 job 管理

### 3.1.2. task 管理

提高 CPU 的使用率, 对CPU资源分配的相关机能

task 的生命周期:
* task 在程序开始执行时生成, 在程序执行结束时终止, 在生命周期中有3中状态可以处于
* 1 run : 分配到了 CPU 资源, 正在执行, 执行中的 task 有可能会进行状态转移
  * 更高优先级的 task 进入了ready队列, 进行切换, 原本 task 进入ready状态(队列)
  * CPU使用时间超过限制, 为了确保其他 task 的执行进行的分时切换, 进入ready状态
  * IO资源等待, 进入 wait 状态
* 2 ready : 除了 CPU 资源以外其他的资源都已经准备完毕, 等待状态, 这里有一个等待队列
* 3 wait : 等待 CPU 资源以外的其他资源 (IO等), 即使有CPU资源也无法继续执行

CPU的分配, 中断:
* 指的是暂停当前程序, 将CPU分配给其他程序, 分为外部调用和内部调用
* 1 外部中断: 一般是IO设备的处理完成后的调用, 系统异常, 系统的倒计时任务响应等
* 2 软件内部中断: 程序调用OS的功能产生的调用, SVC (SuperVisorCall)
* 3 软件异常中断: 浮点溢出, 除以0等

### 3.1.3. 数据管理

指的是OS对磁盘上文件的管理机能

文件编成的5种规格: 主要用在 main framework 系统中
1. 顺序记录: 
   - 只能按照记录的顺序来读取数据 , 只能顺序读取
   - 用在磁带上, 顺序读取设备只能使用这种存储方式
   - 记录效率较高, 适合长久保存的数据
2. 索引记录
   - 对各个文件创建索引 index, 来实现随机访问
   - 存储空间可以分成: 索引区, overflow 区
   - 易于检索, 适合检索和更新比较频繁的应用场景
   - 数据的追加只能放在 overflow区, 数据的消除仅仅作用在索引上
   - 空间的利用效率较低, 如果频繁的追加和消除数据会导致利用率进一步降低
3. 直接编成
   - 每条记录都有一个key, 通过 key 可以获得文件的存储位置, 达到随机访问
   - 需要确保 key 不重复
4. 区分编成
   - 复数个顺序编成的文件作为一个簇, 登录表中进行管理
   - 登陆表也称作 directory, 一般用作程序文件的存储
5. VSAM 虚拟存储文件
   - 将复数个存储设备统合成一个整体的虚拟存储设备
   - 现在的 main framework 比较常用, 不使用具体 1~3 任意一种方法, 而是使用3个数据库来进行管理
   - 输入顺序数据库 等同于1
   - key 顺序数据库 等同于2
   - 相对record库 等同于3

文件系统: 对于 UNIX 或者 Windows 来说, 程序是阶层构造管理的 (directory, folder)
* 阶层的顶点就是根节点, 从根输出发的路径就是 绝对路径 絶対パス
* 同当前位置 (カレントディレクトリ) 出发的路径就是 相对路径 相対パス

备份: 防止意外的文件损坏, 将一部分的数据进行拷贝保管, 出现意外的时候可以 restore リストア
1. full backup : 对所有文件拷贝备份
2. 差分备份 : 根据上一次 full backup 以来的变化, 仅仅将变化的部分进行备份, 恢复的时候需要将起始的 full backup 差分备份都执行, 消耗的时间更长
3. 增量备份 : 根据上一次 full backup 或者增量备份以来的变化, 将变化的部分进行备份, 需要备份的数据最小, 备份消耗的时间最短, 但是还原的时候, 需要用到初始的 full 和之后的所有增量备份, 消耗的时间最长


### 3.1.4. 内存管理

大概分成 实际内存和 虚拟内存两种
* 实际内存管理 : 需要程序考虑到内存的分配
  - 目前只有嵌入式开发的程序系统需要对内存进行实管理
  - 1 swaping 方式 : 在内存不足的时候, 将没有执行的程序暂时移动到外部存储上
  - 2 overlay 方式 : 实现将程序分成可以独立执行的 segment, 由一个在内存中常驻留主控制模块来管理其他模块, 用到谁就把谁调入内存
  - 两种内存管理模式都有可能导致内存碎片化 fragmentation, 将碎片统一的机能称作 compaction

* 虚拟内存 : 程序不需要考虑内存
  - 目前的 main framework, 服务器, 个人电脑的系统都是用的虚拟内存管理
  - 将外部存储的一部分用作 主内存, 实际内存和虚拟化的内存合并成整个 虚拟内存空间
  - 整个虚拟内存按照 页 page 来管理, 页 是固定大小的
  - 如果需要执行的 页 不再主内存, 会发生 page fault 中断
  - paging : 将需要的 页 调入主内存的过程, 分为 page out, page in
  - 页置换算法: 在 page fault 发生且主内存已经没有空间, 需要先进行 page out的时候的选择算法, 常用算法:
    - 1 FIFO (先进先出, 类队列), 认为最早读入的页用到的可能性最少
    - 2 LRU (Least Recently Used), 选择最长时间没有被访问过的页

## 系统构成

根据数据处理的时机来分类:
* batch 处理
  * 存储一定程度的数据 (时, 天, 周) 然后一口气处理
  * 计算工资, 打印订单
* transaction 处理 (トランザクション)
  * 接收到数据的时候立即处理
  * 订票, 预定座位等

数据的处理方式
* 集中处理: 数据和计算能力在单一计算机上
* 分散处理: 复数计算机共同分担任务, 但是管理变得更加复杂, 难以保证数据的一贯性
  * 负荷分散, 机能分散: 任务的分担基准, 根据当前计算机的负荷量或者计算机的性能来分配任务
  * 水平分散, 垂直分散: 根据多台计算机之间是否有等级上的区分来判定

* C/S 系统: 将客户端和服务端分离, 分散处理的一种
  * 2层模型: 客户端-服务端
  * 3层模型: 客户端(GUI,WebPage)-应用端(程序)-服务端(数据库)

### 系统的信赖性

系统需要针对故障的冗余性
* 简单系统 Simplex (シンプレクス) : 毫无冗余的系统, 最低限度的部件
* 2重系统  dual : 2条相同的处理系统同时工作, 互相进行结果的验证
* 2倍系统 duplex: 一条主工作系统, 另一条备用系统, 备用系统平时不工作(不接电)
  * hot standby : 接电, 保持程序载入状态, 可以迅速切换到工作
  * warm standby: 接电, 仅仅只是维持系统启动的状态
  * cold standby: 不接电
 
提高系统信赖性的思考方法:
* fault tolerant フォールトトレラント   : 直接提高系统对故障的耐受力, 例如 hot standby
* fault avoidance アボイダンス          : 提高系统故障的发生率, 避免故障
* fail safe    フェールセーフ           : 故障发生时可以安全停止工作
* fail soft   フェールソフト            : 故障发生时自动确认, 并维持未发生故障机能的继续工作
* fool proof フールプルーフ             : 安全工程, 防止用户操作上的错误导致的故障

### 对信赖的评价指标

信赖性的简称 RAS RASIS:
* (R)ealiability  : 信頼性  系统可以在一定时间内平稳运行
* (A)vailability  : 可用性  在需要系统的时候总是可用
* (S)erviceability: 保守性  系统易于维护
* (I)ntegrity     : 完全性  防止系统故障, 或故障后可修复
* (S)ecurity      : 機密性  防止数据的不正访问, 数据的机密性


指标:
* MTBF  Mean Time Between Failures  : 代表信赖性的指标, 平均无故障工作时间
* MTTR  Mean Time To Repair         : 代表保守性的指标, 故障的平均修理时间
* 稼働率                             : 代表可用性的指标, 需要使用的时候的可用率
* 稼働率= MTBF/(MTBF+MTTR)
* bathtub曲线 バスタブ               : 故障率和时间的二维曲线
  * 初级故障, 由高到低
  * 偶发故障, 中期一条直线
  * 磨损故障, 后期逐渐升高

### 对性能的评价指标

* throughput スループット : 单位时间的处理能力
  * TPS Transactions Per Second
  * TPM Transactions Per Minute
* Response time: 应答时间
  * 对于 transaction 处理的系统, 从输入设备的数据输入, 到最初的结果返回时的时间间隔
* Turn around time  ターンアラウンドタイム
  * batch 处理的系统, 输入数据的读取, 处理, 输出全部结束所花费的时间, 也就是整个流程需要的等待时间

benchmark test ベンチマーク: 对系统的性能评价
* SPEC  Standard Performance Evaluation Corporation
  * 美国几个电脑制作商联合的组织, 定义了几个CPU的评价任务
  * SPECint
  * SPECfp
* TPC   Transaction Processing Performance Council
  * トランザクション処理性能評議会
  * 对于 transaction 处理的系统的评价
  * 每秒钟可以处理的 transaction (TPS) 为基准来代表性能
  * 分成4个子类 ABCD

# 4. HPC

* HPC systems have been developed for scientific computation (numerical simulation).
* Floating-point notation
  * A number is represented by a mantissa (significand) and an exponent, similar to scientific notation
  * Representable range extended
  * Complicated processing needed for arithmetic operations
* The performance of a supercomputer is discussed based on how many floating-point operations the supercomputer can execute per unit time.
* FLOP the number of `FL`oating-point `O`perations `P`er sec

* GPU
  * 64CUDA cores are grouped to on **SM**(Streaming Multiprocessor)
* Graphics Rendering Pipeline
* Texture mapping
* Latency
  * execution time for a task (shorter is better)
*  Throughput
   *  The number of tasks per unit time (more is better).
* CPU: Latency-oriented design (=speculative)
  * CPU has a large cache memory and control unit.
* GPU :Throughput-oriented design (=parallel)
  * GPUs devote more hardware resources to ALUs.



* Speculation is one of key technologies in CPU.
* cache hit ratio
  * The cache hit ratio is very important for modern processors toachieve high performance. To increase the cache hit ratio, cachememory occupies quite a large area of the chip.


* Vector Processing
  * vector operations can be made faster than a sequence of scalar operations on the same number of data items.

