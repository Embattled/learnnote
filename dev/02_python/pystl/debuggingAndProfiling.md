# Debugging and Profiling

该分类用于辅助 python 的开发
* debugger 为 python STL 级别的软件
* profilers 用于对程序进行测速
* audit 用于了解运行时行为??

# pdb — The Python Debugger



# The Python Profilers

提供确定性的分析信息 deterministic profiling of programs.  
包括了两个具体类, 两个类的接口是相同的, 区别在于底层实现
* cProfile  : 适合大多数用户, 是基于 C 语言的, 具有合理的开销, 适合长时间的 大开销的程序, 基于 `lsprof`
* profile   : 由纯 python 实现, 模仿了 cProfile 的接口, 但是具有较大开销, 适合用户需要自定义 profile 行为的时候

一个 profile 是统计信息的集合, 可以通过 `pstats` 模组来格式化输出  

注意: 该模组只适用于分析 python 程序的时间构成, 而不能作为性能基准测试, 因为分析器会为 python 代码引入开销, 而不会影响 C 的代码. 当试图分别对 C 代码和 python 代码进行基准测试的时候, C 的代码总是会显得更快.

## profile and cProfile Module Reference

完整的两个类的文档, 两个类的接口相同


## The Stats Class

