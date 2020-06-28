# Computer Hardware Fundamentals


# HPC

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

