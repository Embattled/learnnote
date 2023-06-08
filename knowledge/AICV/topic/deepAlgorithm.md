# Data Layout

## NHWC Layout

## HCHW Layout


# Optimize Algorithm

## Img2Col - 卷积加速算法





## winograd - 卷积加速算法

由 Shmuel Winograd 于 1980 年提出, 在当时没有引起太大的轰动

于 2016 年的 CVPR 时, 由 Lavin 提出利用 winograd 来进行卷积加速, 并受到了关注
(2015) Fast Algorithms for Convolutional Neural Networks. Andrew Lavin, Scott Gray.
https://arxiv.org/abs/1509.09308


winograd 的主要思想 : 用更多的 加法 来减少 乘法, 从而降低计算量. (一般处理器中 乘法计算的时钟周期要大于加法计算的周期 )

用一维信号的卷积来举例
```py
# 输入信号 
d = [d0, d1, d2, d3]T 
​
# 卷积核 
g = [g0, g1, g2]T

# 得到的输出用 r0 r1 来表示, 即完整的计算应该是  
r0 = d0*g0 + d1*g1 + d2*g2   # 3次乘法+2次加法 
r1 = d1*g0 + d2*g1 + d3*g2   # 3次乘法+2次加法
```

在常见的卷积应用场景中, 往往 stride < kernel_size, 在矩阵乘法中会有规律的存在重复元素, 例如上面的 d1, d2 在两次计算中都存在  


winograd 的转换具体为
```py
m1= (d0 - d2)*g0
m2= (d1 + d2)*(g0+g1+g2)/2
m3= (d2-d1)*(g0-g1+g2)/2
m4= (d1-d3)*g2

# 最终的计算转换为
r0 = m1+m2+m3
r1 = m2-m3-m4

# 总体的计算为 4 次乘法, 加法的次数变多了
```

泛用形式的一维卷积转换则更加复杂一些  
* $Y=A^T[(Gg)]\odot(B^Td)$
* 其中 g, d 为原本的卷积核和输入信号
* $G$ : 卷积核变换矩阵
* $A^T$ : 输出变换矩阵 
* $B^T$ : 输入变换矩阵
* GAB 都是有着固定 pattern 的矩阵

对于二维卷积来说, 同样可以将二维卷积展开, 再进行分块处理后即可获得类似于一维处理的公式  

```py
# input
k0 k1 k2
k3 k4 k5
k6 k7 k8

# kernel
w0 w1
w2 w3

# 二维卷积展开成矩阵向量乘法
# 矩阵
k0 k1 k3 k4
k1 k2 k4 k5
k3 k4 k6 k7
k4 k5 k7 k8

# 向量
w0
w1
w2
w3

# 进行分块, 得到和1维类似的形式
K0 K1
K1 K2 
# 
W0
W1
```

winograd 的加速总结:
* 对于一维卷积, 假设输出长度为 m, 卷积核的长度为 r, 则对应的乘法计算次数为 m+r-1 次
* 对于二维卷积, 假设输出长度为 mXn, 卷积核的大小为 rXs, 则需要的乘法次数为 (m+r-1)X(n+s-1)
* winograd 会铜鼓减少乘法次数来实现速度提升, 但是加法的运算次数和存储空间的要求会增加, 且会随着 卷积核和 tile 的大小而增大, 因此要考虑实际应用中的空间代价
* 转换矩阵越大, 精度损失就会越大. 因为 winograd 卷积会产生一些除不尽的参数.
* winograd 适合较小的卷积核和 tile, 大尺寸卷积核的加油更推荐使用 FFT
* 对于 stride 等于 2 的适合加速收益不高
* 深度可分离卷积用 winograd 收益不高
* tile 块越大, 加速效果越好

