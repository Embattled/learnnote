# 线性代数  



## Eigen value 特征值  

所谓正方形矩阵 A 的特征值, 即对于一个固定的特征向量 $v$, 特征值$\lambda$ 满足
$$Av=\lambda v$$

一个矩阵有多个特征向量, 特征向量和特征值是一一对应的, 特征值分解就是将整个矩阵分解成:
$$A=Q\sum Q^{-1}$$

这里 Q 是由特征向量组成的矩阵, $\sum$ 则是一个对角矩阵, 每个元素都是一个特征值, 且由大到小排列  


## Singular Value Decomposition  奇异值 

奇异值分解同特征值类似, 但是对于矩阵的要求减轻了, 可以是任意 `N*M` 形状的矩阵, SVD 可以定义为
$$A=U\sum V'$$

这里 U, V 分别是 `N*N` `M*M` 的方针, $\sum$ 就是要求的奇异值矩阵 

奇异值矩阵虽然不是严格的方阵, 但也是按着 45度对角线上存放着权重, 其中大部分都是接近或者等于0 的, 如果把接近 0 的部分丢掉, 则实现了 SVD 压缩
$$A_{N\times M}\approx U_{N\times r}\sum V_{r\times M}^{-1}$$
 


# Invertible matrix


## Rule of Sarrus

是一种用于快速记忆 2x2 和 3x3 行列式的值的方法, 

Rule of Sarrus 是 Leibniz Determinants Formula 的特例


