# 特征提取 feature extraction


## ORB (Oriented FAST and Rotated BRIEF) - 一种快速特征点提取和描述算法

Ethan Rublee, Vincent Rabaud, Kurt Konolige, Rary R.Bradski
ORB: An Efficient Alternative to SIFT or SURF  (2011)

ORB 特征分为两个部分
* 特征点提取    ： 由 FAST (Features from Accelerated Segment Test) 算法发展而来
* 特征点描述    ： 由 BRIEF (Binary Robust Independent Elementary Features) 特征算法改进而来
* 总的来说 ORB = oFAST + rrBRIEF。 速度是 sift 的 100 倍, surf 的10 倍, 在 SLAM 和 无人机视觉等领域得到了广泛的应用

### FAST keypoint Orientation

使用 FAST 算法提出特征点后, 附加上一个 特征点方向, 以此来保证 特征点的旋转不变性  


判断特征点:
* 从图像中选择一点 P, 以 P 为圆心画一个半径为 3 像素的圆
* 圆周上如果有连续 `N` 个像素点的灰度值比 P 点的灰度值 大 or 小, 则 P 为特征点
* 这种算法被俗称为 FAST-`N`, 常用的有 FAST-9 和  FAST-12
* 快速算法
  * 直接去检测位置 1, 5, 9, 13 上的灰度值
  * 如果 p 是特征点, 那么这 4 个位置上应该有 3 额或者3个以上的像素值都大于 or 小于 p 的灰度值
  * 如果该条件不满足的话, 则可以直接排除 p 为特征点  


特征点的方向: 实现特征点的旋转不变性
* 使用 矩法 (moment) 来确定 FAST 特征点的方向
* 


### rBRIEF 改进后的 BRIEF

BRIEF 算法： 生成一个二进制串的特征描述符
* 在一个特征点的邻域内, 选择 n 对像素点 Pi, Qi.  i=1,2,...,n
* 比较每个点对的像素值大小, 如果 I(Pi) > I(Qi), 则生成的二进制串中该位为 1, 否则为 0
* 一般 N 会取  128 256 512 等长度, opencv 默认为 256
* 旋转不变性
  * 在 BRIEF 的测试中, 对于旋转不是特别厉害的图像里, BRIEF 生成的描述子的质量非常高, 大多数能优于 SURF
  * 但是在旋转角度大于 30 度后, 匹配率快速降到 0, 因此需要改进

旋转不变的 BRIEF:
* 

