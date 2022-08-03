- [1. Image Signal Processing (ISP)](#1-image-signal-processing-isp)
  - [1.1. Image Pipeline 的操作分区](#11-image-pipeline-的操作分区)
  - [1.2. Image Pipeline 的各种操作及其缩写](#12-image-pipeline-的各种操作及其缩写)
  - [1.3. Gamma](#13-gamma)
  - [1.4. Raw 图像](#14-raw-图像)
- [2. Filter](#2-filter)
  - [2.1. Filter in signal process](#21-filter-in-signal-process)
  - [2.2. Local Linear Filter](#22-local-linear-filter)
    - [2.2.1. Convolutional Filter and Fourier Transform](#221-convolutional-filter-and-fourier-transform)
    - [2.2.2. Deconvolution](#222-deconvolution)
    - [2.2.3. Blur Using Local Linear Filter](#223-blur-using-local-linear-filter)
      - [2.2.3.1. Box Filter / Mean Filter](#2231-box-filter--mean-filter)
      - [2.2.3.2. Gaussian Filter](#2232-gaussian-filter)
    - [2.2.4. Edge Detection Using Local Linear Filter](#224-edge-detection-using-local-linear-filter)
      - [2.2.4.1. Sobel operator 索贝尔算子](#2241-sobel-operator-索贝尔算子)
      - [2.2.4.2. Prewitt operater 普利维特算子](#2242-prewitt-operater-普利维特算子)
      - [2.2.4.3. Laplacian Operator 拉普拉斯算子](#2243-laplacian-operator-拉普拉斯算子)
      - [2.2.4.4. Roberts Cross operator 罗伯茨交叉边缘检测](#2244-roberts-cross-operator-罗伯茨交叉边缘检测)
    - [2.2.5. Gabor Feature Extractor](#225-gabor-feature-extractor)
  - [2.3. Local Non-linear Filter](#23-local-non-linear-filter)
    - [2.3.1. Denoising Using Local Non-linear Filter](#231-denoising-using-local-non-linear-filter)
      - [2.3.1.1. Median Filter 中值滤波](#2311-median-filter-中值滤波)
      - [2.3.1.2. Anisotropic Diffusion Filter  各向异性扩散滤波](#2312-anisotropic-diffusion-filter--各向异性扩散滤波)
      - [2.3.1.3. Bilateral Filter 双边滤波](#2313-bilateral-filter-双边滤波)
      - [2.3.1.4. Bilateral Grid 基于双边滤波的改进 Filter](#2314-bilateral-grid-基于双边滤波的改进-filter)
    - [2.3.2. Feature Extraction Filter Non-linear](#232-feature-extraction-filter-non-linear)
      - [2.3.2.1. AutoEncoder](#2321-autoencoder)
      - [2.3.2.2. Local Binary Pattern](#2322-local-binary-pattern)
  - [Global (Non-local) Filter](#global-non-local-filter)
    - [NL-means](#nl-means)
  - [2.4. Transform Domain Filter 变换域滤波](#24-transform-domain-filter-变换域滤波)
    - [2.4.1. wiener filter 维纳滤波](#241-wiener-filter-维纳滤波)
    - [2.4.2. Wavelet Threshold Denoise 小波阈值滤波](#242-wavelet-threshold-denoise-小波阈值滤波)
    - [2.4.3. BM3D  Block-matching and 3D filtering](#243-bm3d--block-matching-and-3d-filtering)
- [3. Noise in image / signal](#3-noise-in-image--signal)
  - [3.1. Noise Modeling 噪音的建模与计算](#31-noise-modeling-噪音的建模与计算)
    - [3.1.1. Poissonian-Gaussian Model](#311-poissonian-gaussian-model)
      - [3.1.1.1. Raw-Data Poission-Gaussian Modeling](#3111-raw-data-poission-gaussian-modeling)
      - [3.1.1.2. Poissonian-Gaussian Modeling Algorithm](#3112-poissonian-gaussian-modeling-algorithm)
    - [3.1.2. Denoising](#312-denoising)
  - [3.2. Filter Denoise](#32-filter-denoise)
    - [3.2.1. spatial denoise  空间降噪](#321-spatial-denoise--空间降噪)
      - [3.2.1.1. mean filter  平均滤波降噪](#3211-mean-filter--平均滤波降噪)
    - [3.2.2. transform denoise 频域降噪](#322-transform-denoise-频域降噪)
  - [3.3. Ringing Artifect 振铃效应](#33-ringing-artifect-振铃效应)
- [4. HDR high dynamic range](#4-hdr-high-dynamic-range)
  - [4.1. HDR 的实现要求](#41-hdr-的实现要求)
  - [4.2. Multi-Frame Noise Reduction 基础的多帧降噪](#42-multi-frame-noise-reduction-基础的多帧降噪)
  - [4.3. Image Fusion](#43-image-fusion)

# 1. Image Signal Processing (ISP)

作为光线传感器来说, 只能接受光的强度, 无法得知光的颜色
* 基础的相机感光元件是通过颜色滤镜和光线传感器的组合来实现的
* 通过在 Sensor 分小区域附加上 RGB 三色滤镜, 来得到 Bayer 


Image Pipeline : 对 Bayer 图像进行处理, 得到标准 RGB 图像  

一般流程: Bayer -> Pipeline -> YUV -> Jpeg Encoder -> JPG

## 1.1. Image Pipeline 的操作分区

* Software Control (SW controal)

* Image Front End (IFE)
  * 基本上 Bayer 格式处理的流程就在这一部分

* Bayer Processing Segment (BPS)

* Image Process Engine (IPE)


## 1.2. Image Pipeline 的各种操作及其缩写

Image Pipeline 中包括以下一系列操作的组合

* Bad Pixel Correction (BPC) 坏像素点修复 
* Bad Cluster Correction (BCC) 坏簇修复
* Adaptive Bayer Filter (ABF) 修复 R Gr Gb B 的通道平衡
* Green Imbalance Correction (GIC) : Gr Gb 的平衡调整
* De-Mosaic (DM): 去马赛克, 即 Bayer 格式到 RGB 的变换
* 
* Defective Pixel Correction (DPC) : 修复缺陷像素
* Color Correction (CC): 颜色补正
* Global Tone Mapping (GTM) : 调整图像整体的明度
* Image Correction and Adjustment (ICA): 镜头曲线歪曲修复
* Temporal Filter (TF): 时域滤波
* Two Dimension Lookup Table : 色相变更表
* Color Conversion (CV): 色相移动
* Chroma Suppression (CS): 特定亮度的区域进行色彩压制
* Adaptive Spatial Filter (ASF) : 边缘增强
* Grain Adder (GRA) : 提高质感的噪点像素追加, 放在成品图像输出前的最终阶段 


* High Frequency Noise Reduction (HNR) : 高频降噪
* Low-mid Frequency Noise Reduction (LNR) : 中低频降噪
* Hybrid Noise Reduction (HNR) : YUV数据的层面进行降噪处理
* Advanced Noise Redution (ANR) : 保证图像边缘的降噪处理


* PD Pixel Correction (PDPC)  : 通过位相差检出的专用像素点来实现自动对焦
* Black Level Correction (BLC) : 修复黑色像素
* Lens Shading Correction (LSC) : 镜头阴影矫正
* White Balance Gain (WBG)  : 白平衡增益, 由于绿色的光线和其他双色的光强不同, 需要对 RB  通道乘以系数来修正


## 1.3. Gamma

## 1.4. Raw 图像

RAW文件记录的是CCD上的电荷包经过AD转换后的数值, 尚未进行其他非线性的处理

RAW格式的文件在不同的相机上有不同的扩展名和编码方式, 例如NEF, CRW, CR2等, 许多使用的是私有格式

对于CCD数字图像, 可以认为其最终输出包括以下几个部分
1. 偏置: 即使曝光时间为0, 仍会有一个本底图像输出, 记做 `OFFSET`
2. 暗场: 即使没有光照, 仍会有一个随时间增强的暗场图像输出, 记做 `DARKFRAME`
3. 目标: 真正目标天体产生的光电子像, 记做 `SIGNAL`
4. 平场: 对于一个均匀亮度的目标, 可能会输出一个强度不均匀的平场图像, 记做 `FLAT-FIELD`
对于一般的一幅RAW格式的图像, 记做IMAGE, 则 `IMAGE=SIGNAL+DARKFRAME+OFFSET`

要获取理想的图像信号, 较为朴素的处理方式是  
`SIGNAL0 = (IMAGE-DARKFRAME-OFFSET) / NOR(FLATFIELD-DARKFRAME-OFFSET)` NOR 指的是标准化


# 2. Filter

Filter 的概念原本起于数字信号处理, 后来被延申到图像处理, 因此有很多概念或者滤波名称都是沿用着数字信号处理中的名称  


对于数字图像 (or Computer Vision) 领域来说, filter 的用途是用来将图像变得更好, 或者提取图像中的有用特征:
* 抑制不需要的特征    e.g. Denosing
* 强调有用的特征      e.g. Edge Detection, Feature Extraction
* 图像编码 (Encoder)  e.g. Local Binary Pattern, Increment Sign Correlation


CV的 Filter 处理
* 参考全局数值的 Filter:
  * 线性
    * 全局噪音
    * 简单的明度调整
  * 非线性
    * 二值化
    * 标准化
    * 非线性系数乘法 (e.g. gamma补正)
    * Neural Network (Auto Encoder)
* 参考局部的 Filter     : 根据目标像素的相邻像素值来更新对象像素值
  * Kernel Size : 参照的相邻像素的范围
  * 线性 Filter
    * 通用 Convolution Filter, 例如:
    * Blur (Box/Gaussian)
    * Edge Detection (Sobel / Laplacian)
    * Featrue Extraction (Gabor)
  * 非线性 Filter
    * 各种 Denoiser (Bilateral Filter, Guided Filter, Anisotropic Filter, Median Filter)
    * 非线性 Edge Detection (Canny)
    * Encoder (LBP, Increment Sign Correlation)
* 变换域 Filter:
  * 小波阈值滤波
  * 

## 2.1. Filter in signal process

信号根据傅里叶分解, 可以拆分成一系列不同频率的正弦波的叠加, 再根据频率可以进行滤波

频率Filter的种类:
* Low-pass    : 低通
* High-pass   : 高通
* Band-pass   : 中通
* Band-stop   : Low-high-pass, 中间频率被阻断
* Low-band-pass: low-pass + band-pass, 高频和中低频被阻断, 中频和低频通过
* Band-high-pass: band-pass + high-pass
* Low-band-high-pass: 中低频和中高频被阻断

## 2.2. Local Linear Filter

局部线性 Filter : Convolutional Filter, Kernel 的形状和值是固定的

使用局部线性滤波的任务
* Blur, Denoising, Shapening
* Edge Detection, Feature Extraction

分离的 Filter:
* 对于对称的 `k*k` Filter, 可以将其分成两个一维的filter `1*k` `k*1` 
* 具体的效果是降低计算复杂度 $O(n*m*k*k) \rArr O(2*n*m*k)$

### 2.2.1. Convolutional Filter and Fourier Transform

对于卷积操作, 假设 $f,g,h$是 kernel 或者图像
* 存在结合律  : $(f*g)*h = f*(g*h)$  这里 $*$ 是卷积操作
* 存在卷积定理: $F(f*g)=F(f)*F(g)$ 这里 F 是 Fourier Transform

### 2.2.2. Deconvolution

根据卷积核 $f$ 和应用Filter后的图像 $f*g$, 还原出原始图像 $g$

同样利用傅里叶变化 F:
* 卷积定理: $F(f*g)=F(f)*F(g)$ 
* 有 $g=F_{inv}(F(f*g)/F(f))$

由于傅里叶变换的特性, 高频率的信息会丢失, 导致振铃效应 (ringing artifect)

### 2.2.3. Blur Using Local Linear Filter

使用简单的线性 Filter 可以很容易的实现模糊效果, 但通过模糊实现的降噪通常会导致图像的边缘也同时被模糊

#### 2.2.3.1. Box Filter / Mean Filter

最基础的卷积 Filter

* 一般的卷积操作的实际复杂度是 $O(n*m*k*k)$, k是 kernel size
* 利用累加的 Integral Image, 任意 kernel size 的 Box Filter 可以在 $O(n*m)$ 实现

#### 2.2.3.2. Gaussian Filter 

相比于 Box/Mean Filter, 模糊效果较为浅
* 利用二维高斯函数来生成 Kernel, 作为参数的方差 sigma 可以进行指定, 来调整 


### 2.2.4. Edge Detection Using Local Linear Filter

边缘检测算子某种程度上也可以当成 Feature Extractor

#### 2.2.4.1. Sobel operator 索贝尔算子

主要用作边缘检测, 在技术上, 它是一次离散性差分算子, 用来运算图像亮度函数的灰度之近似值。在图像的任何一点使用此算子, 将会产生对应的灰度矢量或是其法矢量  

该算子包含两组3x3的矩阵, 分别为横向及纵向, 将之与图像作平面卷积, 即可分别得出横向及纵向的亮度差分近似值。如果以A代表原始图像, Gx及Gy分别代表经横向及纵向边缘检测的图像灰度值


Sobel的卷积因子  
水平Gx:     
| +1  | 0   | -1  |
| --- | --- | --- |
| +2  | 0   | -2  |
| +1  | 0   | -1  |

竖直Gy:  
|   +1 |   +2 |   +1 |
| ---: | ---: | ---: |
|    0 |    0 |    0 |
|   -1 |   -2 |   -1 |


图像每一个像素的横向纵向灰度值通过`平方相加再开根号`的形式结合, 也可以直接`绝对值相加`, 计算出的灰度值要记得缩放颜色深度  

`The angle of orientation of the edge (relative to the pixel grid) giving rise to the spatial gradient is given by`  
要计算边缘梯度,也就是边缘的方向角度  (空间梯度/`spatial gradient`)   
`角度=arctan(Gy/gx)`

Sobel算子根据像素点上下、左右邻点灰度加权差, 在边缘处达到极值这一现象检测边缘。对噪声具有平滑作用, 提供较为精确的边缘方向信息, 边缘定位精度不够高。当对精度要求不是很高时, 是一种较为常用的边缘检测方法。

#### 2.2.4.2. Prewitt operater 普利维特算子

该算子与 Sobel 类似  
水平Gx:     
| +1  | 0   | -1  |
| --- | --- | --- |
| +1  | 0   | -1  |
| +1  | 0   | -1  |

竖直Gy:  
|   +1 |   +1 |   +1 |
| ---: | ---: | ---: |
|    0 |    0 |    0 |
|   -1 |   -1 |   -1 |


Prewitt算子利用像素点上下、左右邻点灰度差, 在边缘处达到极值检测边缘。`对噪声具有平滑作用`, 定位精度不够高。  
图像每一个像素的横向纵向灰度值通过`平方相加再开根号`的形式结合, 也可以直接`绝对值相加`, 计算出的灰度值要记得缩放颜色深度  

#### 2.2.4.3. Laplacian Operator 拉普拉斯算子

拉普拉斯算子是同时对二维检测, 没有水平/垂直分量, 根据 kernel size 有不同的表现


kernel size = 1
|    0 |    1 |    0 |
| ---: | ---: | ---: |
|    1 |   -4 |    1 |
|    0 |    1 |    0 |

kernel size = 3
|    1 |    1 |    1 |
| ---: | ---: | ---: |
|    1 |   -8 |    1 |
|    1 |    1 |    1 |

#### 2.2.4.4. Roberts Cross operator 罗伯茨交叉边缘检测

Roberts算子采用对角线方向相邻两像素之差近似梯度幅值检测边缘。检测水平和垂直边缘的效果好于斜向边缘, 定位精度高, 对噪声敏感

水平Gx:     
| +1  | 0   |
| --- | --- |
| 0   | -1  |

竖直Gy:  
|    0 |   +1 |
| ---: | ---: |
|   -1 |    0 |

同样图像每一个像素的横向纵向灰度值通过`平方相加再开根号`的形式结合, 也可以直接`绝对值相加`, 计算出的灰度值要记得缩放颜色深度  


### 2.2.5. Gabor Feature Extractor

## 2.3. Local Non-linear Filter

局部非线性 Filter : 根据像素值, Filter 的值也会变化, 没有固定的数字描述

使用局部非线性滤波的任务 : Denoising, Feature Extraction

### 2.3.1. Denoising Using Local Non-linear Filter

基本上, 好的降噪 Filter 都是基于非线性算子的


#### 2.3.1.1. Median Filter 中值滤波

中值滤波的定义: $I'(u,v) = median{I(u+i,v+j)|(i,j)\in R}$ 这里的 R 是滤波核大小

更新像素值为临近像素值排列后的 `中值`  

#### 2.3.1.2. Anisotropic Diffusion Filter  各向异性扩散滤波

将图像看作热量场?! 像素值看作热流, 根据当前像素和周围像素的关系, 来确定是否要向周围扩散, 如果某个邻域像素和当前像素差别较大, 代表这个邻域像素很可能是个边界, 那么当前像素就不向这个方向扩散了


推导上都是热力学上的内容, 比较复杂, 属于迭代行方法  

#### 2.3.1.3. Bilateral Filter 双边滤波

是一种非线性的滤波方法, 于 1998 年被提出  
* 折中了 图像的空间邻近度和像素值相似度
* 考虑了 空域信息 + 灰度相似性
* 实现了较好的 保边去噪效果, 传统的 维纳滤波和高斯滤波会模糊边缘
* pros : 简单, 非迭代, 局部, 对于低频噪声有很好的效果
* cons : 对于彩色图像里的高频噪声, 双边滤波很难去除干净
* 虽然非迭代但是速度很慢

简单来说:
* 不光要求参与计算的像素位置近, 还需要数值相差不能太大
* 比高斯滤波增加了一个 高斯方差, 即基于空间分布的高斯滤波函数
* 在边缘附近, 离得较远的像素不会影响边缘上的像素值

$$I_{bilateral}(x)=\frac{1}{W_p}\sum_{x_i\in\Omega}I(x_i)f_r(||I(x_i)-I(x)||) g_s(||x_i-x||)$$
$$W_p=\sum_{x_i\in\Omega}f_r(||I(x_i)-I(x)||) g_s(||x_i-x||)$$

* $f_r$ 和 $g_s$ 分别代表了 range 和 space 对权重的影响函数, 一般直接都使用高斯核


双边滤波的加速化工作
* 当 $f_r$ 和 $g_s$ 是相同卷积核的时候, 可以将像素值作为第3维, 图像数据升到 3 维, 此时 gs 不变, 但是 fr 变为在第三维上的卷积
* 由于图像的数值范围一般较大, 内存吃紧, 可以使用下采样 dowsample 来降低内存消耗, 由于去噪本身就是模糊的过程, 因此下采样带来的误差通常可以忽略
* 对于下采样和升维操作, 可以定义图像为 $I(x,y)$ 坐标下采样率是$s_s$, 像素值下采样率是 $s_r$, 则建立三维空间的操作可以定义为
$$\Gamma([x/s_s],[y/s_s],[I(x,y)/s_r])+=(I(x,y),1)$$
* 通过高斯滤波后得到 $\hat{\Gamma}=g_{\sigma_s,\sigma_r}\times \Gamma$, 最后在使用上采样映射会二维图像 M
$$M(x,y)\larr \hat{\Gamma}([x/s_s],[y/s_s],[I(x,y)/s_r])$$


#### 2.3.1.4. Bilateral Grid 基于双边滤波的改进 Filter

于 SIGGRAPH 2007 被提出, 是从升维的思想拓展出来的通用化操作, 不单单可以用来降噪

使用 Bilateral Grid 的流程可以归纳为:
1. 下采样, 建立 Grid
2. 对 Grid 进行一些操作
3. 上采样回去
由于 Grid 的数据量较小, 因此可以实现很多实时效果  

Cross-bilateral filter: 建立 grid 时不使用原图的像素值, 而是使用其他图片 e.g. E(x,y), 但是执行卷积的时候仍然使用原图像素值    
$$\Gamma([x/s_s],[y/s_s],[E(x,y)/s_r])+=(I(x,y),1)$$




### 2.3.2. Feature Extraction Filter Non-linear

#### 2.3.2.1. AutoEncoder


#### 2.3.2.2. Local Binary Pattern 

## Global (Non-local) Filter

### NL-means

Buades, Antoni (20–25 June 2005).   
A non-local algorithm for image denoising.   
Computer Vision and Pattern Recognition, 2005.  

能充分利用图像中的冗余信息, 去噪的同时保留图像的细节特征, 执行时间较慢

主题思想: 
* 在整个图像范围内判断像素间的相似度 (对于每一个像素点, 都需要计算它与整个图像的相似度)
* 考虑到执行效率的问题, 在实现时, 设置两个固定大小的窗口 (搜索窗口, 邻域窗口)
  * 搜索窗口: 以目标像素 x 为中心的大窗口 (边长为D)
  * 邻域窗口: 以搜索窗口中每个像素 y 为中心的小窗口 (边长为d), y 会遍历整个搜索窗口
  * 通过比较: x 的小窗口邻域 和所有 y 的小窗口来为每个 y 赋予权值 w(x,y)
* 最终目标像素的值是通过所有 y 的加权和得到
$$\tilde{u}(x)=\sum_{y\in I}w(x,y)*v(y)$$

* w(x,y) 用以通过 x,y 的邻域窗口计算 x,y 之间的相似度, h 为平滑参数, h 越大高斯函数变化越平缓, 去噪水平提升的同时图像变得模糊

$$w(x,y)=\frac{1}{Z(x)}\exp (-\frac{||V(x)-V(y)||^2}{h^2})$$

* V(x)-V(y) 是小窗口像素差的平方平均, $d=2*ds+1$
$$||V(x)-V(y)||^2=\frac{1}{d}\sum_{||z||_{\infty} \le ds}||v(x+z)-v(y+z)||^2$$

* Z(x) 是权重函数的整体归一化系数, 其实就是整个窗口的系数和

$$Z(x)=\sum_y \exp (-\frac{||V(x)-V(y)||^2}{h^2})$$


## 2.4. Transform Domain Filter 变换域滤波

利用正交变换, 在另一个领域或者维度上进行滤波处理 

正交变换: 是信号变化的一系列统称
* 傅立叶变换
* 离散余弦变换
* 小波变换
* 多尺度几何分析（超小波）

### 2.4.1. wiener filter 维纳滤波

维纳滤波是通信领域的通用基本滤波方法, 由  Norbert Wiener 在 1942 年提出来的
* 本质的操作是 : 使估计误差的均方值最小化 (估计误差: 期望响应与滤波器实际输出之差)
* 又称 最小二乘滤波器 或 最小平方滤波器
* 在目前的工程视角下, 由于无法获得正确的系数, 直接将含有噪音的信号输入维纳滤波系统的画, 对于数字图像的降噪效果一般


### 2.4.2. Wavelet Threshold Denoise 小波阈值滤波

由信号处理领域专家 Donoho 1995年 提出的在小波域对白噪声进行降噪的方法  

主要思想:
* 白噪声在小波的各个尺度中均匀分布, 但是相对于主要信号的系数比较小
* 通过一个阈值来将其分开来, 小于阈值的系数就直接归零, 大于阈值的系数保持不变
* 将阈值滤波后的变换域信号 反变换回原本的 空间域(或者时间域)信号




### 2.4.3. BM3D  Block-matching and 3D filtering

BM3D 是 Alessandro Foi 于 2007 年提出的, 发布在 TIP 期刊上, 目前仍然是 CP 系降噪的 SOTA , 可以非常好的保留图像的结构和细节  

BM3D 主要用于去除图像中的 加性高斯白噪声 (Additive White Gaussian Noise, AWGN)


主体思想: 自然图像中本身有很多相似的重复结构
* 图像块匹配的方式来对相似结构进行收集聚合, 然后对图像块进行正交变换, 得到稀疏表示
* 充分利用稀疏性和结构相似性, 进行滤波处理

BM3D整体步骤:  用到了 变换域硬阈值滤波 和 维纳滤波


# 3. Noise in image / signal

噪音, 广泛出现于图像中, 源于图像传感器的物理特性:
* 指经过该设备后产生的原信号中并不存在的无规则的额外信号
* 噪音信号并不随原信号的变化而变化


噪音的分类:
* 基于频率的分类: 高中低频
* 基于时态的分类: fix pattern noise, temporal noise
  * fix pattern noise (FPN) : 与时间无关的噪音, 即噪音幅度不随时间而变化, 固定噪音, 与设备和信号值本身相关, 因此又称相关噪声
    * 可能是 sensor 的物理缺陷, 例如 hot pixel, weak pixel, dead pixel
  * temporal noise (TN) : 随时间变化的噪音, 不稳定的噪音 (暗光环境下录制视频即可看到不断变动的细小噪音), 从视觉上来看, 一般都是高频噪音, 因为与设备无关所以称为不相关噪声

1. 通过多帧合成降低 TN
2. 多帧平均的理论上限是将噪声降至只有 FPN
3. 通过多帧平均来分离出 FPN, 再通过其他手段去除 FPN


* 平场照片 flat-field : 在镜头前盖白色布, 以日光灯为光源连拍
  * 光学系统的渐晕会导致照片的暗角
  * 镜头上的灰尘污渍会造成图片上的黑斑
  * 平场照片用于表示整个光学系统综合的透过函数, 即光源进入 CMOS 前会经过的一个减光模板
  * 因此要想获得正确的图像, 合理利用 flat-field 图像, 将 CMOS获得的信号值除以 减光模板的透光率即可

* SIGNAL-NOISE RATIO 信噪比 (SNR, S/N)  
  * 指一个电子设备或者电子系统中信号与噪声的比例


## 3.1. Noise Modeling 噪音的建模与计算

均值和方差是比较基础的噪音统计方法, 均值用于 FPN, 方差用于 TN
* 均值  : $\mu = \frac{\sum_{i=1}^n X_i}{n}$
* 方差  : $\sigma^2 = \frac{1}{n}\sum_{i=1}^n(u-x_i)^2$


* 通过对纯黑内容 (black) 进行拍照, 可以对 FPN 噪音建模, 根据曝光时间, FPN噪音的幅度会逐渐增大, 注意 FPN 没有方差

* 通过对平场 (flat field) 照片进行拍照与统计, 可以对 TN 进行建模, 随着曝光时间增长, 噪音幅度也会增强, 但因为平场图像是白色, 过曝会导致图像过饱和, 在曲线上看噪音会随着曝光时间增强再减弱最终噪音消失

* 通过对 Gray Scale Chart 进行拍照并统计, 可以得到 "噪声随着亮度的增加而增加, 而噪声的标准差与亮度均值有一定的函数关系" 的结论


### 3.1.1. Poissonian-Gaussian Model

由噪音处理领域的神 Alessandro Foi 于 2008 年提出  

一种对图像噪音进行建模的方法, 对于相关噪声(signal-dependent) 用泊松分布拟合, 不相关噪声(signal-independent) 用高斯分布拟合  
其核心思想是, 相关噪声的标准差是信号强度的一个函数  

公式前提:
* 这里令 $x \in X$ 代表二维图像上的坐标, 有 $z(x)$ 代表观测信号, $y(x)$ 代表原始纯净信号
* 公式上, $E()$ 表示期望, $std()=\sqrt{var()}$ 表示标准差和方差
* $\xi(x)$ 代表一个随机正态分布, 0-means, 1-std   

标准噪声建模定义
* 相关噪声的标准差$std(z(x))$是信号强度$y(x)$ 的一个函数, 定义为 $std(z(x))=\sigma(y(x))$
* 相关/不相关噪声都满足以原始信号$y(x)$为均值的随机分布, $E(z(x))=y(x)$
* 则有噪信号可以表示成：
  * $z(x)=y(x)+\sigma(y(x))\xi(x)$
  * 后半部分代表对噪声整体的模型

Poissonian-Gaussian Modeling:
* $\sigma(y(x))\xi(x) =\eta_p(y(x))+\eta_g(x)$
* 这里$\eta_p,\eta_g$ 代表泊松分布和高斯分布的关联函数
* 泊松分布的部分, 可以套入为 
  * $\chi(y(x)+\eta_p(y(x))) \sim P(\chi y(x))$ 
  * 其中$\chi$ 是一个用于修正的实数常量
  * 如果不加一个修正系数的话, 根据泊松分布的特性, 噪音的分布就直接等于 $y(x)$ 了
* 高斯分布的噪声则可以直接表示
  * $\eta_g(x)\sim N(0,b)$
  * 这里 b 则是描述高斯部分的标准差
* 由于泊松分布的 方差和期望相等的特性
  * 方便书写, 这里令 $fun_\chi = \chi(y(x)+\eta_p(y(x)))$
  * $E(fun_\chi) = var(fun_\chi) = \chi y(x)$ 
* 根据常量部分的特性:
  * $E(\chi(y(x)))=\chi(y(x))$
  * $var(\chi(y(x)))=0$
* 得到:
  * $E(\eta_p(y(x)))=0$
  * $var(\eta_p(y(x)))=y(x)/\chi$
  * 即, 泊松分布部分的方差可以表示成 $ay(x), a=1/\chi$
* 再综合高斯分布噪音
  * 高斯分布有着常量的方差
  * $var(\eta_g(y(x)))=b$
  * 整体下来, 噪音的方差可以表示为
  * $\sigma^2(y(x))=a(y(x))+b$
  * $std(z(x))=\sigma(y(x))=\sqrt{a(y(x))+b}$
* $y(x)$ 是标准后的原始信号强度 
  * $y(x)\in[0,1]$

#### 3.1.1.1. Raw-Data Poission-Gaussian Modeling

实际的Sensor数据有两个额外操作:
* 信号放大  : 由于光感元件的电流很弱, 所以模拟信号放大是必须的
  * 模拟信号放大会重新引入第二个高斯噪声
* 基座电流  : 为了保证一丁点光线都没有的情况下, Sensor仍然有信号输出, 需要给Sensor上加一点偏置电流
  * 而这部分是电子元件给予的固定值, 不会影响泊松分布的结果 (泊松分布影响参数是光子)

基座电流(Pedestal)模型修正:
* $z(x)=y(x)+\sigma(y(x)-p_0)\xi(x)$
  * $=y(x)+\eta_p(y(x)-p0)+\eta_g(x)$
  * 高斯分布的噪声不受信号值的影响

模拟信号放大(Analog Gain)模型修正:
* 将到现在为止的模型重新指定为未放大的原始信号 $\mathring{z}(x)$
* $z(x)=\Theta(\mathring{z}(x))=\theta(\mathring{y}(x)+\mathring{\eta_p}(\mathring{y}(x)-p_0)+\mathring{\eta_g'(x)})+\mathring{\eta_g''}(x)$
* 根据分布的性质, 可以重新求得放大后的期望和方差
  * $E(z(x))=\theta\mathring{y}(x)$
  * $var(z(x))=\theta^2\chi^{-1}(\mathring{y}(x)-p_0)+\theta^2var(\mathring{\eta_g'(x)})+var(\mathring{\eta_g''}(x))$
* 根据 $y(x)=\theta\mathring{y}(x)$ 以及模型的最终方差表达式 $var=ay(x)+b$, 有：
  * $a=\frac{\theta}{\chi}$
  * $b=\theta^2var(\mathring{\eta_g'(x)})+var(\mathring{\eta_g''}(x))-\theta^2\chi^{-1}p_0$
  * 根据实际的特性, 就算pedestal比较大导致 $b<0$, 也不会出现方差小于0这种情况


(异方差的)泊松分布的高斯逼近:
* 在实际应用情况下, 对于期望是$\lambda$的泊松分布, 可以用 $N(\lambda,\lambda)$的高斯分布来替代泊松分布, $\lambda$越大逼近效果越准确
* 回归到最早的式子 $z(x)=y(x)+\sigma(y(x))\xi(x)$
* $\sigma(y(x))\xi(x)=\sqrt{{ay(x)+b}}\xi(x)\sim\eta_h(y(x))$
* $\eta_h(y(x))\sim N(0,ay(x)+b)$ 是最终的异方差的噪音高斯模型


#### 3.1.1.2. Poissonian-Gaussian Modeling Algorithm

从一个有噪声的图片中完成相关噪声的建模

根据 Alessandro Foi 在论文中的定义, 完整算法可以分成两大阶段
1. local estimation of multiple expectation/standard-deviation pairs
2. global parametric model fitting to local estimates

在正式的预测算法之前, 需要对二维图像进行预处理, 包括 小波变换和 基于小波域的分割


**A. Wavelet Domain Analysis:**




### 3.1.2. Denoising 




## 3.2. Filter Denoise


常用的降噪方法:
* wiener filter
* dct
* BM3D
* wavelet-based
* nlmeans
* Bilateral filter

自适应滤波: 近年以来发展起来的一种最佳滤波方法








### 3.2.1. spatial denoise  空间降噪

各种基于卷积的滤波器

#### 3.2.1.1. mean filter  平均滤波降噪

### 3.2.2. transform denoise 频域降噪

傅里叶变换, 小波变换

## 3.3. Ringing Artifect 振铃效应

振铃效应起源于信号传输, 是由于传输线组成的散杂电容导致的信号模糊

在CV领域, Ringing Artifacts 是影响复原图像质量的因素之一, 主要源于图像复原中不恰当的模型, 使得图像中的高频信息丢失  
* 振铃表现为 : 图像灰度剧烈变化处 (edge), 产生了类似钟敲击后的震荡

# 4. HDR high dynamic range

高动态范围:
* HDRI (High Dynamic Range Imaging) 高动态范围成像
  * 目的是正确的表示真实世界中 从太阳光直射到最暗的阴影的大亮度范围
  * 用来实现比普通数位图像技术更大曝光动态范围 (的一组技术)
  * 最早的提出论文是利用多张不同曝光的图片来进行HDR合成, 后来也引申出来了多种技术

目前一些数码相机厂商也开始研发 HDR 技术:
* 富士    : SuperCCD SR
* 佳能    : 高光优先模式
* 尼康    : Active D-Lighting
* 索尼    : D-Range Optimizer

传统数字图像通常会根据人眼视觉系统进行编码, 即 gamma 编码, 数据值和实际显示的亮度是非线性的, 用以适应人眼对于光敏感度的非线性性  

HDR 照片存储的图像数据是 真实世界可以观察到的亮度(luminance) 或者radiance值, 即固定 gamma = 1, 因此需要更多的数据位来保存从 10^-4 到10^8甚至更高的亮度范围, 一般的存储格式是 16 位或者 32 位浮点数, 也有经过优化后的 10~12 位 

从HDR数据到实际显示技术的映射即 HDR tone mapping

## 4.1. HDR 的实现要求


手机摄像头传感器的缺点:
* 光圈较小 : 光子收集量不多, 倒是暗光场景下的高噪声
* 每像素的底片面积较小 : 电子存储量不够多, 导致照片的动态范围无法很大

基础的暗光场景拍照处理方法有两种:
* 引入模拟信号放大 : 会导致噪音也一并放大
* 提高曝光时间 : 更容易因为相机的抖动或目标的移动而产生 motion blur

同样的, 高亮场景下也会导致局部暗部场景进光不足:
* 为了防止高光部分过曝, 因此会限制曝光时间, 导致暗部细节不够
* 若是使用 local tone-mapping, 会类似于直接放大一样导致噪音被一并放大

目前, 对于提高进光量的基础解决方法有以下几种
* 加大设备的光圈
* 光学防抖 : 可以降低相机抖动, 但是对于目标的移动仍然无法解决 blur 
* 闪光灯
* 包围曝光 (exposure bracketing) : 是一次拍摄后, 以中间曝光值和减少曝光值和增加曝光值的方式, 形成3张或者5张不同曝光量的照片. 容易导致细节丢失, 同时使得去马赛克变难


## 4.2. Multi-Frame Noise Reduction 基础的多帧降噪

对于相同内容的多帧图像 $S_n$
* 信号总量 $S=S_1+S_2+...+S_n$
* 噪音方差 $\sigma^2_t=\sigma^2_{t1}+\sigma^2_{t2}+...+\sigma^2_{tn}$
* 信噪比 $SNR = \frac{S}{\sigma_t}=\frac{S}{\sqrt{\sigma^2_{t1}+\sigma^2_{t2}+...+\sigma^2_{tn}}}$
* 由于原始内容相同:
  * $SNR=\frac{nS_1}{\sqrt{n\sigma_{t1}}}=\sqrt{n}\frac{S_1}{\sigma_{t1}}$
  * 即多帧合成可以开平方速率降低 TN


## 4.3. Image Fusion

多帧融合提高图像动弹范围或降噪
* 容易由于目标的移动导致的鬼影问题 (ghosting)
* 多帧图像无法正确的对齐内容 (由于拍摄时间的不同)
