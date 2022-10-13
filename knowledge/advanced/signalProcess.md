- [1. Signal Process](#1-signal-process)
- [2. Noise](#2-noise)
  - [2.1. Noise Classification](#21-noise-classification)
    - [2.1.1. 热噪声 Thermal Noise](#211-热噪声-thermal-noise)
    - [2.1.2. 散粒噪声 Shot Noise](#212-散粒噪声-shot-noise)
    - [2.1.3. 1/f 噪声 Fliker Noise](#213-1f-噪声-fliker-noise)
  - [2.2. Noise in Image Sensor](#22-noise-in-image-sensor)
    - [2.2.1. Image Noise Modeling 图像噪音的建模](#221-image-noise-modeling-图像噪音的建模)
      - [2.2.1.1. Poissonian-Gaussian Model](#2211-poissonian-gaussian-model)
      - [2.2.1.2. Raw-Data Poission-Gaussian Modeling](#2212-raw-data-poission-gaussian-modeling)
      - [2.2.1.3. Poissonian-Gaussian Modeling Algorithm](#2213-poissonian-gaussian-modeling-algorithm)
    - [2.2.2. Variance-Stabilizing Transformation VST](#222-variance-stabilizing-transformation-vst)
  - [2.3. Noise Evaluation](#23-noise-evaluation)
    - [2.3.1. Mean Squared Error MSE](#231-mean-squared-error-mse)
    - [2.3.2. Peak Signal-to-Noise Ratio (PSNR)](#232-peak-signal-to-noise-ratio-psnr)
    - [2.3.3. Structural Similarity (SSIM)](#233-structural-similarity-ssim)
  - [2.4. Ringing Artifect 振铃效应](#24-ringing-artifect-振铃效应)
- [3. Filter](#3-filter)
  - [3.1. Filter in signal process](#31-filter-in-signal-process)
  - [3.2. Local Linear Filter](#32-local-linear-filter)
    - [3.2.1. Convolutional Filter and Fourier Transform](#321-convolutional-filter-and-fourier-transform)
    - [3.2.2. Deconvolution](#322-deconvolution)
    - [3.2.3. Blur Using Local Linear Filter](#323-blur-using-local-linear-filter)
      - [3.2.3.1. Box Filter / Mean Filter](#3231-box-filter--mean-filter)
      - [3.2.3.2. Gaussian Filter](#3232-gaussian-filter)
    - [3.2.4. Edge Detection Using Local Linear Filter](#324-edge-detection-using-local-linear-filter)
      - [3.2.4.1. Sobel operator 索贝尔算子](#3241-sobel-operator-索贝尔算子)
      - [3.2.4.2. Prewitt operater 普利维特算子](#3242-prewitt-operater-普利维特算子)
      - [3.2.4.3. Laplacian Operator 拉普拉斯算子](#3243-laplacian-operator-拉普拉斯算子)
      - [3.2.4.4. Roberts Cross operator 罗伯茨交叉边缘检测](#3244-roberts-cross-operator-罗伯茨交叉边缘检测)
    - [3.2.5. Gabor Feature Extractor](#325-gabor-feature-extractor)
  - [3.3. Local Non-linear Filter](#33-local-non-linear-filter)
    - [3.3.1. Median Filter 中值滤波](#331-median-filter-中值滤波)
    - [3.3.2. Anisotropic Diffusion Filter  各向异性扩散滤波](#332-anisotropic-diffusion-filter--各向异性扩散滤波)
    - [3.3.3. Bilateral Filter 双边滤波](#333-bilateral-filter-双边滤波)
    - [3.3.4. Bilateral Grid 基于双边滤波的改进 Filter](#334-bilateral-grid-基于双边滤波的改进-filter)
    - [3.3.5. Guided Filter](#335-guided-filter)
      - [3.3.5.1. Weighted Guided Image Filter](#3351-weighted-guided-image-filter)
      - [3.3.5.2. Gradient Domain Guided Image Filter](#3352-gradient-domain-guided-image-filter)
      - [3.3.5.3. Multichannel Guided Image Filter](#3353-multichannel-guided-image-filter)
    - [3.3.6. Total Varation Denosing 总变差去噪](#336-total-varation-denosing-总变差去噪)
    - [3.3.7. AutoEncoder](#337-autoencoder)
    - [3.3.8. Local Binary Pattern](#338-local-binary-pattern)
  - [3.4. Global (Non-local) Filter](#34-global-non-local-filter)
    - [3.4.1. NL-means](#341-nl-means)
    - [3.4.2. Weighted Least Squares (WLS)](#342-weighted-least-squares-wls)
  - [3.5. Transform Domain Filter 变换域滤波](#35-transform-domain-filter-变换域滤波)
    - [3.5.1. wiener filter 维纳滤波](#351-wiener-filter-维纳滤波)
    - [3.5.2. Wavelet Threshold Denoise 小波阈值滤波](#352-wavelet-threshold-denoise-小波阈值滤波)
    - [3.5.3. BM3D  Block-matching and 3D filtering](#353-bm3d--block-matching-and-3d-filtering)
- [4. Denoise](#4-denoise)
  - [4.1. Image Denoise](#41-image-denoise)
    - [4.1.1. 滤波 Filters 图像降噪](#411-滤波-filters-图像降噪)
    - [4.1.2. Sparse Representation 稀疏表达](#412-sparse-representation-稀疏表达)
      - [4.1.2.1. K-SVD 算法 与字典学习](#4121-k-svd-算法-与字典学习)
    - [4.1.3. Low Rankness 聚类低秩](#413-low-rankness-聚类低秩)
- [5. Image Signal Processing (ISP)](#5-image-signal-processing-isp)
  - [5.1. Image Pipeline 的操作分区](#51-image-pipeline-的操作分区)
  - [5.2. Image Pipeline 的各种操作及其缩写](#52-image-pipeline-的各种操作及其缩写)
  - [5.3. Gamma](#53-gamma)
  - [5.4. Raw 图像](#54-raw-图像)
  - [5.5. Macbeth Chart](#55-macbeth-chart)
  - [5.6. calibration](#56-calibration)
  - [5.7. lens shading](#57-lens-shading)
- [6. Information Theory](#6-information-theory)
  - [6.1. Random Processes 随机过程](#61-random-processes-随机过程)
    - [6.1.1. 随机过程的概率模型](#611-随机过程的概率模型)
    - [6.1.2. Power Spectral Density (PSD)](#612-power-spectral-density-psd)
      - [6.1.2.1. 均方根 Root Mean Square (RMS)](#6121-均方根-root-mean-square-rms)
      - [6.1.2.2. 谱线 - 对信号的测量](#6122-谱线---对信号的测量)
      - [6.1.2.3. 信号的能量分类](#6123-信号的能量分类)
      - [6.1.2.4. 谱密度 能量谱密度](#6124-谱密度-能量谱密度)
      - [6.1.2.5. 自功率谱  Self Power Spectrum](#6125-自功率谱--self-power-spectrum)
      - [6.1.2.6. 功率谱密度的计算](#6126-功率谱密度的计算)
- [7. 图像复原](#7-图像复原)
- [8. Filter](#8-filter)
  - [8.1. Filter Denoise](#81-filter-denoise)
    - [8.1.1. spatial denoise  空间降噪](#811-spatial-denoise--空间降噪)
      - [8.1.1.1. mean filter  平均滤波降噪](#8111-mean-filter--平均滤波降噪)
    - [8.1.2. transform denoise 频域降噪](#812-transform-denoise-频域降噪)
- [9. HDR high dynamic range](#9-hdr-high-dynamic-range)
  - [9.1. HDR 的实现要求](#91-hdr-的实现要求)
  - [9.2. Multi-Frame Noise Reduction 基础的多帧降噪](#92-multi-frame-noise-reduction-基础的多帧降噪)
  - [9.3. Image Fusion](#93-image-fusion)

# 1. Signal Process

信号处理杂记, 由于波形信号和图像信号的共同点相当多, 该笔记里不分类 


# 2. Noise

噪音, 广泛出现于各种信号, 图像中, 源于图像传感器的物理特性:
* 指经过该设备后产生的原信号中并不存在的无规则的额外信号
* 不相关噪声噪音信号并不随原信号的变化而变化

本章尽量不涉及任何降噪的方法, 只讨论噪声本身的模型

## 2.1. Noise Classification

噪音的多种分类方法:
* 基于频率的分类: 高中低频
* 基于时态的分类: fix pattern noise, temporal noise
  * fix pattern noise (FPN) : 与时间无关的噪音, 即噪音幅度不随时间而变化, 固定噪音, 与设备和信号值本身相关, 因此又称相关噪声
    * 可能是 sensor 的物理缺陷, 例如图像中的 hot pixel, weak pixel, dead pixel
  * Temporal noise (TN) : 随时间变化的噪音, 不稳定的噪音 (暗光环境下录制视频即可看到不断变动的细小噪音), 从视觉上来看, 一般都是高频噪音, 因为与设备无关所以称为不相关噪声

均值和方差是比较基础的噪音统计方法, 均值用于 FPN, 方差用于 TN
* 均值  : $\mu = \frac{\sum_{i=1}^n X_i}{n}$
* 方差  : $\sigma^2 = \frac{1}{n}\sum_{i=1}^n(u-x_i)^2$


加性高斯白噪声 (Additive White Gaussian Noise, AWGN)
* 加性 Additive : 噪声对原始信号的影响为线性的, 即信号无关噪声, i.i.d. 区别于加性噪声的称为散粒噪声, 散粒噪声于信号成相关的泊松分布.
* 高斯 Gaussian : 噪声瞬时值服从高斯分布
* 白噪声 White  : 噪声功率谱密度服从均匀分布, 类似于白光包含了可见光的所有频率. 白噪声也包含了所有频率的波, 并且在各个频率上的功率谱密度都是一样的, 即白噪声的强度不会与像素的颜色相关
* 研究表明, 在低照度和高温的情况下, AWGN 占据了图像噪声的主要部分, 因此很多研究都是基于 AWGN 的

### 2.1.1. 热噪声 Thermal Noise

热噪声属于白噪声, 且
* 理论上 覆盖整个频率范围  
* 实际测量上 噪声等效带宽 B 决定了频率范围
* 实际操作中 将 电路的电压增益平方 等效带宽 定义为 热噪声的等效带宽
  * 电压增益 : 输入电压与输出电压之比, 理想的电压增益是不随电流频率而变化的, 但是实际上随着电流频率上升, 由于电路中电容的存在导致电压增益曲线下降  
  * 没有增益, 就没有信号, 自然就没有噪音, 因此热噪声在有噪信号中并不会存在于所有频率  

### 2.1.2. 散粒噪声 Shot Noise

散粒噪声属于白噪声, 是由 电子本身的离散特性 导致的  
* 细化来讲, 散粒噪声可以由 暗电流, 和 随即光生电子 产生, 具体是什么不清楚
* 粒子 (光子和电子) : 在传感器中发射的概率服从泊松分布  


### 2.1.3. 1/f 噪声 Fliker Noise

任何不同物质的接触面 (包括导体-导体, 导体-半导体, 半导体-半导体等) 在导电率上都会出现波动, 而放大电路中该波动就会引起 1/f 噪声  


在低频时 1/f 噪声是系统的主要噪声, 而高频时候的噪声则会降低到不如 热噪声, 即高频的时候热噪声是主要噪声成分  


## 2.2. Noise in Image Sensor

图像传感器: 将光子转化成电子, 再转换成电压, 途中通过放大器, 最终变成离散的光强数值
* 光传感器基本上维持着: 信号电荷数量随光照强度的增强而增加  

图像传感器噪声: 注意与图像噪声不同, 图像传感器噪声所描述的更多的是噪声的物理硬件由来, 实际上, 图像传感器噪声是图像噪声的主要来源, 了解了噪声的来源才能更清醒的进行降噪  

图像传感器的噪声可以分为两种
* 模式噪声  : 帧与帧之间不发生明显变换, 通过帧间平均无法进行抑制
  * 固定模式噪声 Fix Pattern Noise (FPN)
    * 和 传感器尺寸, 掺杂浓度, 制造过程中的污染, 晶体管的性质等有关  
    * 通常在没有光照的条件下进行测量, 
  * 光照响应非均匀性 Photo-Response Non-Uniform (PRNU) 
    * 和光照有关, 传感器尺寸, 掺杂浓度, 覆盖层厚度, 光照波长 等都会影响, 但是讨论较少  
* 随机噪声  : 随机的, 帧间不同, 可以通过统计分布来描述表达, 可以通过帧间平均来进行降噪, Temporal noise (TN) 
  * 热噪声 Thermal Noise
  * 散粒噪声 Shot Noise
  * 1/f 噪声 Flicker Noise
  * 以上三种噪声是光传感器的最基本的噪声, 初次之外还有 重置噪声 Reset Noise 和 本底噪声 Noise Floor
  * 一般随机噪声的平均都是0, 因此描述随机噪声一般都只用方差  (或标准差)


### 2.2.1. Image Noise Modeling 图像噪音的建模


* 通过对纯黑内容 (black) 进行拍照, 可以对 FPN 噪音建模, 根据曝光时间, FPN噪音的幅度会逐渐增大, 注意 FPN 没有方差

* 通过对平场 (flat field) 照片进行拍照与统计, 可以对 TN 进行建模, 随着曝光时间增长, 噪音幅度也会增强, 但因为平场图像是白色, 过曝会导致图像过饱和, 在曲线上看噪音会随着曝光时间增强再减弱最终噪音消失

* 通过对 Gray Scale Chart 进行拍照并统计, 可以得到 "噪声随着亮度的增加而增加, 而噪声的标准差与亮度均值有一定的函数关系" 的结论

几种噪声模型都只讨论 TN, 即随机噪声  

不同论文中对图像噪声描述的公式的字母不太一样, 这里笔记中使用统一为 Alessandor 的描述方法>:
* 这里令 $x \in X$ 代表二维图像上的坐标, 有 $z(x)$ 代表观测信号, $y(x)$ 代表原始纯净信号
* 公式上, $E()$ 表示期望, $std()=\sqrt{var()}$ 表示标准差和方差
* $\xi(x)$ 代表一个随机正态分布, 0-means, 1-std   

标准随机噪声建模定义, 其核心思想是, 噪声的标准差是信号强度的一个函数
* 噪声的标准差$std(z(x))$是信号强度$y(x)$ 的一个函数, 定义为 $std(z(x))=\sigma(y(x))$
* 相关/不相关噪声都满足以原始信号$y(x)$为均值的随机分布, $E(z(x))=y(x)$
* 则有噪信号可以表示成：
  * $z(x)=y(x)+\sigma(y(x))\xi(x)$
  * 后半部分代表对噪声整体的模型
#### 2.2.1.1. Poissonian-Gaussian Model

由噪音处理领域的神 Alessandro Foi 于 2008 年提出, 是21世纪后续图像噪声建模的基础  

一种对图像噪音进行建模的方法, 对于相关噪声(signal-dependent) 用泊松分布拟合, 不相关噪声(signal-independent) 用高斯分布拟合  


推导过程:   
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

#### 2.2.1.2. Raw-Data Poission-Gaussian Modeling

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


#### 2.2.1.3. Poissonian-Gaussian Modeling Algorithm

从一个有噪声的图片中完成相关噪声的建模

根据 Alessandro Foi 在论文中的定义, 完整算法可以分成两大阶段
1. local estimation of multiple expectation/standard-deviation pairs
2. global parametric model fitting to local estimates

在正式的预测算法之前, 需要对二维图像进行预处理, 包括 小波变换和 基于小波域的分割


**A. Wavelet Domain Analysis:**



### 2.2.2. Variance-Stabilizing Transformation VST

方差稳定化转换 (VST) 是一类统计学上的 数据变形 操作, 用于将 参数分布 的数据方差与数据均值关联的特性减轻, 来方便建模或者其他操作  
* 代表性的分布有 泊松分布 和 二项分布  

即使是很简单的 $y=\sqrt{(x)}$ 对于降低泊松分布的方差期望相关性都有很好的效果  

该类转换的代表性研究是 Anscombe transform  

一般来说, 如果某个分布的 方差和期望的关系已经被函数化建模  $var(X)=h(\mu)$, 那么 一个具有可接受偏差的 VST 可以表示为, 根据具体实际情况还会加上标量偏移以及 缩放因子    
$$y\propto \int^x \frac{1}{\sqrt{h(\mu)}}d\mu$$


例1 : $h(\mu)=s^2\mu^2$, 即标准差和期望成比例, 则有 VST  
$$y=\int^x\frac{1}{\sqrt{s^2\mu^2}}d\mu=\frac{1}{s}\ln(x) \propto \log(x)$$


例2 : $h(\mu)=\sigma^2+s^2\mu^2$, 即在期望较小的时候, 有一个基础方差 sigma, 此时有 VST  



## 2.3. Noise Evaluation

用于评价降噪效果的评价指标, 通常需要一个无信号 niose-free 的原始信号来计算, 一次很多情况只能利用合成的噪声信号来进行评价

### 2.3.1. Mean Squared Error MSE

最基础的信号领域评价指标

假设对于一个单色的 m×n 图像, 其纯净信号为K, 噪音信号为 I, MSE定义为:
$$MSE=\frac{1}{mn}\sum^{m-1}_{i=0}\sum^{n-1}_{j=0}[I(i,j)-K(i,j)]^2$$

### 2.3.2. Peak Signal-to-Noise Ratio (PSNR)


PSNR 通常用于量化经受有损压缩的图像和视频的重建质量, 因为 PSNR 用的是平方, 而一般信号都有较大的 动态范围, 因此 会使用类似分贝的指标 (即 log 化) 来计算 PSNR

The ratio between the maximum possible power of a signal and the power of corrupting noise
* 信号可能的最大值的平方除以 MSE (平均噪音平方)
* 这里 n 是数字化表达信号的时候信号的 bit 数 (动态范围)
* 对于一般有颜色的多通道图像, 没有什么其他的不同, 在计算 MSE 的时候是计算所有像素点的整体的 MSE, 即 要除以 3mn
* 对于一些特殊的颜色格式 HSL or YCbCr, 存在 PSNR 分别在不同通道上计算的情况


$$PSNR=10\times \log_{10}(\frac{MAX_I^2}{MSE})=10\times \log_{10}(\frac{(2^n-1)^2}{MSE})$$  

$$PSNR=20\times \log_{10}(\frac{MAX_I}{\sqrt{MSE}})=20\log_{10}(MAX_I)- 10\log_{10}(MSE)$$

### 2.3.3. Structural Similarity (SSIM)

于 2004 年由 Zhou, Bovik, Sheikh, Simoncelli 提出  

SSIM 用于衡量两张图片的相似度, 对于不同的图片, 一般会在 N×N 的 Windows 上计算而不是整张图片  
* 在超分辨率和图像去模糊上都有很广泛的应用

SSIM 是结构相似性理论的一种实现, 对于图片所具有的属性: 亮度, 对比度, 物体结构
* 人类对于像素的绝对亮度/颜色不敏感, 但是对于边缘和纹理的位置胃肠敏感, SSIM 通过主要关注边缘和纹理的相似性来模仿人类的感知  
* 均值 用于亮度的估计
* 标准差 是对比度的估计
* 协方差 是结构相似程度的度量
* SSIM 是图片三种属性的结合比较

定义为:
* 对于 x,y 的两张图片
* $\mu$ 是平局值 , $\sigma^2$ 是方差
* $\sigma^2_{xy}=E[(x-E[x])(Y-E[Y])]=E[xy]-E[x]E[y]$ 是两张图片的协方差  
* 定义 $l(x,y)$ 为 luminance 即亮度的相似性, 亮度差异越大结果越接近0, 亮度越相似, 结果越接近1. 该参数是尺度不变的, 即图像乘以常数不会影响结果, 因此对于标准化前后的结果都相同
$$l(x,y)=\frac{2\mu_x\mu_y}{\mu_x^2+\mu_y^2}$$
* 定义 $c(x,y)$ 利用 patch 的方差, 实现对 contrast 对比度的相似性, 该值用于对比 patch 中纹理的数量, 如果一个 patch 比另一个 平坦 或者 纹理丰富 得多, 则结果越接近0, 否则结果为1, 该公式也是尺度不变的  
$$c(x,y)=\frac{2\sigma_x\sigma_y}{\sigma_x^2+\sigma_y^2}$$
* 定义 $s(x,y)$ 为结构 structure 的分数, 比较两个 patch 的相关性, 当两个 patch 有很多相同位置和方向的边缘时, 则分会很高  
$$s(x,y)=\frac{\sigma_{xy}}{\sigma_x\sigma_y}$$
* 最终的 SSIM 即该三个参数的乘积, 注意 $\sigma_x\sigma_y$ 被抵消掉了, 为了防止除以0, 有$c_1=(k_1L)^1,c_2=(k_2L)^2$ 是用来维持稳定的常数, L 是图像的动态范围, 而 k1=0.01, k2=0.03

$$SSIM(x,y)=\frac{(2\mu_x\mu_y+c_1)(2\sigma_{xy}+c_2)}{(\mu^2_x+\mu^2_y+c_1)(\sigma^2_x+\sigma^2_y+c_2)}$$



## 2.4. Ringing Artifect 振铃效应

振铃效应起源于信号传输, 是由于传输线组成的散杂电容导致的信号模糊

在CV领域, Ringing Artifacts 是影响复原图像质量的因素之一, 主要源于图像复原中不恰当的模型, 使得图像中的高频信息丢失  
* 振铃表现为 : 图像灰度剧烈变化处 (edge), 产生了类似钟敲击后的震荡




# 3. Filter

Filter 的概念原本起于数字信号处理, 后来被延申到图像处理, 因此有很多概念或者滤波名称都是沿用着数字信号处理中的名称  

Filter 的翻译是滤波器, 是一个处理的概念, 并不单单用于平滑图像或者降噪  

对于数字图像 (or Computer Vision) 领域来说, filter 的用途是用来将图像变得更好, 或者提取图像中的有用特征:
* 抑制不需要的特征    e.g. Denosing
* 强调有用的特征      e.g. Edge Detection, Feature Extraction
* 图像编码 (Encoder)  e.g. Local Binary Pattern, Increment Sign Correlation
* Detail smoothing/enhancement
* Colorization
* Image matting/feathering
* joint upsampling
* HDR compression 
* Multi-scale Decomposition
* Haze removal
* Image decovolution

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
* 变换域 Transform Domain Filter:
  * 小波阈值滤波

Edge-Perserving Filter 是图像处理的一个刚需, 用于保证边缘的同时对图片进行滤波
* Guided Filter
* Bilateral Filter
* (Weighted) Least Squares Filtering


Linear Translation-Invariant (LTI) filters. 线性的平移不变的 Filter.  
* 卷积核不随着空间位置而变化, 不随着图片的内容变化
* Gaussian filter
* Laplacian filetr
* Sober filter

Optimization-based filter approach, 通过优化一个二次函数, 该函数通过一个 Guidance image 来直接最位置的输出内容进行一些限制
* 计算时间长

Linear Translation-Variant Filters - Build the filter using the guidance image
* 定义输出图像 q 输入图像 p 以及 Guidance 图像 I, 图像坐标集 i,j
* 可以定义 LTV Filter 为 : $q_i=\sum_jW_{ij}(I)p_j$
* 其中 $W_{ij}(I)$ 是通过 Guidance 图像来计算 Filter 参数的函数, 它与输入图像 p 无关, 因此对于p 来说这是个线性变换
* Bilateral filter (有缺点 : gradient reversal artifacts) O(Nr^2)
* Guided filter O(N)

## 3.1. Filter in signal process

信号根据傅里叶分解, 信号可以拆分成一系列不同频率的正弦波的叠加, 再根据频率可以进行滤波

频率Filter的种类:
* Low-pass    : 低通
* High-pass   : 高通
* Band-pass   : 中通
* Band-stop   : Low-high-pass, 中间频率被阻断
* Low-band-pass: low-pass + band-pass, 高频和中低频被阻断, 中频和低频通过
* Band-high-pass: band-pass + high-pass
* Low-band-high-pass: 中低频和中高频被阻断

## 3.2. Local Linear Filter

局部线性 Filter : Convolutional Filter, Kernel 的形状和值是固定的

使用局部线性滤波的任务
* Blur, Denoising, Shapening
* Edge Detection, Feature Extraction

分离的 Filter:
* 对于对称的 `k*k` Filter, 可以将其分成两个一维的filter `1*k` `k*1` 
* 具体的效果是降低计算复杂度 $O(n*m*k*k) \rArr O(2*n*m*k)$

### 3.2.1. Convolutional Filter and Fourier Transform

对于卷积操作, 假设 $f,g,h$是 kernel 或者图像
* 存在结合律  : $(f*g)*h = f*(g*h)$  这里 $*$ 是卷积操作
* 存在卷积定理: $F(f*g)=F(f)*F(g)$ 这里 F 是 Fourier Transform

### 3.2.2. Deconvolution

根据卷积核 $f$ 和应用Filter后的图像 $f*g$, 还原出原始图像 $g$

同样利用傅里叶变化 F:
* 卷积定理: $F(f*g)=F(f)*F(g)$ 
* 有 $g=F_{inv}(F(f*g)/F(f))$

由于傅里叶变换的特性, 高频率的信息会丢失, 导致振铃效应 (ringing artifect)

### 3.2.3. Blur Using Local Linear Filter

使用简单的线性 Filter 可以很容易的实现模糊效果, 但通过模糊实现的降噪通常会导致图像的边缘也同时被模糊

#### 3.2.3.1. Box Filter / Mean Filter

最基础的卷积 Filter

* 一般的卷积操作的实际复杂度是 $O(n*m*k*k)$, k是 kernel size
* 利用累加的 Integral Image, 任意 kernel size 的 Box Filter 可以在 $O(n*m)$ 实现

#### 3.2.3.2. Gaussian Filter 

相比于 Box/Mean Filter, 模糊效果较为浅
* 利用二维高斯函数来生成 Kernel, 作为参数的方差 sigma 可以进行指定, 来调整 


### 3.2.4. Edge Detection Using Local Linear Filter

边缘检测算子某种程度上也可以当成 Feature Extractor

#### 3.2.4.1. Sobel operator 索贝尔算子

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

Sobel算子根据像素点上下, 左右邻点灰度加权差, 在边缘处达到极值这一现象检测边缘。对噪声具有平滑作用, 提供较为精确的边缘方向信息, 边缘定位精度不够高。当对精度要求不是很高时, 是一种较为常用的边缘检测方法。

#### 3.2.4.2. Prewitt operater 普利维特算子

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


Prewitt算子利用像素点上下, 左右邻点灰度差, 在边缘处达到极值检测边缘。`对噪声具有平滑作用`, 定位精度不够高。  
图像每一个像素的横向纵向灰度值通过`平方相加再开根号`的形式结合, 也可以直接`绝对值相加`, 计算出的灰度值要记得缩放颜色深度  

#### 3.2.4.3. Laplacian Operator 拉普拉斯算子

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

#### 3.2.4.4. Roberts Cross operator 罗伯茨交叉边缘检测

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


### 3.2.5. Gabor Feature Extractor

## 3.3. Local Non-linear Filter

局部非线性 Filter : 根据像素值, Filter Kernel 的值也会变化, 没有固定的数字描述

使用局部非线性滤波的任务 : Denoising, Feature Extraction

基本上, 好的降噪 Filter 都是基于非线性算子的


### 3.3.1. Median Filter 中值滤波

中值滤波的定义: $I'(u,v) = median{I(u+i,v+j)|(i,j)\in R}$ 这里的 R 是滤波核大小

更新像素值为临近像素值排列后的 `中值`  

### 3.3.2. Anisotropic Diffusion Filter  各向异性扩散滤波

将图像看作热量场?! 像素值看作热流, 根据当前像素和周围像素的关系, 来确定是否要向周围扩散, 如果某个邻域像素和当前像素差别较大, 代表这个邻域像素很可能是个边界, 那么当前像素就不向这个方向扩散了


推导上都是热力学上的内容, 比较复杂, 属于迭代行方法  

### 3.3.3. Bilateral Filter 双边滤波

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


### 3.3.4. Bilateral Grid 基于双边滤波的改进 Filter

于 SIGGRAPH 2007 被提出, 是从升维的思想拓展出来的通用化操作, 不单单可以用来降噪

使用 Bilateral Grid 的流程可以归纳为:
1. 下采样, 建立 Grid
2. 对 Grid 进行一些操作
3. 上采样回去
由于 Grid 的数据量较小, 因此可以实现很多实时效果  

Cross-bilateral filter: 建立 grid 时不使用原图的像素值, 而是使用其他图片 e.g. E(x,y), 但是执行卷积的时候仍然使用原图像素值    
$$\Gamma([x/s_s],[y/s_s],[E(x,y)/s_r])+=(I(x,y),1)$$


### 3.3.5. Guided Filter

最早于 2009 年由 Kaiming He 提出, 也是平滑图像保留边缘的流行算法    
The guided filter can perform as an edge-preserving smoothing operator like the popular bilateral filter.  
* 比 Bilateral 时间快  O(n), 计算时间和 Kernel Size 无关  
* 在边缘上的效果更好

算法推导
* 定义 $\omega_k$ 是以像素 k 为中心的窗口, 半径不影响计算
* 定义 guided filter 是一个输出图像 q 和guidance图像 I 的线性模型
  * $q_i=a_kI_i+b_k, \forall i\in\omega_k$
  * $(a_k,b_k)$ 是一个在$\omega_k$中固定的参数
* 定义$(a_k,b_k)$的最优解法的 loss, 即输出图像 q 和输入图像 p 的差值最小
  * $E(a_k,b_k)=\sum_{i\in\omega_k}((a_kI_i+b_k-p_i)^2+\epsilon a_k^2)$
  * 即 p 和 q 的差值, 加上限定 a 的大小的正则量
  * 根据线性回归解法, a,b 的最优解为
  * $a_k=\frac{\frac{1}{|\omega|}\sum_{i\in \omega_k}I_ip_i-\mu_{I,k}\mu_{p,k}}{\sigma_{I,k}^2+\epsilon}$
  * $b_k=\mu_{p,k}-a_k\mu_{I,k}$
  * 这里, mu 和 sigma^2 分为是图像在窗口 $\omega_k$ 的均值和方差
  * $\mu_{p,k}$ 则是输入图像 p 在窗口 $\omega_k$ 的均值
* 根据公式, 由于每个输出像素 $q_i$在不同窗口下都会输出一个结果, 那么最终的结果可以简单的使用为所有输出的均值
  * $q_i=\frac{1}{|\omega|}\sum_{k:i\in \omega_k}(a_kI_i+b_k)=\bar{a_i}I_i+\bar{b_i}$
* 写成 $q_i=\sum_jW_{ij}(I)p_j$ 即窗口权值和的形式的话
  * $W_{ij}(I)=\frac{1}{|\omega|^2}\sum_{k:(i,j)\in \omega_k}(1+\frac{(I_i-\mu_k)(I_j-\mu_k)}{\sigma^2+\epsilon})$
* 最终 $\epsilon$ 作为控制整个滤波效果的参数
  * 对于一个窗口内的像素, 如果方差远远小于 $\epsilon$ 则会被平滑
  * 反之则会保留原值

对于彩色图像, 原论文也给出了一个基础的RGB通道滤波实现, 在单通道时
$$a_k=\frac{\frac{1}{|\omega|}\sum_{i\in \omega_k}I_ip_i-\mu_{I,k}\mu_{p,k}}{\sigma_{I,k}^2+\epsilon}$$

那么对于彩色 I 图像, a_k 升级成了一个向量, 而原本的 $\sigma_{I,k}^2+\epsilon$ 则变成了 3x3的矩阵, 这里 $\Sigma$ 代表各个通道互相的协方差, U是单位矩阵, 为了和 Guided 图像符号 I 区别开, 这里便于理解, 只有 I 是多通道的, p 是单通道的, 若 p 也是多通道的, a,b都需要再加一维  
$$\vec{a_k}=(\Sigma_k+\epsilon U)^{-1}(\frac{1}{|\omega|}\sum_{i\in \omega_k}I_ip_i-\mu_{I,k}\mu_{p,k})$$

b 的计算从符号上保持不变, 但是乘法变成向量内积, 最终的 b 仍然是标量

$$b_k = \mu_{p,k}-a_k^T\mu_{I,k}$$

#### 3.3.5.1. Weighted Guided Image Filter

将标准化参数 $\epsilon$ 也升级成和图片相同大小的矩阵的形式, 保持了相同的计算复杂度

具体体现为
$$E(a_k,b_k)=\sum_{i\in\omega_k}((a_kI_i+b_k-p_i)^2+\frac{\epsilon}{\Gamma_{I,k}} a_k^2)$$

这里, $\Gamma_I$ 是一个边缘 Mask, 体现为边缘像素的值大于1, 平滑区域像素的值小于1, 最终反映在 epsilon 上就是平滑区域的模糊力度更大, 而边缘则相对更好的保留, $\Gamma$ 的计算方式为窗口方差与全像素窗口方差的比值和:
$$\Gamma_{I,k}=\frac{1}{N}\sum_{j=1}^N\frac{\sigma_{I,k}^2+\lambda}{\sigma^2_{I,j}+\lambda}$$



$\lambda$ 则是引入的又一个小常数, 总是在套娃的把常数替换成更琐碎的常数, $\lambda = (0.001 × L)^2$, L 则是图像数据的动态范围  

最终的 a,b 计算式可以表示为如下, a 的底数变了, b 则是保持一致. 

$$a_k=\frac{\frac{1}{|\omega|}\sum_{i\in \omega_k}I_ip_i-\mu_{I,k}\mu_{p,k}}{\sigma_{I,k}^2+\frac{\epsilon}{\Gamma_{I,k}}}$$

$$b_k=\mu_{p,k}-a_k\mu_{I,k}$$


#### 3.3.5.2. Gradient Domain Guided Image Filter

说是梯度域的GIF, 就是再次将 Guided 图像的窗口方差引入到 损失函数的正规化项中, 且保持了相同的计算复杂度. GDGIF 进一步加强了边缘区域和平滑区域的系数差, 使得GIF最初的参数 $\epsilon$ 的影响更加小了, 基本上不同值的滤波结果不会有太大影响

需要注意的是损失函数

$$E(a_k,b_k)=\sum_{i\in\omega_k}((a_kI_i+b_k-p_i)^2+\frac{\epsilon}{\Gamma_{I,k}} (a_k-\gamma_k)^2)$$

这里对比 WGIF, $\Gamma$ 函数也进行了升级, 对比原本的窗口方差$\sigma_{I_r,k}^2$ ,又乘上了一个固定的 33 窗口的方差 $\sigma_{I_3,k}^2$, 小常数 $\lambda$ 则保持不变同WGIF

$$\Gamma_{I,k}=\frac{1}{N}\sum_{j=1}^N\frac{\chi_{I,k}+\lambda}{\chi_{I,j}+\lambda}=\frac{1}{N}\sum_{j=1}^N\frac{\sigma_{I_3,k}\sigma_{I_r,k}+\lambda}{\sigma_{I_3,j}\sigma_{I_r,j}+\lambda}$$


新加入的 $\gamma$ 项是用来加强边缘保存效果的, $\mu_\chi$ 是被用于计算 $\Gamma$ 的 $\chi$ 矩阵的全局平均值. 整体体现为, 若是在边缘则 $\gamma = 1$, 在平缓区则 $\gamma =0$ 

$$\gamma_k = 1 - \frac{1}{1+e^{4\times \frac{\chi_k-\mu_\chi}{\mu_\chi-min(\chi)}}}$$

最终的计算式, 同样 b 也没有变化， 主要体现在 a 上 

$$a_k=\frac{(\frac{1}{|\omega|}\sum_{i\in \omega_k}I_ip_i-\mu_{I,k}\mu_{p,k})+\frac{\epsilon}{\Gamma_{I,k}} \gamma_k)}{\sigma_{I,k}^2+\frac{\epsilon}{\Gamma_{I,k}}}$$

$$b_k=\mu_{p,k}-a_k\mu_{I,k}$$
#### 3.3.5.3. Multichannel Guided Image Filter

首先在论文中的方程就直接将多通道考虑进去, 即令 m 为通道数, 则有各种损失计算项都是 $\sum_{j=1}^m$

首先将 GDGIF 的损失函数复述一遍
$$E(a_k,b_k)=\sum_{i\in\omega_k}((a_kI_i+b_k-p_i)^2+\frac{\epsilon}{\Gamma_{I,k}} (a_k-\gamma_k)^2)$$

对于 MGIF 来说, 将加入了多通道计算项, 同时把 $\frac{\epsilon}{\Gamma}$ 项集合为系数 $w$, 有损失函数 H:

$$H(a_k,b_k)=\sum_{i\in\omega_k}((\sum_{j=1}^ma_{kj}I_{ij}+b_k-p_i)^2+\sum_{j=1}^mw_{kj}(a_{kj}-\hat{\gamma_{kj}})^2)$$

$w$ 的计算式如下: $\epsilon=(0.001L)^2$ L 是数据范围, 两个$\lambda$ 是新定义的正则化系数, $\sigma_{I,kj}^2$ 是单通道的窗口方差， $E_k$ 是单通道所有像素的平均, $E_j$ 是单像素所有通道的平均. 整的来说, 在边缘上时 w 应该小于1 $\lambda$ 接近1 , 在平滑区域时 w 大于1, $\lambda$ 接近 0

$$w_{kj}=\frac{\lambda_1E_k(\sigma_{I,kj}^2)+\lambda_2E_j(\sigma_{I,kj}^2)}{\sigma_{I,kj}^2+\epsilon}$$

$$\gamma_{kj} = \frac{2}{1+e^{-\frac{\sigma^2_{I,kj}}{E_k(\sigma_{I,kj}^2)}}}-1$$

以上是总体的改进后的边缘检测, 考虑到在实际三通道色彩上的边缘不一定在某一通道上具有梯度, 或者相反的情况, 需要用某种方法综合三通道的梯度, 有  

$$\hat{\gamma_{kj}} = sgn(cov_{jp,k})\gamma_{kj}$$

这里 cov 是协方差, 具体为 I 图像的通道 j 与输入图像 p 的协方差, sgn 是一个信号跳变函数, 对于协方差大于0 输出 1, 协方差 小于 0 输出 -1

以上的全部就是损失函数的定义, 最终的 a,b 线性解则表示为以下  
* 对于每一个通道 j 都有单独的 a_kj, 总体上对于每个像素k, 有向量 $A_k=[a_{k1}, a_{k2},...,a_{km}]^T$
* $A_k=(C_{j1j2}+W)^{-1}(C_{jp}+\Psi)$,这里 $C_{j1j2},C_{jp},W,\Psi$ 都是带有 k 符号的, 即是对应单个坐标的
* $b_k=\mu_{p,k}-\sum_{j=1}^ma_{kj}\mu_{j,k}$ . b 的计算为所有通道都参与
* $C_{j1j2}$ 是 I 图像的各个通道之间的协方差矩阵在 k 位置的值, $m\times m$ 是 
* $W=diag(w_{k1},w_{k2},...,w_{km})$ 对角矩阵
* $C_{jp}=[cov_{1p},cov_{2p},...,cov_{mp}]$ 即 I 图像和滤波图像的协方差
* $\Psi = [w_{k1}\hat{\gamma_{k1}},w_{k2}\hat{\gamma_{k2},...,w_{km}\hat{\gamma_{km}}}]^T$

最终输出图像的单通道可以计算为 

$$Q(k)=\sum_{j=1}^m\bar{a_{kj}}I_{jk}+\bar{b_k}$$

### 3.3.6. Total Varation Denosing 总变差去噪

Total Varation Denosing / Total Variation Regularization / Total Variation Filtering  
是信号处理中常见的降噪方法, 于 1992年由L.I. Rudin, S. Osher和E. Fatemi提出, 又称 ROF 模型

思想: 一个含有噪声的信号相较于其未受噪声影响的信号, 会有较大的 总变差值, 即梯度的绝对值总和较大, 若能找到一个于原始信号相似且总变差较小的信号, 则可作为降噪结果  

定义总变差函数为 TV(y), 定义两个信号的相似程度为 L2 范数 $E(x,y)=\frac{1}{2}||x-y||^2_2=\frac{1}{2}\sum_n(x_n-y_n)^2$

则有 TV 降噪的最优化问题, 其中 x 为噪声信号, y 为降噪结果信号, lamda 为调整参数  
$$\underset{y}{min}E(x,y)+\lambda TV(y)$$

对于图像的二维信号, 则有其他的问题和解法:
* 二维图像的总变差定义不同, 且不可微分, 因此不适合用来求解
$$TV(y)=\sum_{m,n}||\nabla y_{m,n}||_2=\sum_{m,n}\sqrt{|y_{m+1,n}-y_{m,n}|^2+|y_{m,n+1}-y_{m,n}|^2}$$
* 修改后的定义也有 L1 范式下的总变差, 但是也不保证问题是凸优化问题, 因此不能够用通常的凸优化算法来求解, 目前常被使用的有
  * Bregman 布雷格曼方法 1966
  * ADMM  交替方向乘子法 2012
  * 原始-对偶算法 2004
$$TV(y)=\sum_{m,n}||\nabla y_{m,n}||_1=\sum_{m,n}(|y_{m+1,n}-y_{m,n}|+|y_{m,n+1}-y_{m,n}|)$$
* 初次之外还有高阶微分版本
* 双边总变差去噪 bilateral total variation , 2004 年提出


### 3.3.7. AutoEncoder

类似于无监督学习的卷积神经网络, 将卷积核认作 Filter

### 3.3.8. Local Binary Pattern 

## 3.4. Global (Non-local) Filter

### 3.4.1. NL-means

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

### 3.4.2. Weighted Least Squares (WLS)

WLS 是定义了一个全局的损失函数, 来通过优化损失函数实现滤波  


## 3.5. Transform Domain Filter 变换域滤波

利用正交变换, 在另一个领域或者维度上进行滤波处理 

正交变换: 是信号变化的一系列统称
* 傅立叶变换
* 离散余弦变换
* 小波变换
* 多尺度几何分析（超小波）

### 3.5.1. wiener filter 维纳滤波

维纳滤波是通信领域的通用基本滤波方法 (无论是语言信号还是图像信号), 由  Norbert Wiener 在 1942 年提出来的
* 又称 最小二乘滤波器 或 最小平方滤波器
* 维纳滤波的原理是 平稳随机过程的相关特性和频谱特性  
* 本质的操作是 : 使估计误差的均方值最小化 (估计误差: 期望响应与滤波器实际输出之差)
* 在目前的工程视角下, 由于无法获得正确的系数, 直接将含有噪音的信号输入维纳滤波系统的话, 对于数字图像的降噪效果一般
* 通过数学运算可以将降噪运算转化成一个 toeplitz 方程


维纳滤波:
* 


离散信号的维纳滤波:
* 给定长度为 N 的有噪信号 $x[0], x[1],...,x[N-1]$, 假设 x 是由噪音v和信号y构成 $x[n]=y[n]+v[n]$
* 维纳滤波通过 x 的信息, 对信号 y 做出一个估计 $\hat{y}$, 使得 $\epsilon=E((y[n]-\hat{y}[n])^2)$ 最小, 即 MSE 最小
* 最终维纳滤波的形式可以写成 $\hat{y}[n]=\omega^Tx[n]$ omega 是最终求得的权值向量  
* 定义 $h[n]$ 为滤波器的冲激响应, 则信号估计量可以写成 
  * $\hat{y}[n]=h[n]*x[n]=\sum_kh[k]x[n-k]$
  * 将 E 括号拆开
  * $\epsilon=E(y[n](y[n]-\hat{y}[n])-\hat{y}[n](y[n]-\hat{y}[n]))$
  * $\epsilon=E((y[n]-\sum_kh[k]x[n-k])^2)$
* 若想取得信号预测值 epsilon 最小时候的滤波器冲激响应 h
  * 方法1 对h 直接求偏导, 则可以将问题转化成优化问题
  * 方法2 利用正交原理: 若 epsilon 取得最小值, 则残差 $e[n]=\hat{y}[n]-y[n]$ 应该与 $x[n]$ 正交
    * $E((y[n]-\hat{y}[n])x[n-m])=0$
  * 两种方法得到的结论是相同的
  * $E(y[n]x[n-m])=E(\hat{y}[n]x[n-m])$
* 用相关函数改写上述结论
  * $r_{xy}[m]=r_{x\hat{y}}[m]$
* 用功率谱分析的知识将上述等式改写成 输入信号的自相关函数与冲激响应的卷积
  * $r_{x\hat{y}}[m]=h[m]*r_{xx}[m]$
  * $\sum_kh[k]*r_{xx}[n-k]=r_{xy[n]}$
  * 用矩阵的写法为 $R_{xx}h=R_{xy}$
  * $h=R_{xx}^{-1}R_{xy}$
* 带入到噪声信号中
  * 若信号与噪声独立 $R_{vy}=0$
  * 则有 $R_{xy}=R_{yy}$
  * 则有 $R_{xx}=R_{vv}+R_{yy}$
* 至此
  * 只要知道 噪声和信号各自的自相关函数, 就可以确定维纳滤波器的具体形式  
  * 


### 3.5.2. Wavelet Threshold Denoise 小波阈值滤波

由信号处理领域专家 Donoho 1995年 提出的在小波域对白噪声进行降噪的方法  

主要思想:
* 白噪声在小波的各个尺度中均匀分布, 但是相对于主要信号的系数比较小
* 通过一个阈值来将其分开来, 小于阈值的系数就直接归零, 大于阈值的系数保持不变
* 将阈值滤波后的变换域信号 反变换回原本的 空间域(或者时间域)信号


### 3.5.3. BM3D  Block-matching and 3D filtering

BM3D 是 Alessandro Foi 于 2007 年提出的, 发布在 TIP 期刊上, 目前仍然是 CP 系降噪的 SOTA , 可以非常好的保留图像的结构和细节  

BM3D 主要用于去除图像中的 加性高斯白噪声 AWGN  

主体思想: 自然图像中本身有很多相似的重复结构
* 图像块匹配的方式来对相似结构进行收集聚合, 然后对图像块进行正交变换, 得到稀疏表示
* 充分利用稀疏性和结构相似性, 进行滤波处理  


BM3D整体步骤:  用到了 变换域硬阈值滤波 和 维纳滤波
1. Block-Matching : 对每个参考块进行相似性匹配得到三维的组合
2. 3D-Transform   : 对其进行协同变换和滤波 
3. Aggregation    : 对各个参考块对应组合的滤波结果进行整合 
4. 将123步骤作为单次降噪, 执行两遍, 两遍的匹配标准, 权重等会有一些区别  
    * Step 1 : 协同滤波使用 硬阈值(Hard-thresholding) 的方式来去噪, 作为 基础估计 Basic Estimate
    * Step 2 : 使用 Step 1 的结果作为输入, 重新进行相似块匹配, 使用 经验维纳收缩 Emprical Wiener Shrinkage 的方法来进行降噪.
    * 根据基础估计图像的 协同变换系数的功率谱 和噪声的强度, 对原始有噪图像同样位置的 3D 块的协同变换系数进行收缩.
    * 对收缩后的系数进行反变换和整合, 得到最终的降噪结果.
    * 在整个过程中, Step 1 得到的基础估计只是作为得到收缩系数和更准确的相似块的辅助, 整个降噪操作实际还是在原始噪声图像上计算  


# 4. Denoise 

降噪, 所有应用于信号上的处理基本都可以称作 Filter, Filter 可以用于降噪以外的事情, 降噪和 Filter 部分重合的

因此 同 Filter 本身区别开来, 如果是可以用于不止降噪的别的方面的 Filter 在本章中只简述名字

## 4.1. Image Denoise

图像降噪是非常广的世界, 图像可以看作是 2维 信号  
* 多帧降噪 MFNR
* 单帧降噪 SFNR
* Demosaiced Image NR
* Raw Image NR

多帧的基础操作
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
  * 是用于评价降噪方法的主流指标


图像降噪 研究方向分类:
* Filter 滤波降噪
  * Spatial Domain Filters
  * Transform Domain Filter
  * Hybird Domain Filters
* 稀疏表达 Sparse Representation
  * K-SVD
  * Non-Local Sparse Models
  * Non-Local Centralized Sparse Representation (NCSR)
* 聚类低秩 Low Rankness
  * 低秩矩阵恢复, 一副清晰的自然图像其数据矩阵往往是低秩或者近似低秩的, 任何突兀的成分都会增加图像矩阵的秩
  * Nuclear Norm Minimization
  * Weighted Nuclear Norm Minimization
* 统计模型 Statistical Model
  * Hidden Markov Model  (HMM)
  * Gaussian Mixture Model 
* 深度学习 

### 4.1.1. 滤波 Filters 图像降噪

内容参照 Filter 章节


### 4.1.2. Sparse Representation 稀疏表达

基于稀疏性的信号处理是很常见的, 自然界中的信号低频居多, 高频部分基本上都是噪声. 

变换域降噪 : 把所有频率的波看作互相正交的向量, 恢复数据就是找到一组系数, 将各个向量进行权重相加.  

因此基于小波或者傅里叶做也可以算作稀疏表达的基矩阵, 表达系数往往只在低频上比较大, 此时对高频系数进行 (soft-)thresholding, 降低或者去掉高频分量, 就能实现降噪效果, 这也是频率域滤波的原理

而所谓稀疏表达 Sparse Representation, 就是在给定的超完备字典中用尽可能少的原子来表示信号
* 字典: 原子的排列的集合, 对于 `N*T` 的矩阵, 如果 T>N, 则该字典被称为过完备或者冗余字典. 
  * (线性代数的临近概念)
  * 假设列向量两两线性无关, 则 `N*N` 的矩阵秩为 N, 再增加列向量也不会提供额外的信息
* 原子: 信号的基本构成成分, 比如长度为 N 的列向量

对于一维离散信号 y 来说:  
* y 是长度为 N 的已知列向量
* 首先基于频率域变换, 得到一个过完备字典 D: `N*T`,  T 即傅里叶变换的基波数列, 或者说精度
* 求一套系数 x, 使得 `y=Dx`, 具体的:
  * x 是未知的长为 T 的列向量
  * 那么, 该运算代入线性代数, 即 T 个未知数, N 个方程的方程组求解, 由于 T>N, 因此有无限多解
  * 因此图像降噪输入 ill-posed problem 不适定问题, 即有多个满足条件的解, 且无法直接判断哪一个解更合适   
* 此时 对于解 x 的增加的约束即为 稀疏表达, 即希望 x 中的 0 尽可能多, 可以表示成 `min norm(x,0)` (非 0 元素的个数最小)
* 带入到图像中, 那么 y 就是2维矩阵
 

稀疏表达的具体问题 1: 稀疏编码 Sparse Coding     有 OMP 算法
* 字典 D 已知 (频率域的波), 求 y 在 D 上的稀疏表示 x,
$$x= \underset{x}{argmin}.norm(y-Dx,2)^2, s.t. norm(x,1)\le\epsilon $$

稀疏表达的具体问题 2: Dictionary Learning
* D不再是频率域分解而是手工设计的, 同时求出好的 D 和 x
  * 直接进行频率域转换得到的字典是自然正交的
  * 然而正交基往往只能表示图像的某一个特征而不能够同时表示其他特征
  * 因此正交基的稀疏性不如非正交基
* 按顺序 固定D更新x, 固定x更新D, 迭代计算


#### 4.1.2.1. K-SVD 算法 与字典学习

首先, 对于图像来说, 原始样本 Y 也是矩阵, 同理 Y 在字典 D 的稀疏表达 X 也是矩阵  
$$Y=DX$$
具体的优化目标可以表示成下面两种数学式, 其中 $||x_i||_0$ 为零阶范数, 代表向量中不为0的元素的个数

$$\underset{D,X}{min}||Y-DX||^2_F, s.t.\forall i, ||x_i||_0 \le T_0$$
$$\underset{D,X}{min}\sum_i||x_i||_0, s.t.\underset{D,X}{min}||Y-DX||^2_F\le \epsilon$$


K-SVD 即迭代K次的奇异值分解, 奇异值即等同于稀疏表达的字典, 而 K-SVD 属于迭代算法, 其特点是每次更新只更新字典的一个 原子 (对应于 D 中的一列)

首先是稀疏表示的目标函数
* $||Y-DX||_F^2$  实际信号和字典系数乘积的差值最小
* $||Y-\sum_{j=1}^Kd_jx_j^T||_F^2$  列向量乘以行向量, 然后 K 个矩阵相加
* $||(Y-\sum_{j\not ={k}}d_jx_j^T)-d_kx_k^T||_F^2$
* $||E_k-d_kx_k^T||_F^2$  将除了原子 k 的其他原子带来的误差统合为 E_k   

求解更新后的 d_k 和 x_k 就需要用到对 E_k 的 SVD 分解了  
* 首先要确保稀疏性, 对于 原子 k 来说, 列向量 d_k 乘以稀疏行向量 x_k  
  * 对于 x_k 的 0 元素来说, 相当于原子 d_k 不参与对应位置的 Y 的计算
  * 即不参与 目标函数的优化
  * 此时可以删除掉 E_k 对应位置的列, 具体为 $E_k \rArr E_k^{temp}$
* 对只保留 x_k 非0 列的 $E_k^{temp}$ 来进行 SVD 分解
  * $E_k^{temp}=UΣV^T$
  * 那么 U 的第一列即为更新后的 d_k' 
  * V的第一列与 $Σ(1,1)$ 元素的乘积即为新的 x_k
  * 对每个原子 k 逐个更新 字典d_k 和系数 x_k

对于噪声图像 z = y+n , 为什么通过对 z 执行 K-SVD 能够还原出 y 需要别的证明




### 4.1.3. Low Rankness 聚类低秩

居然是图像降噪的一大类  



# 5. Image Signal Processing (ISP)

作为光线传感器来说, 只能接受光的强度, 无法得知光的颜色
* 基础的相机感光元件是通过颜色滤镜和光线传感器的组合来实现的
* 通过在 Sensor 分小区域附加上 RGB 三色滤镜, 来得到 Bayer 


Image Pipeline : 对 Bayer 图像进行处理, 得到标准 RGB 图像  

一般流程: Bayer -> Pipeline -> YUV -> Jpeg Encoder -> JPG

## 5.1. Image Pipeline 的操作分区

* Software Control (SW controal)

* Image Front End (IFE)
  * 基本上 Bayer 格式处理的流程就在这一部分

* Bayer Processing Segment (BPS)

* Image Process Engine (IPE)


## 5.2. Image Pipeline 的各种操作及其缩写

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


## 5.3. Gamma

## 5.4. Raw 图像

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

## 5.5. Macbeth Chart

The `Macbeth ColorChecker Chart`, also misspelled as the `MacBeth` Color Checker Chart
* the industry standard color checking chart for cinematographers and photographers alike.
* Comes with a plastic protective sleeve and cardboard outer sleeve.

目前被称作 `ColorChecker Color Rendition Chart`. `Macbeth` 是其在 1976年的 原论文中的名称

## 5.6. calibration 

日文 キャリブレーション , 原译 矫正, 调整

在数字照片处理领域中, 用于指代将图片信号的颜色的错误修正的操作 (color calibration)

## 5.7. lens shading

lens shading 是一个笼统的称呼, 没有明确的定义, 只是用来指代由于镜头的物理组装性质导致的颜色亮度偏移问题, lens shading 本身可以分为两种  
* Luma Shading (亮度均匀性) 即暗角, 图像呈现出中心区域较亮, 四周偏暗
* Color Shading (色彩均匀性), 图像中心区域于与四周颜色不一致, 中心或者四周出现偏色

Luma Shading 的成因: 凸透镜镜头的光学特征
* 由于凸透镜中心的聚光能力大于其边缘, 导致 Sensor 中心的光线强度远大于其四周, 又称 边缘光照度衰减
* 由于摄像头本身的机械结构误差, 导致光线在镜头内的传播受影响

Color Shading 的成因 : 相对复杂得多
* IR-Cut filter 红外截止滤波片, 用于消除投射到 Sensor 上的不可见光, 防止 Sensor 由于不可见光而产生伪色, 波纹等
  * 干涉型: 最普通的类型, 在可见光区域的透光率较高, IR则相反. 但是在呈角度拍摄照片时, IR透过率会受到影响, 从而导致 R 通道的值不正确引起偏色
  * 吸收型: 表现为蓝玻璃, 对 IR 有很强的吸收作用, 能减轻渐晕和色差问题
  * 混合型
  * 各种 IR-Cut filter 可以减轻 IR 的Shading 问题, 但是会减少进光量, 从而加大了噪声
* Micro Lens : 图像 Sensor 的 RGB 像素感光块, 为了使感光面积不受感光片的开口面积影响, 在每个像素上增加一层微透镜 (Micro Lens), 用于收集光线, 提高感光度
  * 由于微透镜的主光线角 CRA (Chief Ray Angle) 值与镜头的 CRA 不匹配便会导致严重的 Shading 问题  


# 6. Information Theory



## 6.1. Random Processes 随机过程

在信号处理中, 有一类信号很重要, 称为 随机信号 (Random Signal), 也被称为随机过程 (Random processes / stochastic processes)

随机过程的表示: 所有可能的输出 + 每个输出对应的概率
* realization : 对于随机过程 $X(t)$ 有 N 个可能的输出的确定集合 $x_1(t), x_2(t),..,x_N(t)$ 
* 各个输出的概率 $p_1,p_2,...,p_N$ 概率和为 1, 且通常 $N=\infin$
* 对于一个连续时间的随机过程 $X(t)$, 在任意时刻 $t_0$ 所对应的值都是一个随机变量 $X(t_0)$
* 对于一个离散时间的随机过程 $X[n]$, 在任意序号 $n_0$ 所对应的值都是一个随机变量 $X[n_0]$


相关函数: 描述两个随机变量x,y的相似程度
* 如果 x y 都是宽平稳过程, 则相关函数定义为 
$$r_{xy}[k]=E(x[n]y[n+k])$$


### 6.1.1. 随机过程的概率模型  

* 期望 Mean/Expectation : $\mu_X(t_i)=E[X(t_i)]$
* 两个时间点的相关性 Auto-Correlation : 
  * $R_{XX}(t_i,t_j)=E[X(t_i)X(t_j)]$
* 两个时间点的协方差 Auto-covariance  : 
  * $C_{XX}(t_i,t_j)=E[X(t_i)-\mu_X(t_i)]E[X(t_j)-\mu_X(t_j)]=R_{XX}(t_i,t_j)-\mu_X(t_i)\mu_X(t_j)$


### 6.1.2. Power Spectral Density (PSD)

PSD, 功率谱密度, 是一个信号分析方法, 是用于分析随机振动的最常用工具  

对于一个特定的信号来说, 一般有 时域 频率 两个表达形式, 时域表现的是信号随时间的变化, 频域表现的是信号在不同频率上的分量  
* 对时域信号进行傅里叶变换可以得到该信号的频域表示, 从而得到信号在频率上的特性, 进而应用各种处理
* 对于具有随机过程的不确定信号
  * 在建模的时候是无法直接进行傅里叶转换的
  * 针对某一段实际的信号的采样的傅里叶转换不能代表该信号总体的频率描述  

对于随机过程加以优化取得的频率特性, 即 Power Spectral Density (PSD), PSD 是起源于物理上的表示  
* 用于表达信号 能量/变动/方差 与频率的关系的量
* 通常用于描述宽带(随机)信号
* PSD 的数值含义是振幅的平方在每赫兹上, 单位是 $V^2/HZ$

电学中, Power 功率即

$$P=UI=U\frac{U}{R}=\frac{U^2}{R}$$

因此电学中所谓的功率是正比于 电压V的平方.  
将一个随机过程 $x(t)$ 看作是`单位电阻`上的电压, 那么随机过程的方差就可以看作是电压的平方=功率, 即能量
* 信号是功率有限的确定信号
* 因此 $x(t)$ 的瞬时功率的期望是固定值 $E[x^2(t)]$ 


在结论上:
* 功率谱密度和自功率谱是相辅相成的两个信号描述方法, 是起源于计算机离散数值计算的数学现象
* 对于随机信号
  * 自功率谱有不同的振幅
  * 功率谱密度有相同的振幅
* 对于稳定正弦信号
  * 自功率谱有相同的振幅
  * 功率谱密度有不同的振幅
* 对于所有情况, 在评估谱函数的时候最好都使用 RMS 来量化
* 工业上, PSD 主要用于量化随机振动疲劳
* 自功率谱主要用于量化正弦波

图像处理中, 噪音图像对应的离散功率谱密度可定义为 $P(k)$
$$DFT(k)=F(k)=\sum_{n=0}^{N-1}f(n)e^{-j2\pi nk/N}$$
$$P(k)=\lim_{N\rightarrow \infin}\frac{|F(k)|^2}{N} (rad^2/hz)$$

即图像的二维离散傅里叶变换除以对应的图片 shape

#### 6.1.2.1. 均方根 Root Mean Square (RMS)

均方根是描述 AC 波的有效电压或者电流的一种最普遍的表达方式  

$$RMS=\sqrt{\frac{\sum^N_{i=1}x^2[i]}{N}}$$

物理由来:
* 在 DC 直流电路中, 对于电压或者电流可以很简单的定义
* 对于 AC 交流电路中, 对于物理量的描述就复杂得多, 其中 RMS 就是最朴素的一种
* 对于 AC 的波形函数进行 RMS 计算
  * 1.计算波形函数的平方值
  * 2.计算波形函数平方后的函数的时间平均值
  * 3.开根
* 在 AC 交流电路的表示中, RMS 值通常称为有效值或者 DC 等价值, 在日常生活中的交流电压就是用 RMS 来表示的
* 理论中的有效值一般都是用积分直接积出来的

#### 6.1.2.2. 谱线 - 对信号的测量

对于一个固定带宽的信号 e.g. 6000HZ, 所谓以某一个分辨率进行的测量即:
* 对谱进行数字化得到的离散数据点
* 加入以 1HZ 分辨率进行测量, 则会得到 6000个数据点, 这里的 6000 被称为 谱线
* 通常会以不同分辨率进行多次测量, 大致为:
  * 1HZ : 最低表现振幅
  * 4HZ : 中程表现振幅
  * 8HZ : 最高表现振幅
* 对于一个固定信号, 谱线的振幅实际上是谱线的函数, 谱线越多, 每个谱线的振幅越低   


#### 6.1.2.3. 信号的能量分类

首先定义两种不同的信号概念, 概念上这两种信号都是建立在无穷大时间的积分上的, 对于一个信号 $f(t)$, 首先根据 电阻的模拟理解, 可以写出该信号的能量 $E$    

$$E=\int^\infin_\infin f^2(t)dt=\lim_{T\rightarrow \infin}\int^{T}_{-T}f^2(t)dt$$

同理写出对应的 功率, 即能量除以时间  

$$P=\lim_{T\rightarrow \infin}\int^{T/2}_{-T/2}\frac{1}{T}f_T^2(t)dt=\lim_{T\rightarrow \infin}\int^{T}_{-T}\frac{1}{2T}f_T^2(t)dt$$

即 能量就是信号的平方在无穷上的积分, 而功率就是能量除以时间, 此时根据 T的极限存在与否 以及最终的功率值 , 可以进行分类区分  
* 能量信号  : 有限能量 + 零功率, e.g. 单方波, 极限值为0 的波形. 反正曲线的面积, 即能量是有限的, 但是时间 T 可以无穷延申, 导致 P 为 0, 一般都是理论上的信号   
* 功率信号  : 无穷能量 + 有限功率, e.g. 无穷的正弦波, 无穷的白噪声.  
* 非功非能  : 无穷能量 + 无穷功率, e.g. 各种理论上的波形, 单调递增之类的.  
* 一个信号可能既不是能量信号也不是功率信号, 但一定不会同时是两者

定理的信号种类:
* 所有的有限数量脉冲信号都是能量信号
* 所有的周期信号都是功率信号 (幅值有上下界)  
* 对于现实数据, 因其定义域有限, 根据信号在无限时间上的理论延拓方式可以有不同的结果
  * 将定义域是做一个周期, 进行周期延拓, 可以得到结论是功率信号
  * 将定义域以外的时间轴定义为 0, 可以得到结论是 能量信号.    

#### 6.1.2.4. 谱密度 能量谱密度

如果是能量信号 $f(t)$, 因为不存在功率, 所以也就不存在功率谱密度, 但是可以对能量进行计算
* 通过傅里叶变换, 一样可以分理处不同频域分量所对应的能量 得到 $F(\omega)$
* 对于一个固定的频率, 其能量为 $dW=\frac{1}{2\pi}|F(\omega)|^2d\omega$
* 那么进行积分, 就可以得到整个信号的能量
* 此时 $F(\omega)$ 就可以直接定义成能量谱, 即能量与功率的关系   


#### 6.1.2.5. 自功率谱  Self Power Spectrum 

同样也是物理领域的名词  

对于一个 自功率谱密度函数 $Sxx(f)$ 
* 反应了 相关函数在 时域内表达随机信号自身与其他信号在不同时刻的内在联系
* 当随机信号 均值为0 的时候: `自相关函数和自功率谱密度函数互为 傅里叶变换对`
* 物理意义
  * tao=0 时, $Sxx(f)$ 曲线与频率轴f, 所包围的面积就是信号的平均功率
  * $Sxx(f)$ 还表明了信号的 功率密度沿着频率周的分布情况

举例: 对于一个噪声带宽信号, 进行了三次频率分辨率(频谱分辨率)不同的测量, 可以得到振幅不同的 自功率谱  
* 首先信号本身肯定是一个固定存在的东西
* 关于频率分辨率 (单位:HZ), 越小, 即分辨率越高, 越来越多的数据点会被采样用于测量, 这会导致同样的信号被分割成更小的部分
* 我们讲, 同一个信号, 它的 数据总和是相同的, 信号的总数是相同的
  * 信号的总数 可以反映在  `整个频谱的均方根的总和`
  * 对于一个信号, 任意分辨率测量得到的自功率谱的 RMS 总和是相同的

由于, 对于一个噪声带宽信号, 分辨率不同的话, 得到的数据在表现上的振幅会有较大的差异, 因此寻求一个更优的信号表现方法, 即:
* 自功率谱和功率谱密度都是正确的信号表示形式
* PSD 更换函数的方式优化了数据的表示形式, 为了减少了不同分辨率下自功率谱的明显差异的问题
* 即: 频率分辨率归一化

将每条谱线的振幅都除以频率分辨率, 且惯例中信号在功率谱密度的振幅是平方的, 因此最终 PSD 的单位显示为 $V^2/HZ$



#### 6.1.2.6. 功率谱密度的计算


对于一个功率有限的确定信号 $f(t)$, 即功率信号, 从中截取 $|t|\le \frac{T}{2}$ 的一段, 写成一个截尾函数 $f_T(t)$, 即:

对于一个有限的 T, $f_T(t)$ 的能量也应该是有限的:
* 对于能量守恒定理, 信号在时域的能量等于信号在频域的能量
* 令傅里叶变换 $F[f_T(t)]=F_T(\omega)$, 此时可以表示该截尾信号的能量
* 由于 T 是 f(t) 的周期, 因此还可以写出 f(t) 的平均功率
$$E_T=\int^\infin_\infin f_T^2(t)dt=\frac{1}{2\pi}\int^\infin_\infin|F_T(\omega)|^2d\omega=\int^{T/2}_{-T/2}f_T^2(t)dt$$

$$P=\lim_{T\rightarrow \infin}\int^{T/2}_{-T/2}\frac{1}{T}f_T^2(t)dt=\frac{1}{2\pi}\int^\infin_\infin\lim_{T\rightarrow \infin}\frac{1}{T}|F_T(\omega)|^2d\omega$$

对于极限, $T\rightarrow \infin$, 有$f_T(t)\rightarrow f(t)$, 对于功率信号, 这个极限是存在的,直接定义它为 f(t) 的 `功率谱密度函数 PSD`, 记作 $P(\omega)$

$$P=\frac{1}{2\pi}\int^\infin_\infin P(\omega)d\omega$$
$$P(\omega)=\lim_{T\rightarrow \infin}\frac{|F_T(\omega)|^2}{T} (rad^2/hz)$$

功率谱是反应单位频带内信号的功率随频率的变化情况, 即信号功率在频域内的分布情况, 有 $P(\omega)$的米娜及就是该信号的总功率, 功率谱保留了信号的频域信息但是舍去了全部的相位信息  

由于平方的关系, $P(\omega)$ 是偶函数, 也被称为双边功率谱, 为了简化数据存储, 也可以定义单边功率谱$S(\omega)=2P(\omega)$

$$P=\frac{1}{2\pi}\int^\infin_\infin P(\omega)d\omega=\frac{1}{2\pi}\int^\infin_0 2P(\omega)d\omega$$

在实际工程中, 信号的采样结果都是离散的, 即对于离散时间信号 f(n), 进行 PSD 计算可以使用 DFT 离散傅里叶变换  

对应的离散功率谱密度可定义为 $P(k)$
$$DFT(k)=F(k)=\sum_{n=0}^{N-1}f(n)e^{-j2\pi nk/N}$$
$$P(k)=\lim_{N\rightarrow \infin}\frac{|F(k)|^2}{N} (rad^2/hz)$$

# 7. 图像复原

图像复原是一种最广泛的操作, 指的是对 退化图像 进行重建或回复的操作
* 图像退化指代的是图像在形成, 传输, 记录等过程中, 由于多种因素导致的质量下降
* 典型的图像退化为: 模糊, 失真, 噪声等
* 图像复原通过某种退化现象的先验知识来建立退化模型, 来实现对退化图片的重建和复原  

# 8. Filter
## 8.1. Filter Denoise


常用的降噪方法:
* wiener filter
* dct
* BM3D
* wavelet-based
* nlmeans
* Bilateral filter

自适应滤波: 近年以来发展起来的一种最佳滤波方法


Local Filter: 图像内容大部分都是低频的, 相邻像素的差值通常不会很大
* 最基础的均值滤波就是将窗口里的像素求平均即可, 但是会导致边缘细节模糊
* 使用不等权重的窗口函数, 例如高斯函数, 从而赋予中心像素更高的权重, 可以一定程度上更好的保留图像的细节 (考虑空间上的距离)
* 在空间距离的基础上考虑像素值的距离, 就有了双边滤波
* 由于需要考虑图像纹理的性质, 所以 local filter 的窗口无法设置的过大, 限制了参与滤波的像素个数, 从而影响到降噪的效果

Non-local Filter : 尽管像素内容不同, 但是对于 AWGN 来说, 噪声在空域上也是满足 i.i.d 的
* 如果噪声图像上的两个块没有重叠, 即每个像素都是独立的, 那么两个块的误差的方差也是随着块的增大而比例地缩小的
* 如果使用较大的块去比较, 相似性就能得到比较好的保证  
* 一般的图像, 例如天空, 道路, 建筑砖块等, 在不考虑计算复杂度的情况下, 找到足够多的相似块不算难事, 将这些相似块叠加起来, 就能够缩小噪声的方差  
* 每个块的匹配误差都是不一样的, 匹配误差小则可以赋予更高的权重
* 在 Non-Local Means 算法中, 只有每个块的中心像素才会被应用滤波, 因此其时间复杂度非常恐怖, 但是效果巨好    


Transform Domain Filter: 变换域, 即像素值在水平或垂直方向的波动快慢
* 图像变换域的稀疏性 : 图像内容通常以低频为主, 大片纯色或者渐变等, 在频域的图像只有少数几个低频点会有较大的系数或者能量, 而高频点的系数或者能量则可以忽略  
* 根据证明, AWGN 经过正交变换之后仍然是 AWGN, 对于噪声信号的正交变换可以理解为 无噪声信号的变换系数加上 AWGN 的变换系数
* 通过设置阈值的方式, 直接消除掉幅度较小的变换域系数, 可以实现不错的降噪效果, 对于夹杂在低频中的系数的噪音则束手无策. 
* 该方法能够极大的减少噪声, 在一定程度上保留图像的纹理, 对于物体边缘的区域会因为高频信息的丢失产生模糊或者 振铃效应  
* 对于噪声及其强烈的信号, 低频信号的系数甚至不如高频信号, 用硬阈值的方式会消除掉图像原本的信息


Collaborative Filtering : 协同滤波  
* 原理 : AWGN 再怎么正交变换仍然是 AWGN
* 把相似的块叠加起来形成第三维, 然后再前两维变换的基础上, 进行第3维的正交变换  
* 由于叠加的前提是相似的块, 因此在第3维上, 信号部分的变换不大, 即第3维的变换域中, 信号部分集中在低频, 而噪声保持了 AWGN 
* 这一操作可以进一步强化原本的信号部分的变换域系数, 这个时候在进行阈值操作就可以大步幅减少误杀信号的情况 
* 叠加相似块的数量是控制参数, 叠加数量越多, 低频信号越强, 但是由于块之间仅仅只是相似, 会导致块之间的差异点越来越多, 从而使得一部分的信息称为高频信号被抹去, 导致丢失细节  




### 8.1.1. spatial denoise  空间降噪

各种基于卷积或者空间逻辑上的的滤波器

#### 8.1.1.1. mean filter  平均滤波降噪

系数全1的卷积矩阵  

### 8.1.2. transform denoise 频域降噪

傅里叶变换, 小波变换


# 9. HDR high dynamic range

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

## 9.1. HDR 的实现要求


手机摄像头传感器的缺点:
* 光圈较小 : 光子收集量不多, 导致暗光场景下的高噪声
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


## 9.2. Multi-Frame Noise Reduction 基础的多帧降噪

对于相同内容的多帧图像 $S_n$
* 信号总量 $S=S_1+S_2+...+S_n$
* 噪音方差 $\sigma^2_t=\sigma^2_{t1}+\sigma^2_{t2}+...+\sigma^2_{tn}$
* 信噪比 $SNR = \frac{S}{\sigma_t}=\frac{S}{\sqrt{\sigma^2_{t1}+\sigma^2_{t2}+...+\sigma^2_{tn}}}$
* 由于原始内容相同:
  * $SNR=\frac{nS_1}{\sqrt{n\sigma_{t1}}}=\sqrt{n}\frac{S_1}{\sigma_{t1}}$
  * 即多帧合成可以开平方速率降低 TN


## 9.3. Image Fusion

多帧融合提高图像动弹范围或降噪
* 容易由于目标的移动导致的鬼影问题 (ghosting)
* 多帧图像无法正确的对齐内容 (由于拍摄时间的不同)
