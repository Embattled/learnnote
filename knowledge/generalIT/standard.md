# 各种标准


# 字符编码

1. Unicode
    * ISO 定义的标准化编码
    * UCS-2  2字节字符编码
    * UCS-4  4字节字符编码
2. EUC (Extended Unix Code)
    * 主要在 Unix 里使用的文字编码
    * 可以表示汉字, 日语等
3. EBCDIC (Extended BCD Interchange Code)
    * IBM 制作的通用计算机文字编码
    * 基础的不支持特殊语言, 但是可以通过制造商自己扩充来变得支持
4. ASCII (American Standard Code for Information Interchange)
    * ANSI 美国标准协会定制的编码
    * 本来只有7bit, 扩充到8bit, 一般是其他编码的基础
5. JIS 日本工业规格 标准
    * 日本定义的标准, 假名和数字定义在8bit空间里, 汉字定义在16bit空间 
    * 除此之外还有 shift JIS 标准, 提高了8bit和16bit字符同时存在的情况下的识别速度


# 图像格式

1. BMP (BitMap) 无压缩
    * 完全按照像素来存储数据
    * Windows 的标准静态图片存储格式
2. JPEG (Joint Photographic Experts Group) 不可逆压缩
    * ISO 国际标准的图像存储格式
    * 非常高的压缩率以及好的图像表现
3. GIF (Graphics Interchange Format) 可逆压缩存储格式
    * 在网络上较为常用
    * 色深只有 256色

# 动画格式

1. MPEG 系列 (Moving Picture Experts Group) 不可逆压缩
    * 影片的 ISO 标准存储格式
    * MPEG-2 常用在影片动画压缩中, DVD, TV 之类的
    * MPEG-4 压缩率更高, 用在移动设备中较多, 电话会议之类的场景

# 声音格式

标准:
* 采样率: 模拟信号一秒钟采样多少次, 例 44,100
* 数据深: 每一次采用的数据的大小, 例  16bit 

音乐存储用的格式
1. PCM (pulse code modulation) 无压缩存储
    * 音乐CD的存储格式
2. MP3 (MPEG Audio Layer-3) 不可逆压缩
    * 最广泛使用的音乐存储格式
    * MPEG1 的声音格式
3. WMA (Windows Media Audio) 不可逆压缩
    * Windows 电脑的标准声音压缩格式

声音传输格式
1. G.711 (PCM) 无压缩
    * 声音以 64kbit/s 进行传输
    * 用在数码电话上
2. G.729 (CS-ACELP) 不可逆压缩
    * 声音以 8kbit/s 进行传输
    * 用在电话拨号上,IP电话等


# General Device in Daily Life

## 主流显示设备

四种主流显示设备:

1. CRT 显示器
使用电子枪发射电子束, 击打在荧光体上使其发光.  

2. 液晶 显示器 (LCD: Liquid Crystal Display)
当前的主流, 液晶会随着电压的变化而改变分子的排列, 在两枚玻璃的中间封入液晶, 再加上背光板,   
再用电压控制液晶的形态来起到通过和遮蔽光的功能, 来起到显示的效果.  

3. PDP 显示器
再两枚玻璃之间充入高压惰性气体, 利用电压来使气体对应的发光, 类似于氙气灯.  
和一般液晶相比, 对比度高, 视野角度大, 但是耗电.  容易制作大面板, 不容易做小型显示器.  

4. OLED 有機EL 
使用有自发光特性的有机化合物, 不再需要背光板, 省电又薄,  
不容易制作成小面板.  


## 主流打印机分类

1. Dot Impact
网点印压式打印, 通过在纸上打出网点, 来构成文字. 流行于1990年以前.  
噪音大, 清晰度低, 可以使用碳纸来复印.  

2. Ink Jet 
喷墨打印机, 通过喷嘴在纸上喷墨水来打印.  
目前仍在不断发展, 随着喷头的精细程度越来越高分辨率也在上升.  

3. 热转写式打印机
可以细分为热融化型和热升华型.  
都是通过加热激光头或者墨水, 来让墨水转移到纸上.  

4. 激光打印机 
通过激光炮来把碳粉印在枚媒介上, 在单色打印领域有统治地位.  
速度快, 安静, 边际成本低.  

5. 制图仪 plotter
只用在工业设计, 用于打印设计图, 通过XY坐标来移动笔头, 接近于"画".  


## 光盘 

利用激光来实现数据的存储, 和其他的移动存储相比起来便携性较高, 介质的成本较低.  

1. CD (Compact Disc)  
直径8或者12厘米的树脂圆盘, 在盘面上刻印凹凸来代表二进制数据, 读取则是使用激光来照射读取.  
一般的大小是 650~700MB.  
在此基础上根据读取方式可细分为:  
   * CD-ROM (Read Only Memory): 只读光盘, 由制造工厂来写入内容. 作为音乐, 游戏的载体较多. 
   * CD-R (Recordable): 一次性刻录光盘, 用户来自己写入内容, 一般用来做会议记录. 盘面上涂抹了有机色素, 受激光照射会焦化从而形成CD的凹凸数据.  
   * CD-RW (Rewritable): 可重复写入型的CD

2. MO (Magneto-Optical Disc)
磁性和光学技术同时使用的存储媒体, 在刻录的时候先用激光照射, 再用磁性写入数据.  
读取的时候只用激光就可以, 速度较快.  

3. DVD 
CD的升级规格, 原理和CD基本相同. 同样也分为 ROM, R, RW等类别.  
容量方面, 单面一般在 4.7GB, 支持双面刻录和多层刻录, 容量相应的翻倍.  

4. Blu-ray Disc
DVD的升级规格, 使用了青紫色半导体激光, 比DVD的记录密度更高. 同样也根据读写分3类.  
单面的容量在25GB.  


## RAID (Redundant Arrays of Independent)

复数的磁盘合起来用做单一的硬盘来使用, 分 0~6共7种

1. RAID0
最简单的方法, 为了获取更快的速度而存在.  
容量不变, 速度翻倍, 风险翻倍.  

2. RAID1
为了数据安全性, 多个硬盘保存相同的数据, 单硬盘出错时可以保证数据安全.  
容量只由单个磁盘的大小, 速度翻倍.  

3. RAID3
带有数据复原性的组合, 使用一个硬盘专门存储数据的奇偶校验码.  
数据不是连续存储, 而是以byte为单位分散在所有存储磁盘种.  
容量=总容量-校验码硬盘容量, 速度同理=n-1

4. RAID4
同RAID3一样, 只是数据的分散单位变味了block (物理上的IO单位)  

5. RAID5
同样是 block 分割数据, 但是校验码也同样按 block 分割分散存储了.  


## Memory

RAM (Random Access Memory) 随机存储器曾经是用于区别顺序读取存储器(磁带)的类别,  
随着发展目前的 ROM 也能随机存储了, 所以目前 RAM 主要用来指代断电数据不保留的易失存储期.  

* DRAM : 使用电容器制作的存储期, 需要定时刷新, 目前主要用来做电脑主内存  
* SRAM : flip flop回路(双稳态振荡器电路)制作的内存, 不需要刷新, 速度快, 造价贵, 主要用于CPU缓存

当前ROM的分类
* Mask ROM : 制造时已经确定了内容, 不可更改
* PROM (Programmable ROM) : 可以一次写入的ROM
* EPROM (Erasable PROM) : 可以用紫外线来消除内容, 进而多次写入的 ROM
* EEPROM (Electrically EPROM) : 电可擦除ROM, 用电力来以 block 或者 byte 为单位消除数据
* Flash Memory : 同样是根据电力来擦除内容


## Interface

并行传输接口:
1. IDE (Integrated Device Electronics) / ATA (Advanced Technology Attachment)
主要用做硬盘的传输接口, 一根线可以接两个硬盘.  

2. SCSI (Small Computer System Interface)
比VGA还大的接头, 用于电脑周边设备的连接.  

3. IEEE1284
用于打印机和扫描仪的较早的接口, 已经不使用了.  

串行传输接口:
1. USB (Universal Serial Bus)
标准速度: 1.1=12MB/s, 2.0=480MB/s. 通过 hub 可最多连接127台设备.  

2. RS-232C
低速通用接口, 是USB流行以前的标准接口, 被USB替代了.

3. IEEE1394
用在电子照相机比较多, 400MB/s, 最大63台链型连接, 可供给电源.  
可以不经过电脑, 直连多个设备.  

4. 串行 ATA (SATA)
ATA的后继规格, 提高了速度, 但是一条线只能接一个硬盘.  


无线接口:
1. IrDA (Infrared Data Association)
使用红外线进行通信的接口, 距离近, 穿透力差.  

2. Bluetooth
短距离通信的接口, 使用公用的2.4GHz频段的电波, 10m以内有较好的穿透性.  