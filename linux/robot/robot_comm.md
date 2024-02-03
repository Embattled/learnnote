# 关于 Robot 设备通信的小点


## I2C (Inter-Integrated Circuit) 

用于嵌入式和电子领域的通信协议  

是基于 two-wire 的接口， 

## System Management Bus

SMBus or SMB  

简单的  single-ended simple two-wire bus
* 主要用于最轻量化的通信, low-speed
* 例如最便宜的计算主板
* 主要用于控制 ON/OFF
  * reading battery status, and managing other embedded devices within a computer system.

是基于 I2C 的通信协议
* 内建了一些用于差错的 features

所谓的 two wires:
* data line (SDA)
* clock line (SCL)


## SPI (Serial Peripheral Interface)

高速, 全双工协议, 用于短距离通信.  

速度上 SPI 比 I2C 更快, 但是需要的引脚数更多: MISO, MOSI, SCK, SS

## UART (Universal Asynchronous Receiver/Transmitter)

两条线, 分别用于传出数据和接收数据, 因此是异步的.  




## MIPI

Mobile Industry Processor Interface 简称MIPI, 移动产业处理器接口. 是一个联盟, 对应的接口即为 联盟发起的为移动处理器指定的 开放标准和规范

联盟官网 https://www.mipi.org

主要公司有:
* 美国德州仪器 TI
* 意法半导体 ST
* 英国 ARM
* 芬兰诺基亚 Nokia


## CSI (Camera Serial Interface) 

(全称?) MIPI-CSI-2 协议  MIPI 联盟协议的子协议 , 专门针对摄像头芯片的接口而设计的. 主要技术掌握在 日本东芝, 韩国三星, 美国豪威

一种串行数据传输协议, 通常用于图像传感器, 用于传输图像数据. 该接口通常被用于嵌入式系统和移动设备中. 

CSI 接口本身还有一些协议上的规则, 例如 帧同步信号, 数据的时序和格式等.  

CSI协议有两个版本
* CSI-2 : 物理标准上, CSI 有两个差分信号 D-Phy数据差分信号,  C-PHY控制差分信号, 分别用于传输图像数据和控制信息.  
  * 协议内核中可以看到 D-PHY 相关代码
* CSI-3 : 物理标准则是 M-PHY, 且物理层和 CSI-3 协议中还多了一个 Uni-Pro 层

## DSI (Display Serial Interface)

MIPI 联盟设计的 显示设备的协议   

DCS (Display Command Set) : 用于显示模块命令模式下的标准化命令集
