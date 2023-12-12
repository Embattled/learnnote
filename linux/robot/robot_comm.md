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




