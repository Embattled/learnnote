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