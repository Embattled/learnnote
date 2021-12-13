# Netpbm项目

Netpbm is an open-source package of graphics programs and a programming library.
Several graphics formats are used and defined by the Netpbm project. 

They are also sometimes referred to collectively as the **P**ortable a**N**y **M**ap format (PNM).
Not to be confused with the related **P**ortable **A**rbitrary **M**ap format. 

| Type             | ASCII number | Binary number | Extension | Colors                                             |
| ---------------- | ------------ | ------------- | --------- | -------------------------------------------------- |
| Portable BitMap  | P1           | P4            | .pbm      | 0–1 (white & black)                                |
| Portable GrayMap | P2           | P5            | .pgm      | 0–255 (gray scale), 0–65535 (gray scale), variable |
| Portable PixMap  | P3           | P6            | .ppm      | 16777216 (0–255 for each RGB channel)              |

A value of `P7` refers to the `PAM` file format that is covered as well by the netpbm library.

在文件一开始第一行标明 `  P* ` 注明文件格式,然后跟着一行标明图片分辨率和色彩宽度

The PGM and PPM formats (both ASCII and binary versions) have an <u>additional parameter</u> for the maximum value (numbers of grey between black and white) after the X and Y dimensions and before the actual pixel data. ***Black is 0*** and ***max value is white***. 

    注释行是从#到该行末
    如果图象数据以字节格式存储,仅仅在头部的最后一个字段的前面才能有注释,在头部的最后一个字段后面通常是一个回车或换行

