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

# PNM FORMAT
## PBM Format

例如
```
P1
# This is an example bitmap of the letter "J"
6 10
0 0 0 0 1 0
0 0 0 0 1 0
0 0 0 0 1 0
0 0 0 0 1 0
0 0 0 0 1 0
0 0 0 0 1 0
1 0 0 0 1 0
0 1 1 1 0 0
0 0 0 0 0 0
0 0 0 0 0 0
```

但是其实像素内容不需要排列
```
P1
# This is an example bitmap of the letter "J"
6 10
000010000010000010000010000010000010100010011100000000000000
```

## PGM Format
```
P2
# Shows the word "FEEP" (example from Netpbm man page on PGM)
24 7
15
0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
0  3  3  3  3  0  0  7  7  7  7  0  0 11 11 11 11  0  0 15 15 15 15  0
0  3  0  0  0  0  0  7  0  0  0  0  0 11  0  0  0  0  0 15  0  0 15  0
0  3  3  3  0  0  0  7  7  7  0  0  0 11 11 11  0  0  0 15 15 15 15  0
0  3  0  0  0  0  0  7  0  0  0  0  0 11  0  0  0  0  0 15  0  0  0  0
0  3  0  0  0  0  0  7  7  7  7  0  0 11 11 11 11  0  0 15  0  0  0  0
0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
```

## PPM Format
This is an example of a color RGB image stored in PPM format.   
There is a newline character at the end of each line. 

如果是P3格式,数据将以ASCII文本来表示,每个像素的值从0到前面第一行定义的最大值  
如果是P6格式，图象数据以字节格式存储，每个色彩成分`（R，G，B）`一个字节。仅仅在头部的最后一个字段的前面才能有注释，在头部的最后一个字段后面通常是一个回车或换行。P6图象文件比P3文件小，读起来更快  
```
P3
3 2
255
# The part above is the header
# "P3" means this is a RGB color image in ASCII
# "3 2" is the width and height of the image in pixels
# "255" is the maximum value for each color
# The part below is image data: RGB triplets
255   0   0  # red
  0 255   0  # green
  0   0 255  # blue
255 255   0  # yellow
255 255 255  # white
  0   0   0  # black
```


