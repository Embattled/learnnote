# C++ 的 numeric limits

用更方便的方法获取类型的相关信息

* climit
* limits


# <climits>

This header was originally in the C standard library as `<limits.h>`.

This header is part of the type support library, in particular it's part of the C numeric limits interface. 


定义了全部的数字界限 


注意常用的 INT_MAX是C++11才定义的  

| 名称                    | 定义                                     | 备注            |
| ----------------------- | ---------------------------------------- | --------------- |
| CHAR_BIT                | 一个byte中有多少bit                      |
| MB_LEN_MAX              | 一个multibyte character中最大有多少bytes |
| CHAR_MAX                | char 类型的最大值                        |
| CHAR_MIN                | char 类型的最小值                        |
| 在这下面的是C++11加入的 |                                          |
| *_MIN                   | 该类型的最小值                           | *号代表多种类型 |
| *_MAX                   | 该类型的最大值                           | *号代表多种类型 |
| U*_MAX                  | 该类型的最大值                           | *号代表多种类型 |

*号的对应表

| *号填入 | 对应数据类型 |
| ------- | ------------ |
| SCHAR   | signed char  |
| SHRT    | short        |
| INT     | int          |
| LONG    | long         |
| LLONG   | long long    |

只有这么点内容，很容易记

# <limits>

C++的数据界限对应库,定义了一个类和两个枚举类型

## std::numeric_limits