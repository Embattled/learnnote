# 1. C++ 的时间库

C++ 事实上只支持两个时间标准库

* 标准的C语言时间库 `<ctime> "time.h"`
* C++ 集成的 `<chrono>` 库


# 2. ctime time.h

## 2.1. 数据类型

### 2.1.1. time_t 秒数时间

`typedef /* unspecified */ time_t;`  

尽管C语言没有指定 time_t 必须要用什么格式来实现,  
大部分系统都使用 int 来存储从 ` 00:00, Jan 1 1970 UTC` 以来的秒数( POSIX time)  

标准的 time_t 的学名是 : calendar time 

```cpp
// 0 时刻
time_t epoch = 0;

// 输出时间
printf("%s", asctime(gmtime(&epoch)));
// 输出 Thu Jan  1 00:00:00 1970
```

### 2.1.2. tm 日历日期

	
Structure holding a calendar date and time broken down into its components.   
1. 秒
2. 分
3. 小时
4. 这个月的哪一天
5. 月数
6. 年
7. 周几
8. 一整年的第多少天
9. 是否使用夏令时  正数代表有效  0 代表不适用  负数代表信息不足



```cpp
std::struct tm;

int tm_sec;
	// seconds after the minute – [0, 61] (until C++11)[0, 60] (since C++11)[note 1]
    // (public member object)

int tm_min;
	// minutes after the hour – [0, 59]
    // (public member object)

int tm_hour;
	// hours since midnight – [0, 23]
    // (public member object)

int tm_mday;
	// day of the month – [1, 31]
    // (public member object)

int tm_mon;
	// months since January – [0, 11]
    // (public member object)

int tm_year;
	// years since 1900 
    // 注意该项是真实年份减去 1900
    // (public member object)

int tm_wday;
	// days since Sunday – [0, 6]
    // (public member object)

int tm_yday;
	// days since January 1 – [0, 365]
    // (public member object)

int tm_isdst;
	// Daylight Saving Time flag. The value is positive if DST is in effect, zero if not and negative if no information is available
    // (public member object)
```

### 2.1.3. clock_t CLOCKS_PER_SEC

以CPU时钟数表示的时间, 常和 `CLOCKS_PER_SEC` 常量一起使用用来计算程序运行时间  

```cpp
typedef /* unspecified */ clock_t;
		
// 可以用来计算程序运行时间
int main (void)
{
    // 开始计时
   clock_t start = clock();

   int sink=0;
   for(size_t i=0; i<10000000; ++i)sink++;

   clock_t end = clock();
   double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC; 
   printf("for loop took %f seconds to execute \n", cpu_time_used);
}

```

## 2.2. 时间操作

### 2.2.1. time

returns the current calendar time of the system as time since epoch   

```cpp
time_t time( time_t *arg );
// 注意输入的是一个 time_t 的指针 和返回值作用一样
// 既可以通过返回值获取时间 也可以通过传入指针获取
// 不需要指针获取的话传入 nullptr 即可

// Return Value:
// Success      : Current calendar time encoded as time_t object
// Error        : (time_t)(-1)

int main(void)
{
    time_t result = time(NULL);
    if(result != (time_t)(-1))
        printf("The current time is %s( %jd seconds since the Epoch)\n",
               asctime(gmtime(&result)), (intmax_t)result);
}

```

### 2.2.2. clock

返回 CPU 时间,  通过除以 `CLOCKS_PER_SEC` 可以转换成秒数的格式  
since the beginning of an implementation-defined era related to the program's execution


```cpp
std::clock_t clock();

// 获取一个开始时间
std::clock_t c_start = std::clock();

// 获取一个结束时间
std::clock_t c_end = std::clock();
```

### 2.2.3. difftime 计算时间差

```cpp
// Computes difference between two calendar times as time_t 
double difftime( time_t time_end, time_t time_beg );

// 计算 time_end - time_beg
// 如果 end 比 beg　要早 那么会返回负数

```

## 2.3. 时间格式转换

最常用的函数, 用来将各种类型的时间进行转换  

GMT：Greenwich Mean Time 格林尼治标准时间    
    以英国格林尼治天文台观测结果得出的时间，这是英国格林尼治当地时间，这个地方的当地时间过去被当成世界标准的时间
UTC：Coordinated Universal Time 协调世界时  
    让基于原子钟的世界时和基于天文学（人类感知）的格林尼治标准时间相差不至于太大。并将得到的时间称为UTC，这是现在使用的世界标准时间。


GMT并不等于UTC，而是等于UTC+0，只是格林尼治刚好在0时区上。 GMT = UTC+0


### 2.3.1. asctime

将一个日历时间格式 `tm` 转换成一个标准的 25 字符的时间格式:    
`Www Mmm dd hh:mm:ss yyyy\n`   
* 如果 tm 中有数据越界 则不生效
* 如果year大于四位数 不生效
* 该函数不支持 localization 本地化


```cpp
// 在C++ 下 ctime 中只有这一种格式 , 没有剩下下面两个 
char*   asctime  ( const struct tm* time_ptr );

// (since C11)
errno_t asctime_s( char* buf, rsize_t bufsz, const struct tm* time_ptr );

// (since C23)
char*   asctime_r( const struct tm* time_ptr, char* buf );

```

### 2.3.2. localtime gmtime ctime

将标准 time_t 秒数间转换为 tm 日历时间 或者标准字符串时间  

```cpp
// 输入的是 time_t 的指针

// 转换成 local time 
std::tm* localtime( const std::time_t *time );
// 全球时间 UTC
std::tm* gmtime( const std::time_t* time );
// 返回字符串形式的 Www Mmm dd hh:mm:ss yyyy  注意是当地时间 
// 相当于 std::asctime(std::localtime(time)) 
char* ctime( const std::time_t* time );
	
// 注意返回的是一个 tm 指针 (执行成功)
// 执行失败会返回 nullptr


// 获取 time_t
std::time_t t = std::time(nullptr);
// 全球时间 UTC
std::cout << "UTC:       " << std::put_time(std::gmtime(&t), "%c %Z") << '\n';
// 当地时间
std::cout << "local:     " << std::put_time(std::localtime(&t), "%c %Z") << '\n';
// 字符串时间 注意只有 ctime 可以直接输出
std::cout << std::ctime(&result);


// 更改系统时区
std::string tz = "TZ=Asia/Singapore";
putenv(tz.data());
// 输出新加坡时间
std::cout << "Singapore: " << std::put_time(std::localtime(&t), "%c %Z") << '\n';
```

### 2.3.3. mktime

将标准 tm 日历时间转换为 time_t 秒数时间

```cpp
// 输入的是指针
std::time_t mktime( std::tm* time );

// Return
// time_t on success
// -1 value on failure
```
* tm 格式中的 wday(周几) 和  yday(年中天数) 将会被无视  
* tm 的信息超过范围不会造成函数出错, 如果转换成功, tm 超范围的数据还会被修正
* tm 如果非重要信息例如 wday 和 yday 还会被补上