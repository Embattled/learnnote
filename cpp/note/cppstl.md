# 1. C++ 的文件读写 fstream

iostream 标准库,它提供了 cin 和 cout 方法分别用于从标准输入读取流和向标准输出写入流.  
从文件读取流和向文件写入流,这就需要用到 C++ 中另一个标准库 `fstream`  

| 数据类型 | 描述                                                                                                                         |
| -------- | ---------------------------------------------------------------------------------------------------------------------------- |
| ofstream | 该数据类型表示输出文件流, 用于创建文件并向文件写入信息。                                                                     |
| ifstream | 该数据类型表示输入文件流, 用于从文件读取信息。                                                                               |
| fstream  | 该数据类型通常表示文件流, 且同时具有 ofstream 和 ifstream 两种功能, 这意味着它可以创建文件, 向文件写入信息, 从文件读取信息。 |


要在 C++ 中进行文件处理, 必须在 C++ 源代码文件中包含头文件 `iostream` 和 `fstream`

## 1.1. 文件打开与关闭

`ofstream` 和 `fstream` 对象都可以用来打开文件进行***写操作***, 如果**只需要打开文件进行读操作**, 则使用 `ifstream` 对象

```C++
//打开文件时可以先定义对象再调用函数,也可以直接在定义对象时打开
myfile.open(const char *filename, ios::openmode mode);
ofstream myfile ("example.bin", ios::out | ios::app | ios::binary);
//关闭文件时使用 close 方法
myfile.close();
```
| 模式标志    | 描述                                                                   |
| ----------- | ---------------------------------------------------------------------- |
| ios::app    | 追加模式。所有写入都追加到文件末尾。                                   |
| ios::ate    | 文件打开后定位到文件末尾。                                             |
| ios::in     | 打开文件用于读取。                                                     |
| ios::out    | 打开文件用于写入。                                                     |
| ios::trunc  | 如果该文件已经存在, 其内容将在打开文件之前被截断, 即把文件长度设为 0。 |
| ios::binary | 二进制模式打开.                                                        |

## 1.2. 文件读写

**对于字符文件**  
对`ofstream` 或 `fstream` 对象,使用流插入运算符`（ << ）`向文件写入信息  
对`ifstream` 或 `fstream` 对象使用流提取运算符`（ >> ）`从文件读取信息  
还有类似于getline(); 读取一整行的函数

**对于二进制文件**
```cpp
write ( memory_block, size );
read ( memory_block, size );
//memory_block 是一个 char* 用于指向读取到内存的地址或写出到文件的内容源,size是文件块的大小,可以传入 streampos 类型
```

**基础读写**
```cpp
get();  //extracts characters 
unget(); //Makes the most recently extracted character available again. 
putback(char);// Puts the character ch back to the input stream so the next extracted character will be ch. 
//功能是一样的，最大的区别在参数上 unget没有参数，是把已经从流读取出来的那个字符放回去，下次读取的时候可以读到这个字符 而putback是把参数c放入流中
peek(); //reads the next character without extracting it 
```
## 1.3. 检查函数
每一个流对象都有一个 `flag` 用于保存操作时的各种状态  
使用 `clear()` 来清除`flag`

对特定状态的检查函数,返回值都是布尔类型
|        |                                            |                                                                  |
| ------ | ------------------------------------------ | ---------------------------------------------------------------- |
| bad()  | Returns true 如果有读写失败                | 例如对一个没有以写入标志打开的流执行写入或者写入的磁盘已没有空间 |
| fail() | Returns true 在`bad()`的基础上检查格式问题 | 例如文件读出来的是字符但是传输给了一个整数变量                   |
| eof()  | 检查是否到了文件末尾.                      |
| 1.4.   | 1.4.                                       | 1.4.                                                             | 1.4. | 1.4. | 1.4. | 1.4. | 1.4. | 1.4. | 1.4. | 1.4. | 1.4. | 1.4. | 1.4. | 1.4. | 1.4. | 1.4. | 1.4. | 1.4. | good() | 最常用的函数, 对上面所有函数返回`true`的时候,返回`false` | `good()`与`bad()`不是对立函数,good一次检查更多的flag |
---

## 1.5. 文件位置指针操作

tellp() —— 与tellg()有同样的功能，但它用于写文件时。
* 当我们读取一个文件，并要知道内置指针的当前位置时，应该使用tellg()
* 当我们写入一个文件，并要知道内置指针的当前位置时，应该使用tellp()


`tellg() and tellp()`  返回输入或者输出流当前的文件位置指针,返回值的类型是`streampos`

`seekg() and seekp()`  istream 的 seekg 和 ostream 的 seekp  
可以设定输入位置指针,有两个重载  

```cpp
seekg ( streampos ); 
seekp ( streampos ); //直接设定位置

//offset是streamoff类型, 是一个 long long 整型
seekg ( offset, direction );
seekp ( offset, direction );//根据偏移方向设定指针位置
```
| `direction` | 查找方向                                        |
| ----------- | ----------------------------------------------- |
| ios::beg    | offset counted from the beginning of the stream |
| ios::cur    | offset counted from the current position        |
| ios::end    | offset counted from the end of the stream       |

下面的程序可以获得一个文件的大小 `bytes` 形式  
注意用来表示文件大小的`size`是`streampos size;` 使用这个类型可以安全的加减以及转换为数字  

```cpp
// obtaining file size
#include <iostream>
#include <fstream>
using namespace std;

int main () {
  streampos begin,end;
  ifstream myfile ("example.bin", ios::binary);
  begin = myfile.tellg();
  myfile.seekg (0, ios::end);
  end = myfile.tellg();
  myfile.close();
  cout << "size is: " << (end-begin) << " bytes.\n";
  return 0;
```


# 2. 工程要点
## 2.1. 後藤先生からのアドバイス

1. 功能函数和用户界面应该分开， 例如标准信息输出不应该出现在用户处理函数中
2. 凡是涉及到分配内存的功能都应该检查是否成功
3. 虽然图像被视为一维处理，但是从可读性的角度来看，应该认真地以二维阵列的形式在双循环中处理。
4. 注意要经常使用 unsign 
5. 在多重循环中使用条件分歧会对速度产生不利影响
6. 在内存限制不严重的情况下， 不要用float类型
7. 

## 2.2. 关于 new 的检查

通常来说，new 使用 try catch 来捕捉异常  
抛出的异常为 `std::bad_alloc` 类  
```cpp
try {
    while (true) {
        new int[100000000ul];   // throwing overload
  }
}catch (const std::bad_alloc& e) {
    std::cout << e.what() << '\n';
}

```

但是也有阻止抛出异常的方法  
`std::nothrow` 就是专门用来处理new异常的类, new 操作失败时返回的指针为 nullptr 而不是抛出异常  
```cpp
int* p = new(std::nothrow) int[100000000ul]; // non-throwing overload
if (p == nullptr) {
   std::cout << "Allocation returned nullptr\n";
   break;
}
```

# 3. string 字符串操作

字符串实际上是使用 null 字符 `'\0'` 终止的一维字符数组.因此, 一个以 null 结尾的字符串, 包含了组成字符串的字符.

C++ 中，独立的几个 string 对象可以占据也可以不占据各自特定的物理存储区，但是，如果采用引用计数避免了保存同一数据的拷贝副本，那么各个独立的对象（在处理上）必须看起来并表现得就像独占地拥有各自的存储区一样.   

只有当字符串被修改的时候才创建各自的拷贝，这种实现方式称为写时复制（copy-on-write）策略。当字符串只是作为值参数（value parameter）或在其他只读情形下使用，这种方法能够节省时间和空间。
  

## 3.1. C++string 的定义使用


```cpp

//----------定义:
std::string s2 = "c plus plus";
// 初始化为sssss
std::string s4 (5, 's');

//----转为C风格字符指针
//返回该字符串的 const 指针（const char*）
string path = "D:\\demo.txt";
FILE *fp = fopen(path.c_str(), "rt");


//-----流输入
//使用流对字符串进行输入的时候, 会把空格作为输入结束  
string s;
cin>>s;

//-------访问字符串中的字符
//string 可以通过下标来访问字符串中的字符
for(int i=0,len=s.length(); i<len; i++){
    cout<<s[i]<<" ";
}
```

**字符串拼接**  使用 `+` 和 `+=`  
1.  两边可以都是 string 字符串
2.  一个 string 字符串和一个C风格的字符串
3.  一个 string 字符串和一个字符数组
4.  一个 string 字符串和一个单独的字符


**基础8种迭代函数**  
返回对应的迭代器
```cpp
.begin();  //开头
.end();    //末尾
.r*();     //反向迭代器
.c*();     //c++ 11 新标准,const 迭代器,防止更改字符串内容
.cr*();    //顺序为cr
```

## 3.2. 实用操作函数

### 3.2.1. 关于子字符串的操作
```cpp
//获得子函数, pos 开始位置, 默认截取到字符串尾
string substr (size_t pos = 0, size_t len = npos) const;

//删除字符串中的一段 , 不用担心越界
string& erase (size_t pos = 0, size_t len = npos);

//插入字符串, 在指定位置插入另一个字符串
//第一个参入要注意不能越界, 否则会抛出异常
string& insert (size_t pos, const string& str);
```
### 3.2.2. 查找

```cpp
//查询函数, pos 表开始位置, 无视pos之前的匹配项
//返回值为第一个匹配项出现的位置, 若无则返回 string::npos
size_t find (const string& str, size_t pos = 0) const;
size_t find (const char* s, size_t pos = 0) const;


//rfind() 函数 和find的第二个参数功能相反
//第二个参数表示最多查找到第二个参数处, 若无则返回无穷大值
int index = s1.rfind(s2,6);

//查找子字符串和字符串共同具有的字符在字符串中首次出现的位置
//注意是任意一个共有的字符首次出现的位置
constexpr size_type find_first_of( const basic_string& str,size_type pos = 0 ) const noexcept;

```

## 3.3. **从C继承的基础的字符串操作**  

以下函数为c原生函数,包含在`<string.h>`头文件中, 在C++中也在 `<cstring>`中  
使用`<string.h>` 不需要在函数名前加`std::`  
| 函数名                   | 功能                                                                                                     |
| ------------------------ | -------------------------------------------------------------------------------------------------------- |
| 字符串操作函数           |                                                                                                          |
| `strcpy(s1, s2);`        | 复制字符串 s2 到字符串 s1。                                                                              |
| `strncpy(s1, s2, n);`    | 复制s1特定数量的字符到s2,若n大于s1的长度,不足的部分补`\0`,若n小于s1的长度,则s2是一个非`\0`结尾的字符数组 |
| `strcat(s1, s2);`        | 连接字符串 s2 到字符串 s1 的末尾。                                                                       |
| `strncat(s1, s2, n);`    | 连接字符串 s2 的 n 个字符 到字符串 s1 的末尾。                                                           |
| `strtok(str, delim)`     | 根据字符delim分隔字符串str,该函数返回被分解的第一个子字符串，如果没有可检索的字符串，则返回一个`空指针`  |
| 字符串验证函数           |                                                                                                          |
| `strlen(s1);    `        | 返回字符串 s1 的长度。                                                                                   |
| `strcmp(s1, s2);`        | 如果 s1 和 s2 是相同的，则返回 0；如果 s1\<s2 则返回值小于 0；如果 s1\>s2 则返回值大于 0。               |
| `strncmp(s1, s2);`       | 如果 s1 和 s2 的前n个字符是相同的，则返回 0；如果 s1\<s2 则返回值小于 0；如果 s1\>s2 则返回值大于 0。    |
| `strcoll(s1,s2)`     \*  | 功能和来strcmp类似,用法也一样,但是strcoll()会依环境变量LC_COLLATE所指定的文字排列次序来比较              |
| 字符串查找函数           |                                                                                                          |
| `strchr(s1, ch);`        | 返回一个char指针，指向字符串 s1 中字符 ch 的第一次出现的位置。                                           |
| `strrchr(s1, ch);`       | 返回一个char指针，指向字符串 s1 中字符 ch 的最后一次出现的位置。                                         |
| `strstr(s1, s2);`        | 返回一个指针，指向字符串 s1 中字符串 s2 的第一次出现的位置。                                             |
| 字符串字典查找函数       |                                                                                                          |
| `strspn(dest,scr)`       | scr指向的字符串作为合法字典,返回一个长度值,dest从起始开始的合法字符的长度                                |
| `strcspn(dest,scr)`      | scr指向的字符串作为不合法字符字典,返回一个长度值,dest从起始开始出现不合法字符的第一个位置度              |
| `strpbrk(dest,breakset)` | breakset作为终止字典, 返回dest第一次出现终止字符的位,没找到就返回`NULL`                                  |
| 内存操作函数             |                                                                                                          |
| `memchr (ptr,ch,count)`  | 找字符第一次出现的位置, 没找到就返回`NULL`                                                               |
| `memcmp (lhs,rhs,count)` | 与字符串比较类似                                                                                         |
| `memset (dest,ch,count)` | 注意 `int ch`                                                                                            |
| `memcpy (dest,scr,c)`    | copies one buffer to another                                                                             |
| `memmove(dest,scr,c)`    | 函数的功能同memcpy基本一致，但是memmove先创建temp内存,当src区域和dst内存区域重叠时, 可以安全拷贝         |

* 若LC_COLLATE为"POSIX"或"C"，则strcoll()与strcmp()作用完全相同 
* 按照 C94 及 C99 标准的规定，程序在启动时设置 locale 为 "C"。在 "C" locale 下，字符串的比较就是按照内码一个字节一个字节地进行，这时 strcoll 与 strcmp 函数没有区别。在其他 locale 下，字符串的比较方式则不同了，例如在简体中文 locale 下，strcmp 仍然按内码比较，而 strcoll 对于汉字则是按拼音进行的（这也跟操作系统有关，Windows 还支持按笔划排序，可以在“区域和语言设置”里面修改
  


# 4. STL容器

## 4.1. 容器种类  
STL 提供有 3 类标准容器，分别是序列容器、排序容器和哈希容器  其中后两类容器有时也统称为关联容器  
* 序列容器，是因为元素在容器中的位置同元素的值无关，即容器不是排序的。将元素插入容器时，指定在什么位置，元素就会位于什么位置
* 排序容器中的元素默认是由小到大排序好的，即便是插入元素，元素也会插入到适当位置。所以关联容器在查找时具有非常好的性能。
* 和排序容器不同，哈希容器中的元素是未排序的，元素的位置由哈希函数确定
| 容器种类 | 功能                                                                                                                                                       |
| -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 序列容器 | 主要包括 vector 向量容器、list 列表容器以及 deque 双端队列容器。                                                                                           |
| 排序容器 | 包括 set 集合容器、multiset多重集合容器、map映射容器以及 multimap 多重映射容器。                                                                           |
| 哈希容器 | C++ 11 新加入 4 种关联式容器，分别是 unordered_set 哈希集合、unordered_multiset 哈希多重集合、unordered_map 哈希映射以及 unordered_multimap 哈希多重映射。 |

### 4.1.1. 序列式容器  


* `array<T,N>`（数组容器）：表示可以存储 N 个 T 类型的元素，是 C++ 本身提供的一种容器。此类容器一旦建立，其长度就是固定不变的，这意味着不能增加或删除元素，只能改变某个元素的值；
* `vector<T>`（向量容器）：用来存放 T 类型的元素，是一个长度可变的序列容器，即在存储空间不足时，会自动申请更多的内存。使用此容器，在尾部增加或删除元素的效率最高（时间复杂度为 O(1) 常数阶），在其它位置插入或删除元素效率较差（时间复杂度为 O(n) 线性阶，其中 n 为容器中元素的个数）；
* `deque<T>`（双端队列容器）：和 vector 非常相似，区别在于使用该容器不仅尾部插入和删除元素高效，在头部插入或删除元素也同样高效，时间复杂度都是 O(1) 常数阶，但是在容器中某一位置处插入或删除元素，时间复杂度为 O(n) 线性阶；
* `list<T>`（链表容器）：是一个长度可变的、由 T 类型元素组成的序列，它以**双向链表**的形式组织元素，在这个序列的任何地方都可以高效地增加或删除元素（时间复杂度都为常数阶 O(1)），但访问容器中任意元素的速度要比前三种容器慢，这是因为 `list<T>` 必须从第一个元素或最后一个元素开始访问，需要沿着链表移动，直到到达想要的元素。
* `forward_list<T>`（正向链表容器）：和 list 容器非常类似，只不过它以**单链表**的形式组织元素，它内部的元素只能从第一个元素开始访问，是一类比链表容器快、更节省内存的容器。

    `stack<T> `和 `queue<T>` 本质上也属于序列容器，只不过它们都是在 deque 容器的基础上改头换面而成，通常更习惯称它们为容器适配器


### 4.1.2. 关联式容器

关联式容器在存储元素值的同时，还会为各元素额外再配备一个值（又称为“键”，其本质也是一个 C++ 基础数据类型或自定义类型的元素）  
它的功能是在使用关联式容器的过程中，如果已知目标元素的键的值，则直接通过该键就可以找到目标元素，而无需再通过遍历整个容器的方式。   

也就是说，使用关联式容器存储的元素，都是一个一个的“键值对”·`（ <key,value> ）`  
除此之外，序列式容器中存储的元素默认都是未经过排序的，而使用关联式容器存储的元素，默认会根据各元素的键值的大小做升序排序。   

* map 	    定义在`<map>` 头文件中，使用该容器存储的数据，其各个元素的键必须是唯一的（即不能重复），该容器会根据各元素键的大小，默认进行升序排序`（调用 std::less<T>）。`
* set   	定义在`<set>` 头文件中，使用该容器存储的数据，各个元素键和值完全相同，且各个元素的值不能重复（保证了各元素键的唯一性）。该容器会自动根据各个元素的键（其实也就是元素值）的大小进行升序* 排序`（调用 std::less<T>）。`
* multimap 	定义在 `<map>` 头文件中，和 map 容器唯一的不同在于，multimap 容器中存储元素的键可以重复。
* multiset 	定义在` <set>` 头文件中，和 set 容器唯一的不同在于，multiset 容器中存储元素的值可以重复（一旦值重复，则意味着键也是重复的）。

C++ 11 还新增了 4 种哈希容器，即 unordered_map、unordered_multimap 以及 unordered_set、unordered_multiset  
但哈希容器底层采用的是哈希表，而不是红黑树 (无序关联式容器)  




## 4.2. 迭代器

泛型思维发展的必然结果，类似中介的装置，它除了要具有对容器进行遍历读写数据的能力之外，还要能对外隐藏容器的内部差异，从而以统一的界面向算法传送数据。  

### 4.2.1. 迭代器的种类

STL 标准库为每一种标准容器定义了一种迭代器类型，这意味着，不同容器的迭代器也不同，其功能强弱也有所不同。  

**容器的迭代器的功能强弱** ， 决定了该容器是否支持 STL 中的某种算法。  

针对容器的迭代器有：前向迭代器、双向迭代器、随机访问迭代器  
针对输入输出流的迭代器有：输入迭代器、输出迭代器  

1. 前向迭代器（forward iterator）
    假设 p 是一个前向迭代器，则 p 支持 ++p，p++，*p 操作，还可以被复制或赋值，可以用 == 和 != 运算符进行比较。此外，两个正向迭代器可以互相赋值。

2. 双向迭代器（bidirectional iterator）
    双向迭代器具有正向迭代器的全部功能，除此之外，假设 p 是一个双向迭代器，则还可以进行 --p 或者 p-- 操作（即一次向后移动一个位置）。
    **双向迭代器不支持用“< >”进行比较**  

3. 随机访问迭代器（random access iterator）
    随机访问迭代器具有双向迭代器的全部功能。除此之外，假设 p 是一个随机访问迭代器，i 是一个整型变量或常量，则 p 还支持以下操作：
        p+=i：使得 p 往后移动 i 个元素。
        p-=i：使得 p 往前移动 i 个元素。
        p+i：返回 p 后面第 i 个元素的迭代器。
        p-i：返回 p 前面第 i 个元素的迭代器。
        p[i]：返回 p 后面第 i 个元素的引用。
    此外，两个随机访问迭代器 p1、p2 还可以用 <、>、<=、>= 运算符进行比较。另外，表达式 p2-p1 也是有定义的，其返回值表示 p2 所指向元素和 p1 所指向元素的序号之差（也可以说是 p2 和 p1 之间的元素个数减一）
  

不同容器的迭代器
| 容器                               | 对应的迭代器类型 |
| ---------------------------------- | ---------------- |
| array                              | 随机访问迭代器   |
| vector                             | 随机访问迭代器   |
| deque                              | 随机访问迭代器   |
| list                               | 双向迭代器       |
| set / multiset                     | 双向迭代器       |
| map / multimap                     | 双向迭代器       |
| forward_list                       | 前向迭代器       |
| unordered_map / unordered_multimap | 前向迭代器       |
| unordered_set / unordered_multiset | 前向迭代器       |
| stack                              | 不支持迭代器     |
| queue                              | 不支持迭代器     |

### 4.2.2. 定义中迭代器的种类  

迭代器的 4 种定义方式
| 迭代器定义方式 | 具体格式                                    |
| -------------- | ------------------------------------------- |
| 正向迭代器     | 容器类名::iterator  迭代器名;               |
| 常量正向迭代器 | 容器类名::const_iterator  迭代器名;         |
| 反向迭代器     | 容器类名::reverse_iterator  迭代器名;       |
| 常量反向迭代器 | 容器类名::const_reverse_iterator  迭代器名; |

读取它指向的元素，`*迭代器名` 就表示迭代器指向的元素  
* 对正向迭代器进行 ++ 操作时，迭代器会指向容器中的后一个元素；
* 而对反向迭代器进行 ++ 操作时，迭代器会指向容器中的前一个元素。
**注意，以上 4 种定义迭代器的方式**，并不是每个容器都适用。  
有一部分容器同时支持以上 4 种方式，比如 `array、deque、vector`；  
而有些容器只支持其中部分的定义方式，例如 `forward_list` 容器只支持定义正向迭代器，不支持定义反向迭代器。  

### 4.2.3. 和迭代器有关的成员函数  

| 函数成员  | 函数功能                                                                          |
| --------- | --------------------------------------------------------------------------------- |
| begin()   | 返回指向容器中第一个元素的迭代器。                                                |
| end()     | 返回指向容器最后一个元素所在位置**后一个位置**的迭代器，通常和 begin() 结合使用。 |
| rbegin()  | 返回指向最后一个元素的迭代器。                                                    |
| rend()    | 返回指向第一个元素所在位置**前一个位置**的迭代器。                                |
| cbegin()  | 和 begin() 功能相同，只不过在其基础上，增加了 const 属性，不能用于修改元素。      |
| cend()    | 和 end() 功能相同，只不过在其基础上，增加了 const 属性，不能用于修改元素。        |
| crbegin() | 和 rbegin() 功能相同，只不过在其基础上，增加了 const 属性，不能用于修改元素。     |
| crend()   | 和 rend() 功能相同，只不过在其基础上，增加了 const 属性，不能用于修改元素。       |

以上函数在实际使用时，其返回值类型都可以使用 auto 关键字代替，编译器可以自行判断出该迭代器的类型。


## 4.3. array 升级的数组  

rray 容器是 C++ 11 标准中新增的序列容器，它就是在 C++ 普通数组的基础上，添加了一些成员函数和全局函数。  
在使用上，它比普通数组更安全，且效率并没有因此变差。  

```cpp
#include <array>
//在 array<T,N> 类模板中，T 用于指明容器中的存储的具体数据类型，N 用于指明容器的大小，
//需要注意的是，这里的 N 必须是常量，不能用变量表示, 和普通数组一样。
std::array<double, 10> values;
//这种方式创建的容器中，各个元素的值是不确定的（array 容器不会做默认初始化操作）。

//将所有的元素初始化为 0 或者和默认元素类型等效的值：
std::array<double, 10> values {};

//像创建常规数组那样对元素进行初始化：
std::array<double, 10> values {0.5,1.0,1.5,,2.0};
//剩余的元素都会被初始化为 0.0

```
### 4.3.1. 安全的访问  

1. begin() end()
    C++ 11 标准库还新增加了 begin() 和 end() 这 2 个函数，和 array 容器包含的 begin() 和 end() 成员函数不同的是，标准库提供的这 2 个函数的操作对象，既可以是容器，还可以是普通数组。  
    如果操作对象是普通数组，则 begin() 函数返回的是指向数组第一个元素的指针，同样 end() 返回指向数组中最后一个元素之后一个位置的指针（注意不是最后一个元素）。
```cpp
    auto first = std::begin(values);
    auto last = std::end (values);
```
2. get()  `<array>` 头文件中重载了 get() 全局函数，该重载函数的功能是访问容器中指定的元素，并返回该元素的**引用**。
3. at() 返回容器中 n 位置处元素的引用，该函数自动检查 n 是否在有效的范围内，如果不是则抛出 out_of_range 异常。

```cpp
#include <iostream>
//需要引入 array 头文件
#include <array>
int main()
{
    std::array<int, 4> values{};
    //初始化 values 容器为 {0,1,2,3}, 用at 访问并更改值
    for (int i = 0; i < values.size(); i++) {
        values.at(i) = i;
    }

    //使用 get() 重载函数输出指定位置元素
    cout << get<3>(values) << endl;
    //如果容器不为空，则输出容器中所有的元素
    if (!values.empty()) {
        for (auto val = values.begin(); val < values.end(); val++) {
            cout << *val << " ";
        }
    }
}


```
### 4.3.2. array使用方法总结  

访问array容器中单个元素
1. 可以通过容器名[]的方式直接访问和使用容器中的元素，这和 C++ 标准数组访问元素的方式相同
2. `get<n>` 模板函数，它是一个辅助函数，能够获取到容器的第 n 个元素 , 只能访问模板参数指定的元素，编译器在编译时会对它进行检查
3. array 容器提供了 data() 成员函数，通过调用该函数可以得到指向容器首个元素的指针。
```cpp

values[4] = values[3] + 2.O*values[1];
// 使用 at 越界时  会抛出 std::out_of_range 异常
values.at (4) = values.at(3) + 2.O*values.at(1);

cout << get<3>(words) << endl;

cout << *( words.data()+1);
```

访问array容器中多个元素

1. size() 函数能够返回容器中元素的个数（函数返回值为 size_t 类型）
2.  empty() 成员函数，即可知道容器中有没有元素（如果容器中没有元素，此函数返回 true）

```cpp
if(values.empty())
    std::cout << "The container has no elements.\n";
else
    std::cout << "The container has "<< values.size()<<"elements.\n";
```

## 4.4. vector 向量容器

它和 array 容器非常类似，都可以看做是对 C++ 普通数组的“升级版”。不同之处在于，array 实现的是静态数组（容量固定的数组），而 vector 实现的是一个动态数组，即可以进行元素的插入和删除，在此过程中，vector 会动态调整所占用的内存空间，整个过程无需人工干预  

vector 常被称为向量容器，因为`该容器擅长在尾部插入或删除元素`，`时间复杂度为O(1)`；而对于在容器头部或者中部插入或删除元素，则花费时间要长一些（移动元素需要耗费时间），时间复杂度为线性阶`O(n)`。  


vector的源码实现
```cpp
/*
vector就是使用 3 个迭代器（可以理解成指针）来表示的
_Myfirst 指向的是 vector 容器对象的起始字节位置
_Mylast 指向当前最后一个元素的末尾字节
_Myend 指向整个 vector 容器所占用内存空间的末尾字节。

*/
class vector{
    ...
protected:
    pointer _Myfirst;
    pointer _Mylast;
    pointer _Myend;
};
```
当 vector 的大小和容量相等（size==capacity）也就是满载时，如果再向其添加元素，那么 vector 就需要扩容。vector 容器扩容的过程需要经历以下 3 步：
1. 完全弃用现有的内存空间，重新申请更大的内存空间；
2. 将旧内存空间中的数据，按原有顺序移动到新的内存空间中；
3. 最后将旧的内存空间释放。

vector 容器扩容时，不同的编译器申请更多内存空间的量是不同的。以 VS 为例，它会扩容现有容器容量的 50%。  


### 4.4.1. 创建

```cpp
// 最基础的定义
std::vector<double> values;

// 使用 reserve() 来重新分配空间,增加容量
// 重新分配空间后可能导致之前定义的迭代器失效
values.reserve(20);

// 用类似数组的方法定义初始值
std::vector<int> primes {2, 3, 5, 7, 11, 13, 17, 19};
// 定义初始的元素个数
std::vector<double> values(20);

// 定义初始元素个数以及统一的初始值
std::vector<double> values(20, 1.0);

// 和 array 不同, 定义 vector 的时候, 元素个数和初始值都可以是变量
std::vector<double> values(num, value);


// vector的拷贝构造
//可以用一对指针或者迭代器来指定拷贝初始值的范围
int array[]={1,2,3};
std::vector<int>values(array, array+2);//values 将保存{1,2}
std::vector<int>value1{1,2,3,4,5};
std::vector<int>value2(std::begin(value1),std::begin(value1)+3);//value2保存{1,2,3}


```
### 4.4.2. 访问修改元素

1. 下标访问 (有越界可能)
2. at() 访问(抛出 std::out_of_range 异常)
3. front() back() 访问首尾元素 ( 返回引用, 可以修改值 )
4. data 访问
```cpp
// 使用下标进行访问和修改
values[0] = values[1] + values[2];

// at() 方法访问
values.at(0) = values.at(1);

//修改首元素
values.front() = 10;

//输出容器中第 3 个元素的值
cout << *(values.data() + 2) << endl;
```

### 4.4.3. 添加元素

1. push_back() 和 emplace_back() 插入尾部
   * push_back() 向容器尾部添加元素时，首先会创建这个元素，然后再将这个元素拷贝或者移动到容器中（如果是拷贝的话，事后会自行销毁先前创建的这个元素）
   * emplace_back() 在实现时，则是直接在容器尾部创建这个元素，省去了拷贝或移动元素的过程。
   * emplace_back()该函数是 C++ 11 新增加的
   * emplace_back() 的执行效率比 push_back() 高。因此，在实际使用时，建议大家优先选用 emplace_back()。
   * 如果程序要兼顾之前的版本，还是应该使用 push_back()。
2. insert() 和 emplace() 插入指定位置

* insert
    iterator insert(pos,elem)           在迭代器 pos 指定的位置之前插入`一个`新元素elem，并返回表示新插入元素位置的迭代器。  
    iterator insert(pos,n,elem)         在迭代器 pos 指定的位置之前插入 `n 个`元素 elem，并返回表示第一个新插入元素位置的迭代器。  
    iterator insert(pos,first,last)     在迭代器 pos 指定的位置之前，插入`其他容器（不仅限于vector）`中位于` [first,last) `区域的所有元素，并返回表示第一个新插入元素位置的迭代器。  
    iterator insert(pos,initlist)       在迭代器 pos 指定的位置之前，插入初始化列表（用大括号{}括起来的多个元素，中间有逗号隔开）中所有的元素，并返回表示第一个新插入元素位置的迭代器。  
* emplace()
  * emplace() 是 C++ 11 标准新增加的成员函数，用于在 vector 容器指定位置之前插入一个新的元素
  * emplace() 每次只能插入一个元素，而不是多个。
  * 同理, emplace() 在插入元素时，是在容器的指定位置直接构造元素，而不是先单独生成，再将其复制（或移动）到容器中。因此，在实际使用中，推荐大家优先使用 emplace()。
`iterator emplace (const_iterator pos, args...);`  
pos 为指定插入位置的迭代器；`args..`. 表示与新插入元素的`构造函数`相对应的多个参数；该函数会返回表示新插入元素位置的迭代器。  

```cpp
std::vector<int> demo{1,2}; // 1 2 

//第一种格式用法
demo.insert(demo.begin() + 1, 3);//{1,3,2}

//第二种格式用法
demo.insert(demo.end(), 2, 5);//{1,3,2,5,5}

//第三种格式用法
std::array<int,3>test{ 7,8,9 };
demo.insert(demo.end(), test.begin(), test.end());//{1,3,2,5,5,7,8,9}

//第四种格式用法
demo.insert(demo.end(), { 10,11 });//{1,3,2,5,5,7,8,9,10,11}

// emplace
demo1.emplace(demo1.begin(), 3);
```

### 4.4.4. 删除元素

成员函数  
* pop_back()     | 删除 vector 容器中最后一个元素，该容器的大小（size）会减 1，但容量（capacity）不会发生改变。
* erase(pos)     | 删除 vector 容器中 pos 迭代器指定位置处的元素，并返回指向被删除元素`下一个`位置元素的迭代器。该容器的大小（size）会减 1，但容量（capacity）不会发生改变。
* erase(beg,end) | 删除 vector 容器中位于迭代器 [beg,end)指定区域内的所有元素，并返回指向被删除区域`下一个`位置元素的迭代器。该容器的大小（size）会减小，但容量（capacity）不会发生改变。
* clear() |删除 vector 容器中所有的元素，使其变成空的 vector 容器。该函数会改变 vector 的大小（变为 0），但不是改变其容量。


remove() 函数，该函数定义在` <algorithm>` 头文件中。
remove()用于删除容器中指定元素时，常和 erase() 成员函数搭配使用。
* remove() |删除容器中所有`和指定元素值相等的元素`，并返回指向最后一个元素下一个位置的迭代器。值得一提的是，调用该函数不会改变容器的大小和容量。
* 因此无法使用之前的方法遍历容器，而是需要向程序中那样，借助 remove() 返回的迭代器完成正确的遍历。
  * remove() 的实现原理是，在遍历容器中的元素时，一旦遇到目标元素，就做上标记，然后继续遍历，直到找到一个非目标元素，即用此元素将最先做标记的位置覆盖掉，
  * 同时将`此非目标元素所在的位置也做上标记`，等待找到新的非目标元素将其覆盖

```cpp

vector<int>demo{ 1,3,3,4,3,5 };

auto iter = std::remove(demo.begin(), demo.end(), 3); // 1 4 5 4 3 5

// 删除排在后面的无用元素  
demo.erase(iter, demo.end());
```

swap() 函数在头文件 `<algorithm>` 和 `<utility>` 中都有定义，使用时引入其中一个即可   
* swap(beg)、pop_back() |先调用 swap() 函数交换要删除的目标元素和容器最后一个元素的位置，然后使用 pop_back() 删除该目标元素。


### 4.4.5. 迭代器

1. 8大迭代器生成函数, 使用auto接受返回的迭代器很方便, begin和end()配合迭代全部元素
2. 可以使用全局的 begin(vector) 和 end(vector) 获取迭代器  
3. size() 配合下标访问全部元素  

      capacity() 成员函数，可以获得当前容器的容量  
      通过 size() 成员函数，可以获得容器当前的大小。  


* 可以调用` reserve()` 成员函数来增加容器的容量（但并不会改变存储元素的个数）
  * 仅仅只是用来增加容量
* 通过调用成员函数` resize() `可以改变容器的大小，并且该函数也可能会导致 vector 容器容量的增加。
  * 会同时生成对应的初始值, 即调用后容器内就有多少个元素

```cpp
value.reserve(20);
cout << "value 容量是(2)：" << value.capacity() << endl;    // 20 当前的容量
cout << "value 大小是(2)：" << value.size() << endl;        // 15 个元素

//将元素个数改变为 21 个，所以会增加 6 个默认初始化的元素
value.resize(21);

//将元素个数改变为 21 个，新增加的 6 个元素默认值为 99。
value.resize(21,99);

//当需要减小容器的大小时，会移除多余的元素。
value.resize(20);
```


一旦 vector 容器的内存被重新分配，则和 vector 容器中元素相关的所有引用、指针以及迭代器，都可能会失效，最稳妥的方法就是重新生成。   

对于 vector 容器而言，当增加新的元素时，有可能很快完成（即直接存在预留空间中）；也有可能会慢一些（扩容之后再放新元素）。  

若需要保存容器的容量或者大小,尽量使用auto  
vector<T> 对象的容量和大小类型都是 vector<T>::size_type  

```cpp
vector<int>::size_type cap = value.capacity();
vector<int>::size_type size = value.size();

auto cap = value.capacity();
auto size = value.size();
```

### 4.4.6. vector 的容量高级操作

vector 容器的扩容过程是非常耗时的，并且当容器进行扩容后，之前和该容器相关的所有指针、迭代器以及引用都会失效。因此在使用 vector 容器过程中，我们应尽量避免执行不必要的扩容操作。
1. 分配一块大小是当前 vector 容量几倍的新存储空间。注意，多数 STL 版本中的 vector 容器，其容器都会以 2 的倍数增长，也就是说，每次 vector 容器扩容，它们的容量都会提高到之前的 2 倍；
2. 将 vector 容器存储的所有元素，依照原有次序从旧的存储空间复制到新的存储空间中；
3. 析构掉旧存储空间中存储的所有元素；
4. 释放旧的存储空间。

resize(n)
* 强制 vector 容器必须存储 n 个元素，注意，
* 如果 n 比 size() 的返回值小，则容器尾部多出的元素将会被析构（删除）
* 如果 n 比 size() 大，则 vector 会借助默认构造函数创建出更多的默认值元素，并将它们存储到容器末尾
* 如果 n 比 capacity() 的返回值还要大，则 vector 会先扩增，在添加一些默认值元素。

reserve(n) 	强制 vector 容器的容量至少为 n。注意，如果 n 比当前 vector 容器的容量小，则该方法什么也不会做；反之如果 n 比当前 vector 容器的容量大，则 vector 容器就会扩容。

* 避免 vector 容器执行不必要的扩容操作的关键在于，在使用 vector 容器初期，就要将其容量设为足够大的值
* 在 vector 容器刚刚构造出来的那一刻，就应该借助 reserve() 成员方法为其扩充足够大的容量。

### 4.4.7. vector 的容量 fit


```cpp

// 使用 shrink_to_fit() 来调整容量到当前的元素内容大小
vector<int>myvector;
myvector.shrink_to_fit();

// 使用 swap 成员函数来
// 该方法的基础功能是交换 2 个相同类型的 vector 容器（交换容量和存储的所有元素）
vector<int>(myvector).swap(myvector);

//清空 myvector 容器
vector<int>().swap(myvector);
```

swap()成员函数的执行流程:  
1) 先执行 vector<int>(myvector)，此表达式会调用 vector 模板类中的拷贝构造函数，从而创建出一个临时的 vector 容器（后续称其为 tempvector）。
2) 值得一提的是，tempvector 临时容器并不为空，因为我们将 myvector 作为参数传递给了复制构造函数，该函数会将 myvector 容器中的所有元素拷贝一份，并存储到 tempvector 临时容器中。

    注意，vector 模板类中的拷贝构造函数只会为拷贝的元素分配存储空间。换句话说，tempvector 临时容器中没有空闲的存储空间，其容量等于存储元素的个数。
3) 然后借助 swap() 成员方法对 tempvector 临时容器和 myvector 容器进行调换，此过程不仅会交换 2 个容器存储的元素，还会交换它们的容量。换句话说经过 swap() 操作，myvetor 容器具有了 tempvector 临时容器存储的所有元素和容量，同时 tempvector 也具有了原 myvector 容器存储的所有元素和容量。
4) 当整条语句执行结束时，临时的 tempvector 容器会被销毁，其占据的存储空间都会被释放。注意，这里释放的其实是原 myvector 容器占用的存储空间。
经过以上步骤，就成功的将 myvector 容器的容量由 100 缩减至 10。  


## 4.5. deque

主要要点:
1. deque 是 double-ended queue 的缩写，又称双端队列容器。
2. deque 容器中存储元素并`不能保证`所有元素都存储到连续的内存空间中。
3. deque 容器也擅长在序列尾部添加或删除元素（时间复杂度为O(1)），而不擅长在序列中间添加或删除元素。
4. deque 容器也可以根据需要修改自身的容量和大小。

当需要向序列两端频繁的添加或删除元素时，应首选 deque 容器。  


## 4.6. pair 

考虑到“键值对”并不是普通类型数据，C++ STL 标准库提供了 pair 类模板  
专门用来将 2 个普通元素 first 和 second  创建成一个新元素`<first, second>`    

* 注意: pair 类模板定义在 `<utility>` 头文件中

### 4.6.1. 构造函数

```cpp
// c++ 11 标准之前
// 默认构造函数，即创建空的 pair 对象
pair();

// 直接使用 2 个元素初始化成 pair 对象
pair (const first_type& a, const second_type& b);

// 拷贝（复制）构造函数，即借助另一个 pair 对象，创建新的 pair 对象
template<class U, class V> pair (const pair<U,V>& pr);

```


## 4.7. map 

### 4.7.1. 与 unordered_map 的区别与关联

1. 二者的头文件不同
```cpp
//map
#include < map >
//unordered_map
#include < unordered_map >
```

2. 内部实现机理不同

      map
      * map内部实现了一个红黑树（红黑树是非严格平衡二叉搜索树，而AVL是严格平衡二叉搜索树）
      * 红黑树具有自动排序的功能，因此map内部的所有元素都是有序的，红黑树的每一个节点都代表着map的一个元素
      * 因此，对于map进行的查找，删除，添加等一系列的操作都相当于是对红黑树进行的操作

      unordered_map
      * unordered_map内部实现了一个哈希表（也叫散列表，通过把关键码值映射到Hash表中一个位置来访问记录，查找的时间复杂度可达到O(1)
      * 其在海量数据处理中有着广泛应用）
      * 因此，其元素的排列顺序是无序的
  
3. 优缺点以及适用处
  
**map的优点**:
* 有序性，这是map结构最大的优点，其元素的有序性在很多应用中都会简化很多的操作
* 红黑树，内部实现一个红黑书使得map的很多操作在lg(n)的时间复杂度下就可以实现，因此效率非常的高
**map的缺点**:
* 空间占用率高，因为map内部实现了红黑树，虽然提高了运行效率，但是因为每一个节点都需要额外保存父节点、孩子节点和红/黑性质，使得每一个节点都占用大量的空间

**map适用处**:
* 对于那些有顺序要求的问题，用map会更高效一些


**unordered_map**:  
优点:
* 因为内部实现了哈希表，因此其查找速度非常的快
缺点: 
* 哈希表的建立比较耗费时间
适用处:
* 对于查找问题，unordered_map会更加高效一些
* 遇到查找问题，常会考虑一下用unordered_map


**总结**:  
* 内存占有率的问题就转化成红黑树 VS hash表 , 还是unorder_map占用的内存要高
* 但是unordered_map执行效率要比map高很多
* 对于unordered_map或unordered_set容器，其遍历顺序与创建该容器时输入的顺序不一定相同，因为遍历是按照哈希表从前往后依次遍历的
  
**map和unordered_map的使用**:  
* unordered_map的用法和map是一样的，提供了 insert，size，count等操作，并且里面的元素也是以pair类型来存贮的
* 其底层实现是完全不同的，上方已经解释了
* 外部使用来说却是一致的。

### 4.7.2. 构造函数

```cpp
map<int, string> mapStudent;
```

### 4.7.3. 获取值

map 类模板中对[ ]运算符进行了重载，这意味着，类似于借助数组下标可以直接访问数组中元素，通过指定的键，我们可以轻松获取 map 容器中该键对应的值
1. 有当 map 容器中确实存有包含该指定键的键值对，借助重载的 [ ] 运算符才能成功获取该键对应的值
2. 若当前 map 容器中没有包含该指定键的键值对，则此时使用 [ ] 运算符将不再是访问容器中的元素，而变成了向该 map 容器中增添一个键值对
   * 键用 [ ] 运算符中指定的键
   * 值取决于值的数据类型，如果是基本数据类型，则值为 0；如果是 string 类型，其值为 ""，即空字符串（即使用该类型的默认值作为键值对的值）
  
调用  at() 成员方法
1. 成功找到键的键值对，并返回该键对应的值
2. 没有键的键值对，会导致 at() 成员方法查找失败，并抛出 `out_of_range` 异常   

借助 find() 成员方法间接实现此目的
1. 该方法返回的是一个迭代器
2. 如果查找成功，该迭代器指向查找到的键值对
3. 反之，则指向 map 容器最后一个键值对之后的位置（和 end() 成功方法返回的迭代器一样
   
```cpp
// 两个函数  find()  和 end()
// find 返回迭代器指向当前查找元素的位置否则返回map::end()位置
iter = mapStudent.find("123");
if(iter != mapStudent.end())
  cout<<"Find, the value is"<<iter->second<<endl;
else
  cout<<"Do not Find"<<endl;

```
find 函数返回迭代器对象 , 可以使用 `->` 直接获取数据


### 4.7.4. 元素插入

```cpp
// 定义一个map对象
map<int, string> mapStudent;
 
// 第一种 用insert函數插入 pair
mapStudent.insert(pair<int, string>(000, "student_zero"));
 
// 第二种 用insert函数插入 value_type 数据
mapStudent.insert(map<int, string>::value_type(001, "student_one"));
 
// 第三种 用"array"方式插入
mapStudent[123] = "student_first";
mapStudent[456] = "student_second";

// 关于 Insert 函数的返回值

// 这是 insert 函数的构造定义，返回一个pair对象
pair<iterator,bool> insert (const value_type& val);
// 要获取是否成功 , 需要建立一个对应的 pair 
pair<map<int, string/*这里需要和原本的map对象相同*/>::iterator, bool> Insert_Pair;
Insert_Pair = mapStudent.insert(map<int, string>::value_type (001, "student_one"));
//如果插入成功的话Insert_Pair.second应该是true的，否则为false
if(!Insert_Pair.second)cout << ""Error insert new element" << endl;

```
使用 insert 函数插入和使用数组方式插入主要区别在于数据的唯一性  
若关键字已存在 , insert 会返回插入失败 
而数组方式会直接覆盖  

### 4.7.5. 删除元素 
```cpp
//主要通过函数 erase() , 删除成功返回 1 , 否则返回 0

//迭代器作为参数刪除
iter = mapStudent.find("123");
mapStudent.erase(iter);

//用关键字作为参数刪除
int n = mapStudent.erase("123");

//两个迭代器对象作为参数, 表示范围删除 , 下行代码表示清空 map
mapStudent.erase(mapStudent.begin(), mapStudent.end());
//等同于
mapStudent.clear()
```
### 4.7.6. 获取大小

```cpp
//函数定义
size_type size() const noexcept;

//一般使用方法
int nSize = mapStudent.size();
```
### 4.7.7. 其他常用函数
begin()         返回指向map头部的迭代器  
end()           返回指向map末尾的迭代器  
rbegin()        返回一个指向map尾部的逆向迭代器  
rend()          返回一个指向map头部的逆向迭代器

count()         返回指定元素出现的次数  
empty()         如果map为空则返回true  
equal_range()   返回特殊条目的迭代器对  
get_allocator() 返回map的配置器  
key_comp()      返回比较元素key的函数  
lower_bound()   返回键值>=给定元素的第一个位置  
max_size()      返回可以容纳的最大元素个数  
size()          返回map中元素的个数  
swap()           交换两个map  
upper_bound()    返回键值>给定元素的第一个位置  
value_comp()     返回比较元素value的函数  