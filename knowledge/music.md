# 1. 李重光乐理笔记

音的性质
* 人耳能听到的声音 11-20000Hz
* 音乐中经常使用的频率 27-4100Hz

音的性质:
* 高低
* 长短
* 强软
* 音色

乐音与噪音:
* 震动规则, 音的高低十分明显, 叫乐音
* 震动不规则, 高低不明显的叫噪音
* 音乐中不单单只会包含乐音, 噪音也很重要
* 基本乐音的总和, 就构成了 "音乐体系"
* 88键钢琴基本包括了音乐体系中所有的乐音, 是除了管风琴以外音域最广的乐器
  * A2 (27.5) -> c5 (4186)

音名与音组:
* 唱名: do, re, mi, fa, sol, la, si
* 音名: C D E F G A B, 叫做基本音级
* 唱名会根据具体音高改变, 音名不会改变
* 音乐体系上有80多个音, 但基本音名就 7 个
* 音组: 根据具体音高, 在钢琴上可以把音按照12音律分组
  * 在钢琴上: 
  * 大字二组        超低音    A_2     (3) 个键 (la la# si)
  * 大字一组        低音      C_1
  * 大字组                    C
  * 小字组          中音      c
  * 小字一组        中音      c^1
  * 小字二组        中音    c^2
  * 小字三组        高音    c^3
  * 小子四组        高音         c^4
  * 小字五组 (1) 个键 do      c^5
  * 大二和小五都是不完全的 
* 随着调式的变化, 可以将音描述为 升/降  重生/重降


音域: 用于指代一个乐器或者人声所能发出的音的总范围
* 钢琴  : A_2   ->  c^5


## 1.1. 音律

国际标准音: 以小字一组的 A 每秒振动 440 次作为标准, 即 a^1  (中音la)
中央C    : 即小字一组的 C (c^1)


复合音:
* 一般的声音都是又许多声音组合而成的, 即复合音
* 对于弦乐器来说, 除去全弦振动以外, 每一部分同时也在震动
* 全弦振动的音一般最响, 称为`基音`
* 弦的各部分振动产生的音, 一般不容易被听出, 成为 `泛音`
  * 波的分解
* 基因和泛音以高低次序排列起来, 就是分音列


几种音律: 将纯八度 (如C^1到C^2)的频率区间, 怎么细分出其他的音
* 纯八度    : 八度之间频率相差一倍, 即 1:2 , 成为 和谐
* 五度      : 五度之间频率 2:3 , 称为次和谐
* 十二平均律: 12个均等的部分 (半音)
* 五度相生律: 被淘汰了
* 纯律     : 被淘汰了

# 2. 记谱法

五线谱:
* 从下往上依次是          : 一线 到 五线
* 五根线中间从下往上依次是 : 一间 到 四间

谱号: 常用谱号只有三种, 根据谱号放在哪一个线上决定每个线的具体的音
* 高音谱号 G clef `𝄞, U+1D11E` : 代表 g1, 标准五线谱的G谱号会放在二线上, 表示二线是 sol
* 中音谱号 C clef `𝄡, U+1D121` : 代表 c1, 即中央C
* 低音谱号 F clef `𝄢, U+1D122` : 代表 f 



变音记号: 用于音级的升高或者降低以及还原
* 因为五线谱的线间之间都是一个音名差, 不是相等的半音或者全音差
* 调号    : 将变音记号记在谱号后面时, 称作调号

音符:
* 全音符    : 圆
* 二分音符  : 空心符头
* 四分音符  : 实心符头
* 八分音符  : 带符尾
* 16分      : 2符尾
* 32分      : 3符尾
* 64分      : 4符尾

休止符:
* 全休止符  : 四线下方
* 二分休止符: 三线上方
* 四分休止符: 特殊符号
* 八分休止符: 带符头的记号
* 16分    : 2符头
* 32分    : 3符头
* 64分    : 4符头

付点: 
* 单付点  : 延长二分之一音
* 双付点  : 延长四分之三音

延长记号: 在音符的上方, 小弧线, 中间加一点, 表示根据演奏需要自由延长音符时值 

延音线  : 记录在两个同样音高的音上, 表示时值相加的音符, 只弹奏一次, 用于表示比较特殊的时值


## 2.1. 省略记号

省略记号: 因为音域太广, 有时候需要省略简便的符号记录较远的改动

* 移动8度记号 : 用数字`8------`代表虚线内的音符移动8度, 记在五线谱上方表示升高, 下方表示下降
* 重复8度记号 : 用数字`8`写在要重复的单音上下, 表示`同时`演奏对应差8度的音, 上方表示高, 下方表示低8度
* 长时间重复8度记号 : 用字母 `Con----` 表示, 同理五线谱上方表示高8度, 下方表低8度
* 长休止符号  : 休止许多小结的时候用, 用长横线写在 `三线上 (不是上方)`, 两边加上短竖线, 在横线中央上方写上要休止的小节数
* 震音记号 : 短斜线 相当于音符追加的符尾个数
  * 用于表示一个长音的奏法是许多短音的重复
  * 单斜线: 用八分音符演奏
  * 双斜线: 用16分音符演奏
  * 斜线可以记在两个音符中间, 代表两个音为组合的震音
* 反复记号: 代表某一部分的重复演奏, 用斜线画在谱内表示
  * 斜线的数目代表符尾数
  * 斜线旁边加两点表示以小节为单位重复

## 2.2. 奏法记号

* 连音奏法  : 大弧线表示, 弧线内的音要连贯演奏
* 断音奏法  : 用小圆点或者小三角, 代表音之间要制音, 不连续
  * 连音和短音同时出现, 代表稍微连续一下
* 琶音奏法  : 垂直的波浪线, 默认从下往上琶音, 带箭头的话则是箭头方向, 一般都是和弦上的音
* 滑音奏法  : 小弯钩带箭头, 弯钩的的方向代表从低音划过来还是高音划过来
  * 滑音符号记录在小音符上, 代表装饰音奏法
* 短倚音    : 就是小音符, 装饰音, 一般都是32分时值


# 3. 和弦

## 3.1. 音程 - 和弦的基础

`两个音`的高低关系叫做音程
* 根音: 较低的音
* 冠音: 较高的音
* 旋律音程: 先后发声的两个音, 分上/下行音程, 一般不加以说明的就是上行, 如果是下行音程需要标注 例: 下行纯四度 
* 和声音程: 同时发生的两个音


音程的命名
* 度数: 五线谱上两个音之间 线与间的数目 (音名相差的个数+1), 相同音的音程为 1度, 升音降音不改变度数
* 音数: 两个音之间, 相差的全音个数, do到升do为8度 = 6音, 半音为 0.5 音数
* 要确定一个音程, 必须从音数和度数两方面考量
  * 在确定音程的时候不能进行等音替换 `E# - bF`, 音程是不同的

基本(自然)音程的命名 : 纯 增 减 小 大 , 每种度数共 7 个音程, 根据是否包括半音以及半音的个数分配名称
* 纯一度  : 音相同的音程构成纯一度, 音数0 音程1 , 7个基础音构成了 7个纯一度
* 增一度  : 音程1, 音数 1/2, 所有升音都构成纯一度, 当然除了 E B没有升音以外 (增一度不是自然音程)
* 小二度  : 音程2, 音数 1/2, 只存在两个小二度, `E-F` 和 `B-C`
* 大二度  : 音程2, 音数 1, 除了小二度的两个音程, 剩下5个都是大二度
* 小三度  : 音程3, 音数 1.5 , 包括了 `E-F B-C` 音程共产生了 4 个小三度, `D-F E-G A-C B-D`
* 大三度  : 音程3, 音数 2, 共三个 `C-E` `F-A` `G-B`
* 纯四度  : 音程4, 音数 2.5, 四度往上必然会包括 `E-F` 或 `B-C` 因此不是`小`而是`纯`, 有 `C-F D-G E-A` `G-C A-D B-E` 共计6个
* 增四度  : 音程4, 音数 3, 中间的全都是间隔全音, 只存在一个 `F-B`, 又称作 `三全音` 
* 减五度  : 音程5, 音数 3, 同时包括了两个半音 的最短音程, 即 `B-F`
* 纯五度  : 音程5, 音数 3.5, 只包括一个 `E-F B-C` 的五度音程, 同纯四度一样有6个 `C-G D-A E-B` `F-C G-D A-E`
* 小六度  : 音程6, 音数 4, 同时包括了两个半音的三个音程, 有 `E-C A-F B-G`
* 大六度  : 音程6, 音数 4.5, 只包括一个 半音 的六度音程, 共有4个 `F-D G-E` `C-A D-B`
* 小七度  : 音程7, 音数 5, 同时包括了 `E-F B-C` 的七度音程, 有 5 个 `D-C E-D G-F A-G B-A`
* 大七度  : 音程7, 音数 5.5, 除去小7度的5个只剩下两个, 只包括一个半音音程的七度 `C-B  F-E`
* 纯八度  : 音程8, 音数 6 , 无需特别记忆

* 总结:
  * 2,3,6,7 分大小 : 
    * 2,3 包括了半音的音程叫 小, 否则大
    * 6,7 同时包括两个半音的叫 小, 否则大
  * 1, 4, 5 , 8 纯 , 附带 增4 减5
    * 增一度除去 `E-F` `B-C`
    * 增四度 : 唯一的 三全音
    * 减五度 : 唯一的 五度双半音 `B-F`

由于音还会分升降, 因此音程的命名也会随之扩充  
* 在维持音名不变的情况下, 根据音数的增减, 文字会以0.5音数为单位进行变化
* 倍减 - 减 - (小 - 大)纯 - 增 - 倍增

复音程, 在单音程的基础上, 超过8度的音程叫做复音程, 在计算上不需要改变 增减大小纯, 只需要在音程上加8即可  

## 3.2. 音程的协和

就是听起来融合不融合来对音程进行分类, 不同的音乐理论体系, 分类是不同的


协和音程, 分三类
* 极完全协和音程 (纯一度, 纯八度) , 属于特殊分类
* 完全协和音程 (纯四度, 纯五度) 
* 不完全协和音程 (大小三度, 大小六度)  大小36  

不协和音程 : 除了协和音程以外的都是不协和, 大小27, 所有增减, 倍增, 倍减

音程的转位: 音程的上方音与下方音互相颠倒, 可以在一个八度内, 也可以超过八度
* 下方音升高八度, 或倍八度
* 上方音降低八度, 或倍八度
* 上下音同时进行移动
* 转位时, 音程也会顺带转换
  * 18,27,36,45 互换, 即和为9
  * 大小, 增减, 倍增减, 互换,  纯保持纯
  * 特例: 增八度转换后 -> 减八度
* 音程转位不会影响音程的协和属性

等音程: 由于音的叫法不同, 升/重升等, 音程也会随之改变, 但是音本身是相同的情况下(升do和降re), 产生的音程之间互为等音程
* 需要配合调式来理解
* 等音程, 在有些调式里是不协和音程, 在有些调式里就成了协和音程

## 3.3. 和弦

主要按照三度音程关系, 三个以上的音, 构成和弦  

三和弦 : 分 大小增减 四类, 由 3 个三度关系的音构成
* 一个三和弦的音从低到高分别是 : 根音, 三音, 五音, 一般用数字 135 来表示, 根音可以写成 r 
* 因为三度自然音程只有大小三度两种, 那么根音三音的音程和三音五音的音程构成了4种排列组合
  * 大三+小三 = 大三和弦  此时根音到五音为纯五度   比较恢宏大气
  * 小三+大三 = 小三和弦  此时根音到五音为纯五度   比较阴柔忧伤
  * 小三+小三 = 减三和弦  此时根音到五音为减五度   比较收缩 阴森, 不协和三和弦
  * 大三+大三 = 增三和弦  此时根因到五音为增五度   不好描述  不协和三和弦
  * 作曲中会用到增减三和弦的情况较少

七和弦 : 在三和弦的基础上再加上一个音, 4个三度关系的音构成, `所有的7和弦都是不协和的`
* 七和弦的音为 : 根音 三五七音
* 七和弦的命名以及种类:
  * 根据 根三五音构成的三和弦种类来决定第一个标识 `大小增减`
  * 根据 根音到7音的七度音程种类来决定第二个表示 `大小增减`
  * 总共构成了 `9` 种
* 具体的七和弦的表:
  * 大小七和弦  : 大三和弦, 小七度, 5724
  * 小小七和弦  : 小三和弦, 小七度, 2461, 3572, 6135
  * 减小七和弦  : 减三和弦, 小七度, 又称半减七和弦.  减五度只有 B-F, 自然的减7和弦应该只有, 7246
  * 减减七和弦  : 减三和弦, 减七度, 又称减七和弦. 

## 3.4. 和弦的转位


如果和弦的根音为演奏的低音, 那么该和弦称为 `原位和弦`, 原位和弦之间的音高低会差距很大, 例如 C 和弦和 B 和弦之间
* 低音 : 指的是演奏的时候最低的音
* 根音 : 指的是构成和弦的根音, 决定了和弦的名称, 在原位和弦中是低音, 否则有可能不是



| 整理表格-名称 | 转音方法                                | 记法                      |
| ------------- | --------------------------------------- | ------------------------- |
| 六和弦        | 351, 三和弦的3音为低音, 根音向上转位    | $?三_6$                   |
| 四六和弦      | 513, 三和弦的5音为低音, 根和3音向上转位 | $?三_{\underset{4}{6}}$   |
| 五六和弦      | 3571, 七和弦的3音为低音                 | $??七_{\underset{5}{6}}$  |
| 三四和弦      | 5713, 七和弦的5音为低音                 | $??七_{\underset{3}{4 }}$ |
| 二和弦        | 7135, 七和弦的5音为低音                 | $??七_2$                  |
和弦的
* 汉字数字代表和弦的种类
* 阿拉伯数字代表的是和弦的转为,  低音到  5音和根音(三和弦)  7音和根音(七和弦) 的音程 


转位: 以和弦的 357音为低音的和弦, 叫做 `转位和弦`  
* 三和弦有三个音, 除去根音之外, 剩下两个音作为低音的情况, `三和弦有两个转位` 
* 第一转位, 又称 六和弦, 以三和弦的三音为低音的和弦, 此时根音向上进行转位
  * 此时三音作为低音到最高音的音程为六度, 因此叫做六和弦 写作$?三_6$
* 第二转位, 又称 四六和弦, 以三和弦的五音为低音的和弦, 根音和三音向上转位
  * 作为低音的五音到根音为四度, 到最高音三音为六度, 叫做四六和弦, 写作 $?三_{\underset{4}{6}}$


* 七和弦, 除去根音之外, 剩下三个音作为低音的情况, `七和弦有三个转位`
  * 第一转位, 根音上转, 三音为低, 称 五六和弦, 低音三音到 七音为五度, 到根音为六度 
  * 第二转位, 根三上转, 五音为低, 称 三四和弦, 低音五音到 七音为三度, 到根音为四度
  * 第三转位, 根三五上, 七音为低, 称 二和弦, 低音七音到 根音为二度


如何识别和构成和弦: 最重要的基石是 熟记各种和弦的音程结构和各个音之间的音高关系   
在从曲谱中识别和弦的时候, 由于转位和弦的存在, `低音 ≠ 根音`
* 假设低音是根音
  * 原位三和弦.  135
  * 原位七和弦. 1357
* 假设低音是三音
  * 三和弦的转位, 六和弦.  351
  * 七和弦的转位, 五六和弦.  3571
* 假设低音是五音
  * 三和弦的转位, 四六和弦. 513
  * 七和弦的转位, 三四和弦, 5713
* 假设低音是七音
  * 二和弦, 7135. 

## 3.5. 等和弦

等和弦也存在, 由于等音替换而产生的等和弦, 有两种
* 不因等音变化而改变和弦的音程结构
* 由于等音变化而改变和弦的音程结构


# 4. 调

## 4.1. 什么是调


调 : 由基本音级所构成的音高位置, 一个正规的调 基本音级之间的音程需要保持与C调一致是固定的
   `1全2全3半4全5全6全7半1`  


* C 调 : 比较特殊, 又称 基本调, 由七个基本音级所构成的调, 它的调号标记是没有升降记号的, 也是反推其他调式的基础

其他的调式也是第一个音来确定调, 而调式的更改不知道为什么总是按照纯五度来连续相生
* 可能是体现在调试的样子上只会增加一个升号标记
* 亦或者是因为 7 和 5 是互质数, 按照5 来加一轮就每个调都有了
* 每产生一个新调, 其调号也按照纯5度关系向上增加一个升号

升号调按照 纯五度相生   4 1 5 2 6 3 7 4 1
* 最终的衍生调式顺序是C  `G D A E B #F #C`,   即 4 1 `5 2 6 3 7 4 1`
* 除了C调以外按照升号添加的位置 `#F #C #G #D #A #E #B`, 即 `4 1 5 2 6 3 7` 4 1
* 在五线谱上声明调式的时候, 升号的书写顺序必须按照上面的顺序书写, 即按照纯五度推导的顺序

升号调总结
* 要判断升号调具体是哪一个调, 可以直接看最后一个升号的是哪一个音, 对应的调式就是下个音的升号调
* 7种派生升号调只有两个有升号, `#F #C`
* G 调 : 5 - 6 - 7 - 1 - 2 - 3 - #4 - 5
  * G 调 之所以常用, 是因为 G 调只需要加一个 #号 
  * `#4`
* D 调 : 2 - 3 - #4 - 5 - 6 - 7 - #1 - 2
  * #4, `#1`
* A 调 : 6 - 7 - #1 - 2 - 3 - #4 - #5 - 6
  * #4, #1, `#5`
* E 调 : 3 - #4 - #5 - 6 - 7 - #1 - #2 - 3
  * #4, #1, #5, `#2`
* B 调 : 7 - #1 - #2 - 3 - #4 - #5 - #6 - 7
  * #4, #1, #5, #2, `#6`
* #F调 : #4 - #5 - #6 - 7 - #1 - #2 - #3 - #4
  * #4, #1, #5, #2, #6, `#3`
* #C调 : 对比C全加一个升号
  * #4, #1, #5, #2, #6, #3, `#7`

降号调, 同理降号调是按照 C 调往下推纯五度来相生的, 最终的顺序同升号调是正好相反的
* 1 `4 7 3 6 2 5 1` 4
* 衍生调的顺序 `F bB bE bA bD bG bC`
* 降号的顺序  `7 3 6 2 5 1 4`

降号调总结
* 同理, 要快速判断是哪一个降号调, 直接看 从右往左的第二个降号是什么音, 就是什么调
* 降号调除了 `F` 其他调式名上都有降号
* F 调 : 4 - 5 - 6 - b7 - 1 - 2 - 3 - 4
  * `b7`
  * 也是只有一个降号, 便于记忆
* bB调 : b7 - 1 - 2 - b3 - 4 - 5 - 6 - b7
  * b7, `b3`
* bE调 : b3 - 4 - 5 - b6 - b7 - 1 - 2 - b3
  * b7, b3, `b6`
* bA调 : b6 - b7 - 1 - b2 - b3 - 4 - 5 - b6
  * b7, b3, b6, `b2`
* bD调 : b2 - b3 - 4 - b5 - b6 - b7 - 1 - b2
  * b7, b3, b6, b2, `b5`
* bG调 : b5 - b6 - b7 - b1 - b2 - b3 - 4 - b5
  * b7, b3, b6, b2, b5, `b1`
* bC调 : 对比 C 全部加一个降号就是了
  * b7, b3, b6, b2, b5, b1, `b4`


那么调式在谱子的一开始可以通过给五线谱添加升降标记来表示, 即通过识别五线谱的最初的升降标记来直接确定乐曲

## 等音调

听起来完全相同的调式, 区别仅仅在于叫法不同

由于大二度两个音, 各自升降即代表了相同的音高
* #C = bD
* #D = bE
* #E = F
* #F = bG
* #G = bA
* #A = bB
* B = bC

反映在调式上 7+7 的 14种 派生调式上, 就有
* `#C` = `bD`
* `#F` = `bG`
* `B` = `bC`

算上 C 调, 也就是说 15 种有名称的调式上 有 3对等音调, 最终代表不同音高的调有 12 个, 这与 12音律的相符合!

## 五度圈

调的关系

就是上面按照 升降 纯五度进行调式推理的 图像化描述

画成一个圆, 有 12 个刻度

以 C 调为起点, 向两侧分别推理 7个 升号调和降号调

最终在 C 的圆圈对侧 有三个重复的, 那三个就是 等音调

升:  
`1 5 2 6 3 7 4` 

降:  
`1 4 7 3 6 2 5` 

这个顺序对于判断调式非常重要



# 调式

文字上的定义: 几个音 (不超过7个, 不少于3个) 按照一定的关系, 构成一个因组织, 并且以某一个音为中心


调性: 调式所具有的特性, 这个词的判断很模糊, 分广义和狭义
* 例如 大调式具有 大调性
* 小调式 具有小调性
* 狭义? : 大调式具有的特性才能叫做大调性
* 广义? : 凡是具有大调式的特点的许多调式, 所拥有的共性, 称为大调性


一个调式中, 音之间的 稳定与否的关系是非常重要的  
* 在 C大调中
  * C E G  起着中心的稳定的作用, 叫做稳定音
  * C 调在稳定音中 最为稳定, 因此 C 就是主音
  * B D F A 则不稳定
  * 为啥?

一首歌曲中, 歌曲最后的结束音, 结尾音, 一般都是 主音, 很少例外

实际上音乐发展中, 同一首歌内部也会进行变调


## 自然大调式

自然大调式: 应该是最常用的 大调, 即 大调的其中一种
* 由 7个音构成
* **其稳定音合起来构成一个 大三和弦**
* 而不稳定音以 二度音程关系倾向于 稳定音
区别大调式的方法
* 就是 7个音中是否存在 主音上 大三度的音
* 说人话就是 音级的关系是 全全半全全全半


调式中的名词: 以 C调距离
* 主音, C
* 属音, 下属音 : 主音的上下纯五度, 即 G, F
* 中音, 下中音 : 主音和属音的正中间的音, 在这里是 E, A. 没有说固定音程?
* 上主因, 导音 : 剩下两个音, 主音的正上和正下, 这里是 D, B
* 这样一个调式中 7 个音都有各自的位置别名


## 小调式 - 三种

小调也有超级多, 在本书中介绍了 3 种小调

小调则是完全独立的, 不从自然大调里派生, 表现为其的音级关系不同

自然小调: 全半全全半全全
* 自然, 从记忆上可以用大调的音调整顺序来快速记忆, 但是从原理上应该要留心区别
* 用大调的音表示为   `A B C D E F G` A

和声小调: G变为 #G, 听起来有点邪恶?
* 全半全全半 增2度 半 : 

旋律小调: F 变为 #F, 听起来和 大调又靠近了
* 全半全全 全全 半
* 旋律小调没有区分上行下行

