
# 1. 算法笔记

# 2. 串

## 2.1. 序列相关

### 2.1.1. 最长不下降子序列

最长不下降子序: 整个子序列单调不降，并且是序列中最长的单调不降子序列

nlogn算法:
* 序列A, 令`F[t]` 表示从1到t的子序列中, 最长不下降子序列的长度
* 初始时设置 `F` 为0
* 单纯使用 `F` 可以得到 $O(n^2)$ 算法
  * 有`F[i]= max(1,F[j]+1) j=1,2,...,i-1 where A[i]>A[j]`

* 通过加入另一个序列 `D`, `D[k]` 表示所有的长度为k的不下降子序列中, 末尾最小的值
* 在此过程中可以保证
  * `D[k]` 的值全程不下降
  * `D` 全局有序
* 利用D可以得到另一种计算最长不下降子序列的方法
  * 当前已经得到的最长子序列长度为 len, 即可以访问到 `D[len]`
  * 对每一个新点 `A[i]` 
    * 若 `A[i]>D[len]`, 则可插入 `D[len+1]=A[i]`
    * 否则逆序遍历 `D`, 访问到第一个 `A[i]>D[t]`, 更新 `D[t+1]=min(D[t+1],A[i]`
  * 单纯遍历 D 的复杂度是 $O(n)$, 总的复杂度是$O(n^2)$
  * 由于D有着全局有序的特性, 因此可以使用二分查找, 将总的复杂度降到 $O(nlogn)$


```cpp
vector<int> LIS(vector<int>& A) {
    int n=obs.size();
    
    vector<int> F,D;

    // 注意D的索引k 代表的是子序列的长度, 但在实际实现中代表的是长度-1
    for(auto &o:obs){
        // 在D中二分查找
        // 如果要求子序列严格递增, 则需要使用 lower_bound
        auto p=upper_bound(D.begin(),D.end(),o);
        // 如果当前是历史最大值
        // 在D的末尾更新当前值
        if(p==D.end()){
            D.push_back(o);
            F.push_back(D.size());
        }
        else{
            // 将 p 转化为个数 
            F.push_back(p-D.begin()+1);
            // 因为 upper_bound 的特性, o一定是小于 *p 的, 因此不需要再比较
            // *p=min(*p,o);
            *p=o;
        }

    }
    
    
    return F;

}
```



## 2.2. 字符串匹配

找出两个字符串的相同字串, 匹配两个字符串


### 2.2.1. KMP

* 每次匹配失败后模式串的移动距离不一定是1
* 计算模式串的的移动距离, 移动距离只与模式串有关

算法:
1. 创建模式串的移动距离数组 `next[]`
2. 移动距离的计算: 
   * 取当前字符前面的字符串
   * 其前缀字符串和后缀字符串的相同字符的最大个数
3. 第1,2个字符串的next固定为0
   * 表示匹配出错的时候都是直接从头匹配没有区别
   * 但是在实现的时候为了方便可以令 `next[0]=-1`, 用来区别一般回溯到0和回溯到头了

实现方法: 若 i 是主指针, j 是计算next的匹配指针
* 若 `t[i]==t[j]` 则 `t[0...j]==t[i-j...j]`, 因此有 `next[i+1]=j+1`
* 若 `t[i]!=t[j]` 则回溯 `j==next[j]`, 进入下一轮循环再次比较 `t[i] t[j]`

一般算法实现
```cpp

string s;
int n=s.length();
// 创建next数组
vector<int> next(n,0);

// 朴素算法
next[0]=-1;
// i是主指针, j设置初值为-1
int i=0,j=-1;

// 另一种方法
int i=1, j=0;

// 获取
while(i<s.length-1)
{
    // 如果是已经回溯到头了
    // 或者匹配成功
    if(j == -1 || s[i] == s[j])
    {
        // 先进行指针右移
        i++;j++;
        // 右移后的 next[i] 可以设置成 j, 暂时新的不管 s[i] 是否等于 s[j]
        next[i] = j;
    }
    // 否则回溯 j=next[j]
    else j = next[j];
}

// 进行匹配, 此时 i 是目标串的指针, j 是模式串的指针
i=0;j=0;
while(i<sr.length&&j<t.length)
{
    // 如果j 是-1代表尽头
    // 或者正常匹配到了
    if(j==-1 || s[i]==t[j])
    {
        i++;
        j++;
    }
    //j回退, i不会退, 相当于模式串往右移动
    else j=next[j];               
}
// 如果结束时 j 等于 t的长度
if(j>=t.length)
    //匹配成功，返回子串的起始位置
    return (i-t.length);         
else
    return (-1);                  //没找到
```


最初的实现, 有很多冗余
```cpp
while(i<n){
   // 主副指针的字符相同
   if(s[i]==s[j]){
       // 先右移
       i++;
       j++;
       // 如果 ij字符不同
       // i位置匹配失败时, 匹配指针移动到 j
       if(s[i]!=s[j]){
            next[i]=j;
            // 同时 j 也向前回溯
            j=next[j];
       }
       else{
            // 如果下一对字符ij仍然相同, 则有i位置匹配失败时, 模式串移动到j的回溯位置
            next[i]=next[j];
            // 进入下一轮循环
       }
   }
   // ij字符不同
   else{
       // j指针已经指向最前了
       if(j==0){
           // next[i]=0 省略了
           i++;
       }
       else{
            // j 再次向前回溯
            // 此处的实现较为冗余
            j=next[j];
       }
   }
}

// 匹配过程
// 此处j 是模式串的指针
i=0;j=0;
while(i<n){
   // 字符相同, 指针后移
   if(sr[i]==s[j]){
       i++;
       j++;
   }
   // 字符不同
   else{
       if(j==0){
           i++;
       }
       else{
           j=next[j];
       }
   }
}
```


## 2.3. 回文串 palindromic

基础算法:
* 使用DP思想
* 一开始以每个单独字符串以及两个字符串为中心查看是否是回文
  * 这里对字符可以进行预处理, 从而统一奇数长度回文和偶数长度回文
  * 在所有字符两边加入无关字符, 例如 `#`
  * 最后求得的回文长度整数除以2
* 逐渐向两边延长
* 复杂度 $O(n^2)$


### 2.3.1. Manacher 算法 马拉车算法
    
* O(n)时间求回文, 需要无关字符插入预处理
* 半径radius与半径数组 
  * 半径数组记录每个位置的字符为回文中心求出的回文半径长度
* 最右回文右边界R
  * 右边界指的是: 这个位置及这个位置之前的回文子串, 所到达的最右边的地方, 是一个历史值
  * 算法开始前 R=-1
  * 遍历`s[0]` 时 R=0
* 最右回文右边界的对城中心C
  * 对于遍历字符串中的每个字符p, p位置对应的最右边界对应的回文子串的中心为 C, 和R匹配
  * p不一定等于 C

算法流程:
* 情况1: 下一个要遍历的位置在最右回文右边界R 的右边 `(p+1>R[p])`
  * p+1的为值为新点, 以p+1为中心向两边扩散寻找回文, 复杂度 $O(n)$
  * 在扩散的同时更新回文半径数组radius, 最右回文右边界R, 对称中心C
* 情况2: 下一个要遍历的位置在R及R的左边 `p+1<=R[p]`
  * p+1为旧点,记为 p1, 分三种情况
  * 令p2 是p1 以C为对称中心的对称点, 
  * cL是以C 为对称中心时候的回文子串左边界
  * pL是以p2为对称中心时候的回文子串左边界
  * 子情况1: `cL<pL`
    * p1的回文半径就是p2的回文半径 `radius[p2]`
  * 子情况2: `cL>pL`
    * p1的回文半径就是p1到R的距离 `R-p1+1`
  * 子情况3: `cL=pL`
    * p1的回文半径仍然需要向外拓张才能知道, 从R之后开始向外拓张, 并更新 R和C
* 复杂度: 因为 R是一直保持单向右移, 而所有 `p+1<R` 的情况都是 O(1)复杂度, 直接得到回文半径, 因此总复杂度为 $O(n)$


```cpp
int manacher(string olds){
    // 字符串预处理
    string s;
    for(char c:olds){
        s.push_back('#');
        s.push_back(c);
    }
    int res=0;
    s.push_back('#');

    // 设置基本信息
    int n=s.length();
    int r=-1;
    int c=0;

    // 半径数组
    vector<int> radius(n);

    for(int i=0;i<n;i++){
        // 新点直接遍历
        if(i>r){
            r=i;
            c=i;
            int j=0;
            while(i-j>=0&&i+j<n&&s[i-j]==s[i+j]){
                r=i+j;
                radius[i]=j;

                j++;
            }
        }
        else{
            // 旧点分情况讨论
            int i2=c+c-i;
            int cl=c-radius[c];
            int pl=i2-radius[i2];

            // 左镜像点
            if(cl<pl){
                radius[i]=radius[i2];
            }
            // 左边界
            else if(cl>pl){
                radius[i]=r-i;
            }
            else{
                // 发现重合, 直接跳过重合部分
                radius[i]=r-i;

                // 继续拓展
                c=i;
                int j=r-i+1;
                while(i-j>=0&&i+j<n&&s[i-j]==s[i+j]){
                    r=i+j;
                    radius[i]=j;

                    j++;
                }
            }

        }
        // 视情况使用半径得出结果, 注意该半径值是经过预处理的, 因此仍需/2
        // res=max(res,radius[i]/2);
        // res+=(radius[i]/2)+(radius[i]&1);
    }
    return res;
}

// 省略写法, 省略预处理的步骤
vector<int> manacher(const string &s) {
    int n = s.size();
    vector<int> radius(n);
    // r 的定义直接写在循环体中
    // 不记录r对应的中心点c, 而是直接记录 r对应的左边界
    for (int i = 0, l = 0, r = -1; i < n; ++i) {
        // 直接进行3重判断
        //                      相当于 c+c-i,左镜像点. 相当于 cl>pl的情况
        //                      直接赋值两者之间较小的一方也是正确的
        int j = (i > r) ? 1 : min(radius[l + r - i], r - i + 1);

        // 注意此处的半径 j 是包含中心点的
        // 方便书写, 直接都加上 while 循环
        while (i >= j && i + j < n && s[i - j] == s[i + j])
            j++;
        // 赋值完成后 j 减1
        radius[i] = j--;

        // 重新赋值 l 和 r 
        if (i + j > r) {
            l = i - j;
            r = i + j;
        }
    }
}

```


# 树

## 键树 查找树 Keyword Tree Search Tree

用空间换时间的一种树
* 可以不再拘泥于二叉树的形式, 根节点的子树>=2
* 结点中存储的不是某一个关键字, 而是构成关键字的单字符
* 利用各个关键字的公共前缀来减少查询时间
* 比哈希树快

### 双链树

仍然保持二叉树的形式, 左右两个子节点

结点:
* symbol : 存单个字符

### Trie树 字典树



# 3. 图

## 3.1. 强连通分量 SCC

Strongly Connected Componted  

在有向图中:  
* 对于两个顶点 i,j, 互相之间有有向路径, 称为强连通
* 对于一个图, 每两个顶点都强连通, 称为强连通图
* 子图是强连通图称为强连通分量

### 3.1.1. Kosaraju 算法

* 应用了原图G和反图GT
* 原理: 转置图(反图, 所有边调转方向) 具有和原图一样的强连通分量
* 因为用了两次深搜所以效率比其他两种慢


1. 对原图进行深搜生成树
   * 这里依据遍历到的时间分配一个完成时间 (即搜索完子节点返回递归的时间)
   * 根据完成时间压栈, 完成时间大的在栈顶
   * 递归返回的时候压栈即可
2. 使用返图
   * 同样深度搜索, 但是每次开始的节点是栈顶元素
   * 凡是能访问到的节点构成一个强连通分量

```cpp
const int MAXN=110;
int n;
bool flag[MAXN];//访问标志数组
int belg[MAXN];//存储强连通分量,其中belg[i]表示顶点i属于第belg[i]个强连通分量
int numb[MAXN];//结束时间标记,其中numb[i]表示离开时间为i的顶点
AdjTableadj[MAXN],radj[MAXN];//邻接表,逆邻接表
//用于第一次深搜,求得numb[1..n]的值
voidVisitOne(int cur,int &sig)
{
    flag[cur]=true;
    for(int i=1;i<=adj[cur][0];++i)
        if(false==flag[adj[cur][i]])
            VisitOne(adj[cur][i],sig);
    // 根据出递归的顺序赋时间戳
    numb[++sig]=cur;
}
//用于第二次深搜,求得belg[1..n]的值
voidVisitTwo(int cur,int &sig)
{
    flag[cur]=true;
    // 能访问到的所有节点赋统一标号
    belg[cur]=sig;
    for(int i=1;i<=radj[cur][0];++i)
        if(false==flag[radj[cur][i]])
            VisitTwo(radj[cur][i],sig);
}
//Kosaraju算法,返回为强连通分量个数
int Kosaraju_StronglyConnectedComponent()
{
    int i,sig;
    //第一次深搜
    memset(flag+1,0,sizeof(bool)*n);
    for(sig=0,i=1;i<=n;++i)
        if(false==flag[i])
            VisitOne(i,sig);
    //第二次深搜
    memset(flag+1,0,sizeof(bool)*n);
    for(sig=0,i=n;i>0;--i)
        // 根据出递归的时间戳顺序访问, 可以保证和原图同样的深度搜索
        if(false==flag[numb[i]])
            VisitTwo(numb[i],++sig);
    return sig;
}
```

### 3.1.2. Tarjan 算法

* 同样是深度优先+栈
* 原理: 任何一个强连通分量, 必定是原图的深度优先搜索树的子树

```cpp

// 两个数组
int low[N],dfn[N];
// 布尔存储是否已入栈
bool instack[N];
// 栈
stack<int>st;
// 图的结构体
struct LIST
{
    int v;
    LIST *next;
};
// 图的存储格式: 链接表
LIST *head[N]={NULL};

// 递归的 tarjan 函数
void tarjan(int v)
{
    // 访问到一个新节点, 先赋值时间戳
    dfn[v]=low[v]=time++;
    // 压栈以及标注已压栈
    st.push(v);
    instack[v]=true;
    // 遍历所有子节点
    for(LIST *p=head[v];p!=NULL;p=p->next)
    { 
        // 如果是新节点   
        if(!dfn[p->v])
        {
            // 进入递归 tarjan
            tarjan(p->v);
            // 比较递归结束时的祖先值
            low[v]=min(low[v],low[p->v]);
        }
        // 如果是旧节点, 说明成环
        else if(instack[p->v])
            // 更新本节点的祖先值
            low[v]=min(low[v],dfn[p->v]);
    }

    // 在子节点递归结束后, 如果当前节点的两值仍然相等, 说明是一个根节点
    if(dfn[v]==low[v])
    {
        // 该强连通分量的所有 两值都不相同, 且在栈中的位置在该节点之上
        cout<<"{ ";
        // 这个循环可以清空栈直到该节点, 并作为连通分量输出
        do
        {
            v=st.top();
            st.pop();
            instack[v]=false;
            cout<<v<<' ';
        }while(dfn[v]!=low[v]);
        cout<<"}"<<endl;
    }
```

### 3.1.3. Gabow 算法

* 本质上是 Tarjan 的变形
* 使用两个栈来辅助求出连通分量的根, 而不是使用两个数组
* 算法更精密, 时间更少(不需要频繁更新数组)

```cpp
const intMAXN=110;
typedef int AdjTable[MAXN];//邻接表类型
int n;
int intm[MAXN];//标记进入顶点时间
int belg[MAXN];//存储强连通分量,其中belg[i]表示顶点i属于第belg[i]个强连通分量
int stk1[MAXN];//辅助堆栈
int stk2[MAXN];//辅助堆栈
AdjTablead j[MAXN];//邻接表
//深搜过程,该算法的主体都在这里
void Visit(int cur,int& sig,int& scc_num)
{
    int i;

    // 记录时序
    int m[cur]     = ++sig;

    // 双栈入栈
    stk1[++stk1[0]] = cur;
    stk2[++stk2[0]] = cur;

    // 循环遍历所有次节点
    for(i=1;i<=adj[cur][0];++i)
    {
        // 当前节点还没被访问
        if(0==intm[adj[cur][i]])
        {
            // 进入递归
            Visit(adj[cur][i],sig,scc_num);
        }
        // 递归结束
        // 访问如果当前节点还未分配分量编号
        else if(0==belg[adj[cur][i]]) 
        {
            // 对栈2出栈
            while(intm[stk2[stk2[0]]]>intm[adj[cur][i]])
                --stk2[0];
        }
    }
    // 结束遍历子节点后
    // 当前节点仍然属于自己
    if (stk2[stk2[0]]==cur)
    {
        // 根出栈, 分量数加1
        --stk2[0];
        ++scc_num;
        // 栈1 出栈
        do{
            belg[stk1[stk1[0]]]=scc_num;
        }while(stk1[stk1[0]--]!=cur);
    }
}
//Gabow算法,求解belg[1..n],且返回强连通分量个数,
int Gabow_StronglyConnectedComponent()
{
  int i,sig,scc_num;
 
  memset(belg+1,0,sizeof(int)*n); 
  memset(intm+1,0,sizeof(int)*n);
  sig=0;scc_num=0;stk1[0]=0;stk2[0]=0;
  for(i=1;i<=n;++i)
  {
    if(0==intm[i])
      Visit(i,sig,scc_num);
  }
 
  return scc_num;
}
```


## 最短路径

求取一个图的最短路径, 有多种分类
* 最短路. 最长路.
* 是否有负值权值

### Bellman-fold




### SPFA

* SPFA(Shortest Path Faster Algorithm)算法是求单源最短路径的一种算法
* 是Bellman-ford的队列优化


### 差分约束系统 System of Difference Constraints

定义一个系统为差分约束系统:  
* n 个变量, m 个约束条件
* 形成了 m个不等式, 形如 $a_i-a_j\le k_m$ , 这里k是常数
* 差分约束系统可以解释为求解一组变量的特殊不等式组
* 一般的目标是求出最大/小的解或者求证是否有解


对于单个约束条件, 可以将其理解成三角不等式   
* 将约束条件转换成图, 即可有 $k_m=w(a_i,a_j)$
* 求解不等式可以理解成求解最短路径  



# 4. 数学

## 4.1. 质数

### 4.1.1. 埃拉托色尼筛选法 the Sieve of Eratosthenes
```cpp
void sieve(int n) {
    vector<int> divisor[n];
    for(int i = 1; i < MN; i++)
        divisor[i] = i;
    for(int i = 2; i < MN; i++)
        if(divisor[i] == i)
            for(int j = 2*i; j < MN; j += i)
                divisor[j] = i;
}
```