# Information Security

为了保证情报的CIA, 建立一个可靠的 ISMS (Information Security Management System)

* Confidentiality   : 機密性, 保证情报的访问安全, 不泄露 
* Integrity         : 完全性, 保证情报的完整性, 不被篡改
* Availability      : 可用性, 保证整个信息系统的稳定性, 各种意外对应等, 任何时候都可正常访问


PDCA Circle: Plan - Do - Check - Act 的循环, 建立一个 ISMS

JIS Q 27001:2006 中的 ISMS 的建立顺序
1. ISMS 的应用范围确定
2. ISMS 的基本方针定义
3. Risk Assessment 风险分析和评估的进行方法
4. Risk 的确定
5. 实施风险评估
6. Rise 对应的实施
7. 管理目的和管理策略的选择
8. 残留 Risk 的确认与承认
9. ISMS 的导入和运行
10. 适用宣言书的编成

## Risk 分析

各种概念

* 信息资产: 通过信息安全想要保护的东西, 例如软件设计书, 顾客信息, 经营信息, 技术文档
* 脆弱性  : 信息安全层面上的弱点, 包括公司建筑物的物理弱点, 程序的Bug等技术缺陷, 组织上的管理缺陷等
* 威胁    : 使得信息资产发生损失的原因. 地震火灾等物理威胁, 数据的偷窃和不乏侵入等

Risk分析: 对信息资产潜在的各种风险进行查出, 并进行评估和对策设计. 将风险进行数值化, 根据预算来对高威胁的风险进行对应.  

风险的对应种类:
* 风险回避  : 对于损失较大, 发生频率较高的风险, 使用策略来杜绝这种风险.
* 风险转移  : 对于损失较大, 发生频率较低的风险, 将对应的风险转移到其他公司, 例如购买数据保险, 或者干脆将对应风险的业务外包给其他公司.
* 风险最优化: 损失较小, 发生频率高. 尽可能减少风险带来的损失, 或者通过ISMS来减少风险的概率
* 风险保留  : 损失较小, 发生频率也低. 不进行任何对应, 控制系统的成本. 

# 加密技术

目前的技术主要分为 对称加密 和 非对称加密, 除此之外还有 One-way hash

* 通用密钥方式: 加密和解密使用相同的密钥, 密钥为送收双方都知道, 对称加密
  * DES
  * AES
  * 加密和解密速度快
  * 需要事先将密钥在双方间共享
  * 每一对通信双方, 都需要一个对应的密钥, 密钥的数量可能会巨大
* 非对称加密, 公私密钥方式:
  * RSA
  * 加密和解密使用不同的密钥
  * 算法比较复杂, 时间消费长
  * 密钥管理容易, 可以和任意对象进行通信
  * 使用公钥加密, 只有私钥可以解密, 一般用来做信息传送
  * 使用私钥加密, 任何公钥都可以解密, 一般用来做身份认证 


# 认证技术

One-way Hash 在身份认证和消息确保中被应用的很广, 代表的是 MD5

基于 Hash 的信息内容确保流程:
1. 送信方, 计算信息的 Hash值, 和信息原文一起发送
2. 收信方, 计算收到的信息的 Hash值
3. 收信方对两个 Hash 值进行比较
4. 如果一直的话, 证明 信息原文没有被更改

基于非对称加密的 身份认证 和消息确保
* 发送方:
  * 计算消息的 哈希值, 
  * 哈希值用 私钥加密, 发送加密后的哈希值和信息
* 接收方:
  * 使用发送方的公钥解密 哈希值
  * 计算消息的 哈希值, 并比较
* 如果哈希值一致的话, 可以证明:
  * 发送方的本人身份
  * 消息没有被更改
  
# 网络攻击

基于网络的不正当访问威胁可以包括以下:
* 服务器内的信息泄露
* Web内容被篡改
* Dos 攻击导致 服务器工作停止
* 服务器被当作跳板攻击其他网站
* 服务器被当作骚扰邮件的中继

不正当访问的对策:
* 关闭服务器的不必要端口 Port
* 设置防火墙, 只允许设定好的特定通信内容 (package filtering)
* package filtering , 识别数据包中的 端口, 对通信进行限制



计算机病毒的定义:
* 传染能力: 程序内能够自发的拷贝程序本体, 并发送给其他的系统
* 潜伏能力: 能够控制程序的启动时间, 次数, 使得在发动之前不被察觉
* 发病能力: 破坏文件, 篡改数据等设计者的非法意图

计算机病毒的对策:
* 禁止未被许可的软件安装
* 及时更新新版的系统软件和相关服务软件
* 导入防病毒软件, 并经常更新病毒特征库