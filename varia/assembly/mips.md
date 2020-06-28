# 1. MIPS assembly MIPS指令集

## 1.1. 寄存器
MIPS的32个Int寄存器  
* R0 : 恒0
* R1 : 供编译器使用
* R2-R3 : 函数返回值
* R4-R7 : 函数参数
* R8-R15: 临时空间
* R16-R23:存储集群此 不能通过函数更改
* R24-R25:临时空间
* R26-R27:供系统使用
* R28-R30:分别是 全局  栈  Frame 指针
* R31 : 返回地址
  
## 1.2. 指令格式
拥有三种命令,分别命令的格式,从最高位到最低位,都是32位   


1. R格式(Register format) 命令的操作数只能是寄存器,如add
   `op rd,rs1,rs2`  
   `000000<rs1><rs2><rd>00000<op>`  
   前面6个0, 中间三个寄存器分别是第一二操作数和目标寄存器,后接5个0,最后是6位的操作指令  
   
2. I格式(Immediate format) 命令的第二个操作数是立即数,在命令后加i,如addi
    `op rd,rs,immediate`  
    `<op><rs><rd><immediate>`  
    先是6位的操作命令,然后是第一操作数和目标寄存器,最后是16位的立即数  
3. J格式 (Jump format)
    6 + 26  
    `<op> <target address>`


MISP对于数据的存放,支持两种排序, 数据高位在高地址和数据高位在低地址都支持

## 1.3. 指令系统
### 1.3.1. 数据读取存储
* lw $1,12($3)
* sw $1,-8($5)

### 1.3.2. 无条件转移命令 jump

* j imm     `立即数跳转(label)`
* jr $1     `跳转目标地址在寄存器中`
* jal imm   `函数跳转(会把当前地址存储到$ra)`

### 1.3.3. 条件转移指令

**比较与分支**  
* beq rs,rt,offset  `if R[rs]==R[rt] the PC-relative branch`
* bne rs,rt,offset  `if R[rs]!=R[rt] the PC-relative branch`

**比较并设置零**
* slt $1,$2,$3      `if ($2<$3)  $1=1    else $1=0`
* slti $1,$2,imm    `if ($2<imm) $1=1    else $1=0`


# 2. MIPS 汇编编程
MIPS汇编以`.s`拓展名结尾  

## 2.1. 编译器内容以及main主函数

```
    .text
    .globl main
main:

    jr $ra
```

## 2.2. 数据设定

```
    .data
data1: .word 255
data2: .word 127
data3: .word -1

    .text
    .globl main
main: 
    lw, $t1,data1
```

## 2.3. 3.while 循环

```
loop:   


    bne $t0,$s5,exit
    j loop

exit:
```


### 2.3.1. 读取32位立即数

lui $s0,1234 # 读取一个立即数到寄存器的高16位
addi $s0,5678 # 再加上想要的立即数的低位


# 3. Datapath

Single-Cycle Execution 单命令循环执行  
