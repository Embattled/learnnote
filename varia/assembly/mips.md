# MIPS assembly MIPS指令集

## 1. 寄存器
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
  
拥有两种命令,分别命令的格式,从最高位到最低位    
1. 命令的操作数只能是寄存器,如add
   `op rd,rs1,rs2`  
   `000000<rs1><rs2><rd>00000<op>`  
   前面6个0, 中间三个寄存器分别是第一二操作数和目标寄存器,后接5个0,最后是6位的操作指令  
   
2. 命令的第二个操作数是立即数,在命令后加i,如addi
    `op rs rd immediate`  
    `<op><rs><rd><immediate>`  
    先是6位的操作命令,然后是第一操作数和目标寄存器,最后是16位的立即数  



MISP对于数据的存放,支持两种排序, 数据高位在高地址和数据高位在低地址都支持

## jump 命令

* j     立即数跳转(label)
* jal   函数跳转(会把当前地址存储到$ra)
* jr    跳转目标地址在寄存器中
* beq   条件跳转(branch on equal)
* bne   条件跳转(branch on not equal)

# MIPS 汇编编程
MIPS汇编以`.s`拓展名结尾  

## 1. 编译器内容以及main主函数

```
    .text
    .globl main
main:

    jr $ra
```

## 2. 数据设定

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

## 3.while 循环

```
loop:   


    bne $t0,$s5,exit
    j loop

exit:
```