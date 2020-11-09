# argparse  

argparse组件可以很方便的写一个命令行界面，可以很容易的定义程序的参数，并从`sys.argv`中提取出来，同时还会自动提供错误信息。  

argparse可以输出错误信息，在识别格式问题后在执行计算之前退出程序  
**argparse的使用**
```python
# 定义一个命令行组件
# 在定义的时候可以指定程序的主要目标，在-h中会显示
import argparse
parser = argparse.ArgumentParser(description="calculate X to the power of Y")


# 加入一个必须(位置)参数，并定义help内容
parser.add_argument("echo", help="echo the string you use herep")
# 默认会把参数视作字符串，可以指定数据的格式
parser.add_argument("square", help="display a square of a given number",type=int)


# 加入一个可选参数，接受后面的一个整数为参数 --verbosity 1
# 可选参数可以指定长短命令版本
parser.add_argument("-v","--verbosity", help="increase output verbosity")
# 加入一个可选标志参数，输入该参数则代表执行动作，不能再后跟参数
# action="store_true"
parser.add_argument("-V","--verbose", help="increase output verbosity", action="store_true")
# 给参数限制可以接受的值  choices=[0, 1, 2] 
parser.add_argument("-v", "--verbosity", type=int, choices=[0, 1, 2], help="increase output verbosity")
# 用另一种方法增加信息冗余 action="count" ,统计某一参数出现了几次
# 可以识别 -v -vv -vvv '--verbosity --verbosity'
# 指定 default=0 作为参数不出现时候的默认值，否则变量会是None不能用来和整数比较 
parser.add_argument("-v", "--verbosity", action="count", default=0, help="increase output verbosity")



# 可以创建一个矛盾参数组，当中的参数不能同时出现
group = parser.add_mutually_exclusive_group()
# 在group的方法中添加参数
group.add_argument("-v", "--verbose", action="store_true")
group.add_argument("-q", "--quiet", action="store_true")


# 导出参数
args=parser.parse_args()
print(args.echo)
print(args.square)
```  
