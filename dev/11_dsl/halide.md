# 1. Halide introduce

一种专门用于图像处理的 Domain Specific Language (DSL)

官方 API 文档 : https://halide-lang.org/docs/


主要的概念:
* Var
* Expr
* Func
* Buffer
对于 Halide in C++, halide 的相关类都是放在命名空间 `Halide` 里的

```cpp
Halide::Func gradient;
Halide::Var x, y;
Halide::Expr e = x + y;

gradient(x, y) = e;
Halide::Buffer<int32_t> output = gradient.realize({800, 600});

```

## 1.1. Var

Var 对象则会被当作Func定义里的变量, 即他们本身并没有任何意义, 仅仅只是作为一个 name, 被用于 definition of a Func

由于编译的过程中, 会丢失代码中的变量名, 所以在 Halide 中定义各种步骤的时候可以加上字符串形式的说明符, 用于在 debug 的过程中实现内容显示  

```cpp
Halide::Var x, y;

Halide::Var x("x"), y("y");

```
一般来说, 会定义名称为 x,y 的两个 Var 用于表示二位图像的两个坐标轴, 如果顺序是 x,y, 则:
* x 代表 column index
* y 代表 row index
* 从 numpy ndarray 的角度去关联, 即定义越靠前的变量, 维度越靠后, 顺序遍历的速度越快

## 1.2. Expr

Expr 可以理解为表达式对象, 一个 Expr 的定义中所使用的 Var, 代表了 Var 所可以取的所有值的运算的集合.  通过 Expr 可以简化一个 Halide pipeline 的定义, 完全不使用 Expr 也可以定义简单 Halide 函数.  

Expr 可以理解为非坐标的变量集合, 可以对 Expr 对象进行整体的运算. 通过对 Expr 对象进行操作即可逐步的定义完整的 Halide Pipeline

```cpp

// For each pixel of the input image.
// value 代表的 Expr 直接代表整个图像
Halide::Expr value = input(x, y, c);

// 不适用像素值而是各个像素的坐标来计算 Expr
Halide::Expr e = x + y;

// 对 e 进行整体的操作
// cast 类型转换
value = Halide::cast<float>(e);

// 倍数
value = value * 1.5f;

// Halide 库中的一些基础运算函数
value = Halide::min(value, 255.0f);
```

## 1.3. Func

一个 Func 对象代表一个 pipeline stage, 代表了某个 stage 下一张图片的每个像素所拥有的值, 可以理解为是一整张 computed image

通过C++中的 Var, Expr, Func 等内容定义了 Halide 函数的模板, 需要将一个 Halide 函数 Func 实例化才能够真正运行计算过程.

```cpp
// 定义一个名为 gradient 的函数
Halide::Func gradient;

Halide::Func gradient("gradient");

// 定义了一个计算图像, 他的每个坐标的像素值是坐标的和
gradient(x,y) = x + y;

// 使用 Expr 来简化 Halide 函数的定义
Halide::Expr e = x + y;

/* 
    对 e 进行各种运算操作
 */
// Halide 函数的运算结果直接用基于 x,y 定义的 Expr 来表示, 这里的 Var 的数目应该需要匹配
gradient(x,y)=e

// 实例化一个 Func 
gradient.realize({800, 600});
```

## 1.4. Buffer

Halide 中通过 Buffer 来定义一个内存空间用于存储计算结果数据, 同时也代表了告诉 Halide Func 要处理的图像数据的解析度
```cpp
// Buffer 本身是一个容器, 可以直接用来存储输入数据
// 这里的图像载入应该是 Halide 对应的特殊的读取函数
Halide::Buffer<uint8_t> input = load_image("images/rgb.png");

// grident.realize({800,600}) 为函数实例化, 同时代表处理的数据对象是 800 x 600, 同时执行了处理
// Halide Buffer 为一个缓冲区定义, 用于接纳所有的处理结果
Halide::Buffer<int32_t> output = gradient.realize({800, 600});

for (int j = 0; j < output.height(); j++) {
    for (int i = 0; i < output.width(); i++) {
        // 查看 output 图像值直接通过括弧坐标即可访问
        output(i,j);
    }
}
```

# 2. Compile

Halide 算法的编译, 通过 g++即可, 但是要使用 c++17标准, 同时 Halide 的库本身是需要用到 LLVM, 即 clang 来编译的

`g++ *.cpp -g -I <path/to/Halide.h> -L <path/to/libHalide.so> -lHalide -lpthread -ldl -o 输出 -std=c++17 LD_LIBRARY_PATH=<path/to/libHalide.so> ./lesson_03`


# 3. Debug

Halide 本身因为是独立的编译系统, 有自己的 Debug 步骤和方法


通过设置对应的环境变量来控制 Halide pipeline 的 dump 过程:
* `HL_DEBUG_CODEGEN=1`  : print out the various stages of compilation, and a pseudocode representation of the final pipeline.
* `HL_DEBUG_CODEGEN=2`  : shows the Halide code at each stage of compilation, and also the llvm bitcode we generate at the end.




## 3.1. print

不同于 C 的 printf, print 函数是 Halide 空间下的打印函数

比起直接打印具体的值, Halide 的print 则是在不影响对象的具体行为的前提下, 附加一层输出的效果

```cpp
Func func_4_3{"Function_lession_4_3"};
func_4_3(x,y)=sin(x)+print(cos(y));
func_4_3.realize({4,4});
printf("\nEvaluating sin(x) + cos(y), and just printing cos(y)\n");
func_4_3.realize({4, 4});

// 同 trace_store 不同, 这里仅仅只是单纯打印计算结果, 而没有任何说明文字, 也可以像 printf 一样添加各种其他输出内容, 该函数的使用有点类似于 python 的 print
Func func_4_4{"Function_lession_4_4"};
// 同样不会影响到计算过程
func_4_4(x, y) = sin(x) + print(cos(y), "<- this is cos(", y, ") when x =", x);
func_4_4.realize({4,4});

```

print_when() 功能则是更加细化成条件输出, 即可以设置成只有出现非法值得时候在进行打印
```cpp
        Func g;
        e = cos(y);
        e = print_when(e < 0, e, "cos(y) < 0 at y ==", y);
        g(x, y) = sin(x) + e;
        printf("\nEvaluating sin(x) + cos(y), and printing whenever cos(y) < 0\n");
        g.realize({4, 4});


```

## 3.2. cout

Halide 编译甚至可以直接使用 c++ 的输出流来打印, 具体输出的则是 Expr 的最终表达式, 在构建非常复杂的算法的时候很有用, 会在编译 Halide Filter的时候进行输出

```cpp
        Var fizz("fizz"), buzz("buzz");
        Expr e = 1;
        for (int i = 2; i < 100; i++) {
            if (i % 3 == 0 && i % 5 == 0) {
                e += fizz * buzz;
            } else if (i % 3 == 0) {
                e += fizz;
            } else if (i % 5 == 0) {
                e += buzz;
            } else {
                e += i;
            }
        }
        std::cout << "Printing a complex Expr: " << e << "\n";

```

## 3.3. print_loop_nest

输出 Halide 进行优化后的 pseudocode , 显示所编译出来的 for 循环构成

```cpp
Func gradient("gradient");
gradient(x, y) = x + y;
gradient.trace_stores();

Buffer<int> output = gradient.realize({4, 4});
gradient.print_loop_nest();
```

# 4. Buffer

# 5. Halide Pipeline

Halide Pipeline 主要部件的详细介绍

## 5.1. Expr

Expr 可以理解为表达式对象, 是一种轻量化的数据结构, 表达了一个 scalar expression.
* 一个 Expr 的定义中所使用的 Var, 代表了 Var 所可以取的所有值的运算的集合.
* Expr 可以用于实现  计算, 存储常量和变量
* 通过 Expr 的结合可以实现复杂的表达式, 可以简化一个 Halide pipeline 的定义, 完全不使用 Expr 也可以定义简单 Halide 函数.  

然而, Expr 并不是一个用于存储pipeline的中间值 (intermediate values) 的数据结构, 这是因为:
* Expr 可能需要非常重的计算
* Expr 可能不符合 Halide complier 的优化策略

事实上, 很多 Halide 头文件下的 Halide 实用函数都是以 Expr 作为输入值和返回值的. 

### 5.1.1. Expr routine


#### 5.1.1.1. 逻辑选择

根据一定的逻辑选择分支值
* select    : 类似于 numpy.where 或者 C 语言中的三元表达式, 根据条件从两个输入中选择一个到输出中,


select 详解
```cpp
// 基本用法, Halide select 的特点是它不会根据 condition 来计算 true_value 或 flase_value, 而是总是计算好 true_value 和 false_value 再进行赋值, 这点要注意
// Typically vectorizes cleanly, but benefits from SSE41 or newer on x86. 
Expr Halide::select 	(
        Expr  	condition,
		Expr  	true_value,
		Expr  	false_value 
	) 		

// 高级用法, 接受多个 condition , 类似于 C 语言中的 switch
// 如果 c0 为真, 则返回 v0, 否则判断 c1..., 直到最后的 cn 为真则返回 vn, 否则返回 vn+1
template<typename... Args, typename std::enable_if< Halide::Internal::all_are_convertible< Expr, Args... >::value >::type * = nullptr>
Expr Halide::select 	( 	
        Expr  	c0,
		Expr  	v0,
		Expr  	c1,
		Expr  	v1,
		Args &&...  	args 
	) 		

```


## 5.2. Func

This class represents `one stage` in a Halide pipeline, and is the unit by which we schedule things.  

一个 Func 对象代表一个 pipeline stage, is a conceptually a mapping from array locations to values.
* By default they are aggressively inlined, so you are encouraged to make lots of little functions, rather than storing things in Exprs.   
* 官方推荐使用 Func 来存储 intermediate results, 这是因为 Func 是 Halide 最主要的用于表达计算的数据结构
* 同时 Func代表了某个 stage 下一张图片的每个像素所拥有的值, 可以理解为是一整张 computed image.
* Func 可以理解为一个 输入像素坐标, 然后输出一个 single output value 的函数


Func 在 pipeline 定义中的种类:
* pure definition       : `f(x,y) = x,y`
  * The declaration of a function that specifies the computation to be performed, without any reference to the memory layout or allocation of the data being processed.
  * 单纯用于表示计算过程的 function, 不涉及任何内存layout 或者数据分配
  * 这个概念和 function definition 形成对比, 而后者是包括了 数据的存储和访问形式的.
  * 对于一个 `Func`, 它的第一个 `definition` 必须是一个 `pure definition`, like a mapping from Vars to an Expr.

* function definition   : 是基于 pure definition 的延申. 简而言之 definition can include computed expressions on both sides
* 根据具体的操作可以大概进行分类.
  * update definition   : define a function that updates a `previously defined function with new values`. 体现为把之前的 Func 对象的值进行更新.
  * reduction definition: is an update definition that recursively refers back to the function's current value at the same site. 根据Func当前的值进行更新的表达式.  
  * extern definition   : specify a Halide function that is implemented in some other language or library, such as C or CUDA. 
  * extern C definition : similar to an extern definition, but is used specifically for functions that are implemented in C or C++.
* Halide 文档中关于 Reduction 的解释:
* An reduction is a function with a two-part definition.
  * It has an initial value, which looks much like a pure function.
  * an update definition, which may refer to some RDom. 


Func Update definition 的写法 rule:
* Each Var used in an update definition must appear unadorned in the same position as in the pure definition in all references to the function on the left and right-hand side. 
* 如果某一个 Var 要被用于 Update definition (即可以选择用或者不用, 如果用的话就必须), 起码不被修饰的用在和 Pure Definition 相同的位置上
  * `f(x,y) = x+y;`   : Pure definition
  * `f(x, 3) = f(x, 0) * f(x, 10);` : 只使用 x, 而 x 的位置和 Pure definition 相同, 且没有被修饰过
  * `f(0, y) = y * 8;` : 只使用 y, 式子右边没有 f 因此这不是 reduction definition.
  * `f(x, x + 1) = x + 8;` : x 的某个出现位置和样子与 pure definition 中的一致, 因此在第二个参数上的 `x+1` 也合法
  * `f(y / 2, y) = f(0, y) * 17;` : 同理合法
* 不符合上条规则的写法:
  * `f(x, 0) = f(x + 1, 0);` : 等式右边出现 f 的地方, 其 Var 必须也是 Pure 的
  * `f(y, y + 1) = y + 8;`   : 等式左边的 f 在定义的时候是 `f(x,y)`, 如果使用 y 的话, 第二个参数必须是 `y`, 不能是修饰后的 `y+1`
  * `f(y, x) = y - x;`       : 同理, 如果使用了某个 Var, 则在 f 里的位置必须与 Pure definition 相同
* Free variables can't appear on the right-hand-side only
* Var 不能只出现在等式右边, 例如 `f(3, 4) = x + y;`



### 5.2.1. Func scheduling

对于定义好的 Func, 可以通过调用对应的句柄来让 Halide 进行并行优化, 这是一个比较有技术含量的工作, 一般的情况下交给 Halide 来自动优化就能取得比较好的效果  

各种手动 scheduling 接口 `Func.*`:
* `reorder(y,x)`    : 手动指定 loop 的执行顺序, 顺序为从左到右是最内loop到外
* `parallel(y)`     : 开启在某个特定坐标轴上的并行执行
* `split(x,x_outer,x_inner,factor)` : 将某个坐标轴的 loop 拆分成两个, 会影响最终生成的 psedocode
  * 具体的循环内的操作则是 `x_execute = x_outer * factor + x_inner;` 不会影响实际执行的顺序
  * 主要配合 verctorize 来使用, 将内循环直接向量化用于加速
* `fuse(x,y,fused_x_y)`             : 和 split 相对比, 把某两个坐标轴整合成一个, 会减少最终的 loop 数量 
  * 具体的循环内则是 `int y = fused / 根据输入自动推算的width; int x = fused % 根据输入自动推算的width;`  不会影响结果
* `tile(x,y,x_outer,y_outer,x_inner,y_inner,4,4);` 相当于 split 和 reorder 的结合, 相当于以下的代码:
  * `Func.split(x, x_outer, x_inner, 4);`
  * `Func.split(y, y_outer, y_inner, 4);`
  * `Func.reorder(x_inner, y_inner, x_outer, y_outer);`
  * 具体的参数意思还是要参照文档
  * 通常情况下进行最优化并行不会显式调用 reorder 和 split , 因为他们是最为原始的设置, 通常情况下只使用 tile 就足够了
* `vectorize(x);`       : 在计算的时候将某一个 loop 直接向量化加速
  * 通常和 split 来结合使用, 即 split_factor 即为硬件所支持的最大向量长度
  * `Func.split(x, x_outer, x_inner, 4);` 
  * `Func.vectorize(x_inner);`
* `unrolling(x_inner);` : 把某一个轴的循环展开, 即从代码上取消 for 循环, 转而使用 `x=0,x=1,...,x=end` 的形式生成代码
  * `Func.split(x, x_outer, x_inner, 2);`
  * `Func.unroll(x_inner);`
  * 通过使用第二个参数来实现的 The shorthand for this is: `Func.unroll(x, 2);`


一个 tutorious 中的整合了所有 scheduling 方法的高速化计算 example:

```cpp
Func gradient_fast("gradient_fast");
gradient_fast(x, y) = x + y;

// 在整个图像上进行 block 化, 同时对于 横纵的block index 进行 fuse 并行
Var x_outer, y_outer, x_inner, y_inner, tile_index;
gradient_fast
    .tile(x, y, x_outer, y_outer, x_inner, y_inner, 64, 64)
    .fuse(x_outer, y_outer, tile_index)
    .parallel(tile_index);

// 在小的 block 中再次 tile, 水平方向上对 x 进行向量化计算
// 纵方向上因为 y_paris 的 factor 只有2, 因此直接展开 loop 
Var x_inner_outer, y_inner_outer, x_vectors, y_pairs;
gradient_fast
    .tile(x_inner, y_inner, x_inner_outer, y_inner_outer, x_vectors, y_pairs, 4, 2)
    .vectorize(x_vectors)
    .unroll(y_pairs);
```

#### 5.2.1.1. Func.tile

tile() 是最常用的接口, 是 reorder 和 split 的整合, 完整的函数重载定义如下  

```cpp
/* 
Split two dimensions at once by the given factors, and then reorder the resulting dimensions to be xi, yi, xo, yo from innermost outwards.
This gives a tiled traversal. 
实现的是 reorder 并且 split, reorder 的顺序是固定的, 从内到外依次是 xi, yi, xo, yo 原来如此
*/

Func & Halide::Func::tile 	( 	
        const VarOrRVar &  	x,
		const VarOrRVar &  	y,
		const VarOrRVar &  	xo,
		const VarOrRVar &  	yo,
		const VarOrRVar &  	xi,
		const VarOrRVar &  	yi,
		const Expr &  	xfactor,
		const Expr &  	yfactor,
		TailStrategy  	tail = TailStrategy::Auto 
	) 		

/* 
A shorter form of tile, which reuses the old variable names as the new outer dimensions. 
省略了 new outer dimension, 直接将旧的 Var 作为 Outer dimension, 感觉会容易搞混
*/
Func & Halide::Func::tile 	( 	
        const VarOrRVar &  	x,
		const VarOrRVar &  	y,
		const VarOrRVar &  	xi,
		const VarOrRVar &  	yi,
		const Expr &  	xfactor,
		const Expr &  	yfactor,
		TailStrategy  	tail = TailStrategy::Auto 
	) 		

/* 
A more general form of tile, which defines tiles of any dimensionality.
高维度通用 tile 重载, 可以对任意维度的数据进行 tile
*/
Func & Halide::Func::tile 	( 	
        const std::vector< VarOrRVar > &  	previous,
		const std::vector< VarOrRVar > &  	outers,
		const std::vector< VarOrRVar > &  	inners,
		const std::vector< Expr > &  	factors,
		const std::vector< TailStrategy > &  	tails 
	) 	
/* 
The generalized tile, with a single tail strategy to apply to all vars. 
高维度通用 tile 重载, 只使用一个 TailStrategy 应用在所有 Var 上
*/
Func & Halide::Func::tile 	( 	
        const std::vector< VarOrRVar > &  	previous,
		const std::vector< VarOrRVar > &  	outers,
		const std::vector< VarOrRVar > &  	inners,
		const std::vector< Expr > &  	factors,
		TailStrategy  	tail = TailStrategy::Auto 
	) 	

/* 
Generalized tiling, reusing the previous names as the outer names. 
同理, 高维度的 tile 重载, 同时不重新定义 outer names
*/
Func & Halide::Func::tile 	( 	
        const std::vector< VarOrRVar > &  	previous,
		const std::vector< VarOrRVar > &  	inners,
		const std::vector< Expr > &  	factors,
		TailStrategy  	tail = TailStrategy::Auto 
	) 		
```

##### 5.2.1.1.1. TailStrategy  Halide::TailStrategy

用于在 tail 的时候, 定义如何处理尾部. 即 维度的宽并不能被 factor 整除的情况下, 如何对应. 因为有的时候可以进行重复计算, 而有些时候不可以.   

感觉很有用, 先定义好章节以后再看 TODO

### 5.2.2. Func realize

和 Halide 函数的实例化相关   


### 5.2.3. Func Debug

通过一系列的 Func 对象方法, 可以实现对多个环节的 dump 以及打印, 从而实现多种量级的 debug


#### 5.2.3.1. trace

`Func.trace_*`


``` cpp
// 跟踪所有 evaluations , 会打印 the value of a Func
// 会打印 Func 在具体 realize 的时候的所有计算结果
Func func_4_1{"Function_lession_4_1"};
func_4_1(x,y)= x+y;
func_4_1.trace_stores();
output=func_4_1.realize({8,8});

// 具体的打印结果为
/*
Begin pipeline Function_lession_4_1.0()
Tag Function_lession_4_1.0() tag = "func_type_and_dim: 1 0 32 1 2 0 8 0 8"
Store Function_lession_4_1.0(0, 0) = 0
Store Function_lession_4_1.0(1, 0) = 1
...
Store Function_lession_4_1.0(7, 7) = 14
End pipeline Function_lession_4_1.0()
*/
```
#### 5.2.3.2. HTML 输出底层编译结果代码
通过将输出以 HTML 的形式表示, 方便查看和理解 Halide 的最佳化结果
```cpp
Func gradient("gradient");
Var x("x"), y("y");
gradient(x, y) = x + y;

// 带有语法高亮的 HTML 的 Halide 编译结果
gradient.compile_to_lowered_stmt("gradient.html", {}, HTML);
```

#### 5.2.3.3. print_loop_nest();

通过调用 `Func.print_loop_nest()` , 可以在 Halide Func 运行的时候打印其优化后的 loop 结构, 从而方便判断优化结果是否符合内存 cache 的顺序  

```cpp
Func func_5_1{"Function_lession_5_1"};
func_5_1(x,y) = x+y;
func_5_1.print_loop_nest();
func_5_1.trace_stores();
// func_5_1.realize({4,4});  打印优化结果并不需要具体实例化一个 Func 

/*
produce Function_lession_5_1:
  for y:
    for x:
      Function_lession_5_1(...) = ...
*/
```

## 5.3. Buffer

Buffer 在一定程度上也是一个虚拟的 Buffer, 在定义的时候需要指定  大小或者维度  


### 5.3.1. set_min

是 buffer 的一个非常有意思的方法, 它可以指定该 buffer 的起点坐标, 即作为高维 array 它的起点index 可以不是 0 , 这对于只处理图像的某些部分来说非常有用

```cpp
Func func;
// ... 定义func
// buffer 本身只有 4*4 的大小, 类似于只想在 4*4 的图像上应用处理, 或者只想在图像的某个 4*4 大小上应用
Buffer<int> buffer(4, 4);
// 通过定义 set_min 来指定该 buffer 的起始 index
buffer.set_min(100,5);
// 在实例化 func 的时候, x(维度1) 从 100 开始,  y(维度2) 从 5 开始
func.realize(buffer);
```

# 6. Generator

是一种用于提前将 Halide pipeline 进行编译的方法
* 比起将 pipeline 定义在 main() 中, 这种方法将 pipeline 实现为函数, 更加贴合实际使用
* 具体的使用需要将所有 pipeline 定义在一个从 `Halide::Generator` 继承的自定义类中

```cpp
// 公有继承 Halide::Generator
class MyFirstGenerator : public Halide::Generator<MyFirstGenerator> {

// 默认会将需要的元素定义为公有成员
public:
    // 定义该 Halide pipeline 的输入
    // 会自动的作为参数出现在 generated function
    // 具体顺序符合 在这里声明的顺序
    Input<uint8_t> offset{"offset"};
    Input<Buffer<uint8_t, 2>> input{"input"};

    // 同理, 也事先定义好 pipeline 的输出部分
    Output<Buffer<uint8_t, 2>> brighter{"brighter"};

    // 通常 var 成员也是定义在 public 中
    Var x, y;

    // 定义一个类似于调用 realize 的实现 pipeline 的函数
    void generate(){
        brighter(x, y) = input(x, y) + offset;
        // Schedule it.
        brighter.vectorize(x, 16).parallel(y);        
    }
};
```

# 7. RDom 

一个快速定义 reduction 的类

* 运行一个 reduction function 首先根据所定义的 domain 来初始化数据, 然后根据所定义的 update 规则来更新每一个像素值
* `Halide::RDom` 的作用就是快速来定义一个 reduction
  * 在 Halide 函数即便没有实例化的时候, 也可以快速的指定对一个范围进行的运算, 例如
    * `f(x,1)=f(x,1)+1` , `f(x,2)=f(x,2)+1`, ..., `f(x,50)=f(x,50)+1`.
    * 要固定的执行前 50 列的修改, 需要定义 50 行 update 很麻烦, 这时候可以直接使用 RDom 来进行有范围的 Update
    * `RDom r(0,50);`  `f(x,r)=f(x,r)+1`;  


* reduction 的妙用:
  * 用于去定义一个递归函数, pure halide function 不支持递归函数
  * 用于执行 scattering operations, left-hand-side of an update definition may contain general expressions

## 7.1. example


### 7.1.1. RDom 的 reduction function

使用 RDom 的 reduction function 的定义
```cpp
Func f;
Var x;

// creates a single-dimensional buffer of size 10
RDom r(0, 10);

// the initial value, 一个 reduction function 的初始化
f(x) = x; 
// 使用 RDom 来定义 update 规则
f(r) = f(r) * 2;
Buffer<int> result = f.realize({10});
```

### 7.1.2. recursive function

使用 RDom 可以实现 Halide Pure function 无法实现的递归函数的定义, 例如 斐波那契数列

```cpp
Func f;
Var x;

// 定义一个能用于计算前20 位斐波那契数的 Buffer
RDom r(2, 18);

// 对 function 进行初始化
f(x) = 1;
// 建立更新规则
f(r) = f(r-1) + f(r-2);
```

### 7.1.3. scattering operation

使用 Halide 来实现统计整张图片上的像素值直方图  
```cpp
ImageParam input(UInt(8), 2);
Func histogram;
Var x;
RDom r(input); // Iterate over all pixels in the input

// 初始化
histogram(x) = 0;

// 建立 Update rule, 用于统计像素值的直方图
histogram(input(r.x, r.y)) = histogram(input(r.x, r.y)) + 1;
```


使用 Halide 来计算积分图像
```cpp
// 输入是2维 float
ImageParam input(Float(32), 2);

// 顺序计算水平
Func sum_x, sum_y;
Var x, y;

// 整张图片进行遍历
RDom r(input);

// 初始值
sum_x(x, y)     = input(x, y);

// 注意, 使用 r.x, r.y 来指代 Var 的时候Halide 不会进行并行优化, 采用这种写法因为需要满足 x 的从左到右顺序执行
sum_x(r.x, r.y) = sum_x(r.x, r.y) + sum_x(r.x-1, r.y);

// 同理进行初始化
sum_y(x, y)     = sum_x(x, y);

// 顺序更新 y 轴方向上的积分值
sum_y(r.x, r.y) = sum_y(r.x, r.y) + sum_y(r.x, r.y-1);
```

然而, 计算积分图像的时候, 假设先进行 水平计算 sum_x, 再进行 sum_y, 那么 sum_x 的计算中是 y 轴可并行的, 同理 sum_y 中 x 轴是可并行的, 那么, 需要并行的轴不使用 RDom.x RDom.y 即可

```cpp
ImageParam input(Float(32), 2);
Func sum_x, sum_y;
Var x, y;
// 整张图片上遍历
RDom r(input);
// 初始值
sum_x(x, y)   = input(x, y);
// y 使用 Var y 而不是 r.y, 使得 Halide 会对  y 轴进行并行处理
sum_x(r.x, y) = sum_x(r.x, y) + sum_x(r.x-1, y);

// 同理
sum_y(x, y)   = sum_x(x, y);
// 根据 x 轴的积分结果来进行全局积分图像的时候可以进行纵向并行
sum_y(x, r.y) = sum_y(x, r.y) + sum_y(x, r.y-1);

// 主动的调用 Func 的 Parallel 接口来开启并行优化
// 注意, 直接使用 Func.parallel() 只会实现初始化 initialization step 的并行
sum_x.parallel(y);
// 通过调用 Func.update() 来得到 update step handle 来开启对更新过程的并行
sum_x.update().parallel(y);

// 更加细化的, 由于计算 sum_y 时候的缓存非连续性, 可能导致优化不充分, 对此可以手动的调用
sum_x.compute_at(sum_y, y);
// 这将会导致, sum_x 不会按顺序被执行, 而是 computed only as necessary for each scanline of the sum in y
// 只有在 sum_y 的计算需要对应的 sum_x 值的时候, 执行 sum_x 对应的部分
```


# 8. Halide::functions

定义在 Halide 命名空间里的各种实用函数接口, 可以用于快速实现一些基本的图像操作  


## 8.1. sum

