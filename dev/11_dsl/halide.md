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

## 5.1. Var

A Halide variable, to be used when defining functions.

仅仅只是作为一个 Index 符号, 在不同的表达式里可以有不同的意思:  
* It is just a name, and can be reused in places where no name conflict will occur.
* It can be used in the `left-hand-side` of a function definition
* or as an `Expr`. As an Expr it always has type `Int(32)`. 

```cpp
// Constructor
Halide::Var::Var 	( 	const std::string &  	n	) 	// with the given name. 
Halide::Var::Var 	( 		) 	// with an automatically-generated unique name. 

// Get the name of a Var. 
const std::string & Halide::Var::name 	( 		) 	const  
```

### 5.1.1. implicit

`static Var Halide::Var::implicit 	( 	int  	n	) 	`    
类的静态全局函数, Implicit var constructor.   用于隐式 Var 的构造

Implicit variables:  are injected automatically into a function call if
* the number of arguments to the function are fewer than its dimensionality
* and a placeholder `_` appears in its argument list. 
* Implicit var 在 Halide Header 中定义了 _0 到 _9 共10个, 这些 Var 不能被用于任何函数参数以及 Func声明

```cpp
Func f, g;
Var x, y;

// 对于 f 进行 pure definition, 赋予了 f 两个维度
f(x, y) = 3;

// 以下四种定义完全一样, 区别只是显式的 Var 和隐式的 Var
g(_) = f*3;               // g(_0, _1) = f(_0, _1)*3;
g(_) = f(_)*3;            // g(_0, _1) = f(_0, _1)*3;
g(x, _) = f(x, _)*3;      // g(x, _0) = f(x, _0)*3;
g(x, y) = f(x, y)*3;      // g(x, y) = f(x, y)*3;

// 需要清晰的明确 Var 的补充是以 右边为主的, Var 只是一个占位符, 和之前被用于如何定义完全无关, 下面的定义会将 g 定义为 4 维度的数据
g(x, y, _) = f*3;         // g(x, y, _0, _1) = f(_0, _1)*3;

// 对于一个相同的 Func 可以混用不同的 Implicit Var
Func h;       
h(x) = x*3;   // 单维度数据
// 混用, 这对于代码的理解非常不利, 应当尽量避免  
g(x) = h + (f + f(x)) * f(x, y);
g(x, _0, _1) = h(_0) + (f(_0, _1) + f(x, _0)) * f(x, y);
```


## 5.2. Expr

Expr 可以理解为表达式对象, 是一种轻量化的数据结构, 表达了一个 scalar expression.
* 一个 Expr 的定义中所使用的 Var, 代表了 Var 所可以取的所有值的运算的集合.
* Expr 可以用于实现  计算, 存储常量和变量
* 通过 Expr 的结合可以实现复杂的表达式, 可以简化一个 Halide pipeline 的定义, 完全不使用 Expr 也可以定义简单 Halide 函数.  

然而, Expr 并不是一个用于存储pipeline的中间值 (intermediate values) 的数据结构, 这是因为:
* Expr 可能需要非常重的计算
* Expr 可能不符合 Halide complier 的优化策略


事实上, 很多 Halide 头文件下的 Halide 实用函数都是以 Expr 作为输入值和返回值的. 
* 对于一个 `Func`, 那么 `typeof(Func(x,y)) == Expr`, 根据库函数头的形式决定等式两边的 Func 是用 `Func` 形式还是 `Func(x,y)` 形式


### 5.2.1. Expr routine

#### 5.2.1.1. 数值类型转换

Halide::cast 用于显式的将一个 Expr 进行内存转换

具体的写法有两种
```cpp
template<typename T >
Expr Halide::cast 	( 	Expr  	a	) 	

Expr Halide::cast 	( 	
    Type  	t,
		Expr  	a 
	) 		
```

#### 5.2.1.2. 基础数值操作

函数头
* clamp     : 相当于 min 和 max 的结合, 将一个输入的 Expr 限制在对应的区间以内  
* select    : 类似于 numpy.where 或者 C 语言中的三元表达式, 根据条件从两个输入中选择一个到输出中

clamp
```cpp
/* 
Clamps an expression to lie within the given bounds.
The bounds are type-cast to match the expression. Vectorizes as well as min/max.
 */

Expr Halide::clamp 	( 	
    Expr  	a,
		const Expr &  	min_val,
		const Expr &  	max_val 
	) 	

```


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


## 5.3. Func

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
  * 对于一个 `Func`, 它的第一个 `Pure definition` 必须是由 Var 构成的, 而不能是一些固定值
  * 对于一个 `Func`, 如果它有 `update`, 那么它的第一个 `Pure definition` 总是优先于所有 `update definition`, 会产生独立的两个 for loop

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

### 5.3.1. Func var scheduling

同实际使用的计算资源 (CPU or GPU) 无关的 scheduling 接口, 主要用于算法层面上的并行研讨
* `reorder(y,x)`    : 手动指定 loop 的执行顺序, 顺序为从左到右是最内loop到外
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

### 5.3.2. Func CPU scheduling 

对于定义好的 Func, 可以通过调用对应的句柄来让 Halide 进行并行优化, 这是一个比较有技术含量的工作, 一般的情况下交给 Halide 来自动优化就能取得比较好的效果  


计算优化 scheduling 接口 `Func.*`:

* `parallel(y)`     : 开启在某个特定坐标轴上的并行执行

* `vectorize(x);`       : 在计算的时候将某一个 loop 直接向量化加速
  * 通常和 split 来结合使用, 即 split_factor 即为硬件所支持的最大向量长度
  * `Func.split(x, x_outer, x_inner, 4);` 
  * `Func.vectorize(x_inner);`
* `unroll(x_inner);` : 把某一个轴的循环展开, 即从代码上取消 for 循环, 转而使用 `x=0,x=1,...,x=end` 的形式生成代码
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

#### 5.3.2.1. Func.tile

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

**TailStrategy  Halide::TailStrategy**


用于在 tail 的时候, 定义如何处理尾部. 即 维度的宽并不能被 factor 整除的情况下, 如何对应. 因为有的时候可以进行重复计算, 而有些时候不可以.   

感觉很有用, 先定义好章节以后再看 TODO

### 5.3.3. Func GPU scheduling

使用 GPU 时候的 pipeline scheduling 接口同纯 CPU 的接口不太一致  

可以在编译的时候为 Target 结构体添加 `Debug` flag 来显式提示所有的 GPU pipeline
`target.set_feature(Target::Debug);`


`Func.gpu_*` 相关接口:
* `gpu_tile(i, block, thread, 16)`  : 同 CPU 的tile 一样, 相当于把一个 var 进行分割, 但进行的优化处理是基于 GPU 的计算原理的
  * 相当于 `gpu_blocks` 和 `gpu_threads` 和 `tile`/`(split+reorder)` 的结合, 在实际使用的时候可以进行等价替换
  * `gpu_blocks(var block)`     : 等同于 CUDA 里的 block 
  * `gpu_threads(var thread)`   : 等同于 CUDA 里的 thread
* 


### 5.3.4. Statical declaration 静态声明

静态声明接口 (Statical declaration) `Func.*`:
* `bound(var, Expr min, Expr extent)`     : 用于静态指定某一个 Var 的 range, 最经典的莫过于 color channel, 来方便 Halide 执行某些特殊优化
  * 如果在 pipeline 生成的时候推算出需要限制以上的范围的话, 则会产生 runtime error
* 


### 5.3.5. loop 与 store 结构

`compute_*, store_*` 系列: 它用于调整整个 Halide pipeline 管线的循环嵌套逻辑: 该逻辑管理与 CPU 或 GPU 执行的设置相互独立   


* 说明:
  * **对于默认完全不加修饰的 Func stage Pure Definition, 靠前定义的 Func 会作为内联函数完全嵌入后面的 Func**
  * The default schedule fully inlines 'producer' into 'consumer'
  * 这在某些时候会导致 producer 的一些基础资源被重复计算, 因此适时的手动将一些 Func 添加计算顺序是很有必要的
* `Func.compute_*`   : 调整某个 Func 的计算循环级
* `Func.store_*`     : 调整某个 Func 的存储循环级, 该接口一般作为附加选项添加到 compute_ 上, 用以实现 存储和计算的分离, 达到更好的效果

#### 5.3.5.1. compute_* 系列接口

* `Func::compute_root 	() 	`
  * Compute all of this function once ahead of time. 
  * 在执行到 Func 的时候, 会计算所有将要被用到的值, 存储到缓存里
  * 理论上会最大化降低 redundance work, 但是会占用最多的缓存空间, 同时会显而易见的降低 cache 利用率
  * 同时, 需要考虑 cache 时效, 即该 Func 的值被利用的时候, 有可能已经被替换出 CPU cache 了, 从而导致性能下降
* `Func::compute_at ( Func, Var)`
  * Compute this function as needed for each unique value of the given var for the given calling function f. 
  * 当该 Func 被需要得时候才会计算对应的值, 且计算结果不会被存储到缓存里, 从缓存 cache 的角度上说这命中率足够高, 但是会产生非常多的重复计算
  * `Var` 参数被用来考虑 compute_at 所对应的所需要的 Func 的值
    * 比如说如果 Var 是 x 即内循环的话, 即是 计算单个像素要的 producer Func 的值并存储到缓存里
    * 如果 Var 是 y 即外循环的话, 即 计算单行像素 所需要的 producer Func 的值并存储到缓存里
    * 根据情况, 如果 Var 越靠外, 那么 licality 的性能越低, memory cost 越高, 但是 redundant work 会显著降低
  * 重载函数: `Func::compute_at 	( 	LoopLevel  	loop_level	) 	` 使用 LoopLevel 类作为参数而非 Func, Var
  * `Func::compute_at (consumer, Var)` 会同时应用到所有的 consumer stage, 因此参数并不需要指定 `consumer.update(idx)`
    * 这种情况下 producer 在被使用的时候会重复计算
    * 可以使用 sotre_root 来减少计算量, 但是需要权衡 memory cost 和 cache 命中率
* `Func::compute_with ()`  : TODO 介绍有点少


特殊使用情况
```cpp

/*
 In this case neither producer.compute_at(consumer, x)
nor producer.compute_at(consumer, y) will work, because
either one fails to cover one of the uses of the
producer. So we'd have to inline producer, or use
producer.compute_root().
 */
Func producer, consumer;
producer(x, y) = (x * y) / 10 + 8;
consumer(x, y) = x + y;
consumer(x, 0) += producer(x, x);
consumer(0, y) += producer(y, 9 - y);


// 因为直接使用会有 Var 混淆 所以可以通过定义 wrapper 来实现对不同 update stage 的 compute_at
Func producer_1, producer_2, consumer_2;
producer_1(x, y) = producer(x, y);
producer_2(x, y) = producer(x, y);

consumer_2(x, y) = x + y;
consumer_2(x, 0) += producer_1(x, x);
consumer_2(0, y) += producer_2(y, 9 - y);

// 尽管核心相同, 但是通过 wrapper 可以避免 halide 混淆 Var
producer_1.compute_at(consumer_2, x);
producer_2.compute_at(consumer_2, y);

```

#### 5.3.5.2. store_* 系列接口

从 compute 系列接口有些类似, 但是指定的不是计算过程而是存储过程, 该系列结果是 optional, 只在特殊情况下用于将 存储循环级别 以及 计算循环级别分开来, 用以达成更高水平的对 locality 和 redundant work 的 trade-off

* `Func::store_at 	( Func, Var	)`:
  * 假定 `Func g` 是 producer, `Func F` 是 consumer
  * `g.compute_at(f, x).store_at(f, y);`  所实现的效果是  
    * 在 行层面(y 循环) 上生成缓存
    * 在 x 循环上执行 compute_at
    * 由于 缓存保存在了 外循环上, 所以当 f 的计算频繁依赖 前后的 x 的时候 (`x-2, x-1` 之类的), 相关的值已经被计算过了所以会直接读取缓存
    * 相比于只有 `compute_at`的调配, 保持了完全的 locality 性能的同时, 用少量的 memory cost 显著降低了 redundant work 
  * 根据具体的 `comsumer F`, Halide 会自动进行缓存优化
    * 为假如 F 只依赖与有限范围内 x 的 `g` 值
    * 那么即使缓存 `store_at(f,y)` 定义在了 y 循环上, 也不会把整行的 g 值都存储, 而是通过 circular buffer 即循环利用 Buffer 来实现超高的 locality
  * 同样的具有 `LoopLevel` 作为参数输入的重载
* `Func::store_root () 	`
  * 同 `store_at` 等效, 但是 schedules storage outside the outermost loop. 
  * 将存储过程安排在最外层之外, 即会存储所有中间值



#### 5.3.5.3. Func.update()

获取单个下一个 update definition 句柄 , 根据 update definition 的定义顺序依次赋予 index 
* `Stage Halide::Func::update 	( 	int  	idx = 0	) 	`
* Get a handle on an update step for the purposes of scheduling it. 
* 对于 对应的 update, 各种 scheduling 接口指定应用在使用了 Var 的维度上
* Because an update does multiple passes over a stored array, it's not meaningful to inline them. So the default schedule for them does the closest thing possible. It computes them in the innermost loop of their consumer.

```cpp
// Consider the definition:
Func f;
f(x, y) = x * y;
// Set row zero to each row 8
f(x, 0) = f(x, 8);
// Set column zero equal to column 8 plus 2
f(0, y) = f(8, y) + 2;

// 对于各个 update , 只能把 scheduling 应用在 Var 的维度上
// pure definition 的 scheduling 不需要使用 update
f.vectorize(x, 4).parallel(y);

// f(x, 0) = f(x, 8); index =0 , 且只能对 x 进行调整
f.update(0).vectorize(x, 4);

// f(0, y) = f(8, y) + 2; index =1 , 且只能对 y 进行调整
Var yo, yi;
f.update(1).split(y, yo, yi, 4).parallel(yo);
```



### 5.3.6. Func realize

和 Halide 函数的 JIT 实例化相关   

### 5.3.7. Func compile_to

和 Halide 函数的 编译 相关, 这种编译方法比较原始, 不需要用到 Generator , 是独立出来的 AOT/JIT 编译方法  

`Func.compile_to_*` 系列函数 : 根据输出种类的不同还可以进行细化分类  
* static_library : 静态库, 最为实用的种类
  * `static_library`  : 将 Halide 函数编译为 `static library` , 表现为 `header pair`, 生成的函数名和参数名都可以手动指定  
  * `multitarget_static_library`  : 似乎是为了方便编译的 batch 处理, 同时为多个 Targe 编译不同的 `header pair`  
* object  : 目标文件, 作为一种较为低层的接口, 一般不会被主动用到
  * `compile_to_object`           : 将文件编译为 `.o` 或者 `.obj` 结尾的文件, generally call compile_to_static_library or compile_to_file instead
  * `compile_to_file`             : 将文件编译为 `object file and header pair`
  * `multitarget_object_files`    : 同上一个 multitarget 类似, 但是所有的输出不再是单个 bundled library, 而是分别的 object files
* 其他特殊种类: 
  * `compile_to_c`                  : 将 Halide pipeline 编译为 C 代码, 似乎只是为了 debug 用的, 矢量化和并行化都不会出现在生成的 C 代码中
* 主动 JIT 编译:
  * `compile_jit(target)`         : Eagerly jit compile the function to machine code. 
  * 默认使用 `Halide::get_jit_target_from_environment()` 的返回值作为 Target
  * 在使用 GPU 接口的时候, jit 编译不会默认生效, 需要手动定义启用了 GPU feature 的 target 结构体, 并将其传入 compile_jit, 否则会导致 CPU 模拟 GPU 处理
  



通用参数说明:
* `filename_prefix`   : 即生成的文件名, 不包括后缀, 后缀名是自动追加的
* `args`              : Halide Pipeline 所需要用到的参数
* `fn_name`           : 生成的函数的名称, 会定义在对应的输出头文件里
* `target`            : Halide 所独有的 cross-compliation 机制, generate code for any platform from any platform, 指定要输出的目标平台, 具体的操作通过配置一个 Target 结构体来实现    

```cpp
void Halide::Func::compile_to_static_library 	( 	
    const std::string &  	filename_prefix,
		const std::vector< Argument > &  	args,
		const std::string &  	fn_name = "",
		const Target &  	target = get_target_from_environment() 
	) 	

/* example */
Param<uint8_t> offset;
ImageParam input(type_of<uint8_t>(), 2);
brighter(x, y) = input(x, y) + offset;
brighter.compile_to_static_library("lesson_10_halide", {input, offset}, "brighter");



```

### 5.3.8. Func Debug

通过一系列的 Func 对象方法, 可以实现对多个环节的 dump 以及打印, 从而实现多种量级的 debug


#### 5.3.8.1. trace

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
#### 5.3.8.2. HTML 输出底层编译结果代码
通过将输出以 HTML 的形式表示, 方便查看和理解 Halide 的最佳化结果
```cpp
Func gradient("gradient");
Var x("x"), y("y");
gradient(x, y) = x + y;

// 带有语法高亮的 HTML 的 Halide 编译结果
gradient.compile_to_lowered_stmt("gradient.html", {}, HTML);
```

#### 5.3.8.3. print_loop_nest();

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

## 5.4. Buffer

Buffer 在一定程度上也是一个虚拟的 Buffer, 在定义的时候需要指定  大小或者维度  


### 5.4.1. set_min

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

是一种更加结构化的书写 Filter 的方法, Generator is a class used to encapsulate the building of Funcs in user pipelines. 
* 比起将 pipeline 定义在 main() 中, 这种方法将 pipeline 实现为函数, 更加贴合实际使用
* 具体的使用需要将所有 pipeline 定义在一个从 `Halide::Generator` 继承的自定义类中
* 使用 Generator 和 AOT 或 JIT 无关, 都可以使用, 但是对于 AOT 模式特别的方便


## 定义书写

将 pipeline 定义为一个 class  `class myGenerator : public Generator<myGenerator> `
* 具体的处理管道需要定义在 `generate()` 函数里
* 对于管道中的各种 `Func` 可以在一开始全部定义在 `private` 里
  * 保留的函数名 `generate()` 只关注算法, 如果不定义 `schedule()` 的话将 schedule 写在 generate() 里面也没问题
  * 保留的函数名 `schedule()`, which (if present) should contain all scheduling code.

```cpp
// 需要公有继承 Halide::Generator
class Blur : public Generator<Blur> {
public:
    // 将需要的 Input 元素定义为公有成员
    // 会自动的作为参数出现在 generated function, 顺序符合 在这里Input声明的顺序
    Input<Func> input{"input", UInt(16), 2};

    // 同理, 也事先定义好 pipeline 的输出部分
    Output<Func> output{"output", UInt(16), 2};
    void generate() {
        blur_x(x, y) = (input(x, y) + input(x+1, y) + input(x+2, y))/3;
        blur_y(x, y) = (blur_x(x, y) + blur_x(x, y+1) + blur_x(x, y+2))/3;
        output(x, y) = blur(x, y);
    }
    void schedule() {
        blur_y.split(y, y, yi, 8).parallel(y).vectorize(x, 8);
        blur_x.store_at(blur_y, y).compute_at(blur_y, yi).vectorize(x, 8);
    }
private:
    Var x, y, xi, yi;
    Func blur_x, blur_y;
};

```

## 定义输入输出  

定义输出的时候一般会定义为 Buffer, 但是 Halide 本质上是支持把 Func 定义为 Input 的, 以下是原话:

* An `Input<Func>` is (essentially) like an ImageParam, except that it may (or may not) not be backed by an actual buffer, and thus has no defined extents.
* 除了 Buffer 以外, Func 可以直接被定义为 Input 
  * `Input<Func> input{"input", Float(32), 2};`
  * 定义 Func 为 Input 的时候, 可以可选的将其维度和数据类型告知在大括号里
  * 如果不指明的话, 会根据输入的 Func 来进行自动推断  
  * 如果指明了 维度或这类型的话, 那么维度 or 类型匹配会被进行, 并且在不匹配的时候出现编译报错  
  * `Input<Func> input{ "input", 3 };` 只指定维度的情况

* A Generator must explicitly list the output(s) it produces:
* 一个 Generator 必须要有 `Output`, 似乎只能有一个?
  * 同样的可以把 `Func`定义为 Output 可选的将其维度和数据类型告知在大括号里, 如果指定的话会进行匹配检测  
  * Output 可以被定义为 `Tuple` 使用内嵌一个大括号来分别指定 tuple 的元素类型
  * `Output<Func> output{"output", {Float(32), UInt(8)}, 2};`
* Output 可以被定义为标量, 即 0 维数据
  * `Output<float> sum{"sum"};`
  * `Output<Func> {"sum", Float(32), 0}`
* Output 可以定义为 Func 数组  
  * `Output<Func[3]> sums{"sums", Float(32), 1};`  此时三个 Func 都是相同的类型与维度
  * 甚至 Func 数组都可以不指定大小, 但是有一定的限制
    * `Output<Func[]> pyramid{ "pyramid", Float(32), 2 };`

```cpp
// output 为 tuple 的例程
class Tupler : Generator<Tupler> {
  Input<Func> input{"input", Int(32), 2};
  Output<Func> output{"output", {Float(32), UInt(8)}, 2};
  void generate() {
    Var x, y;
    Expr a = cast<float>(input(x, y));
    Expr b = cast<uint8_t>(input(x, y));
    output(x, y) = Tuple(a, b);
  }
};
```

## 编译与库生成

* 在使用的时候和 halide 目录下的 `tools/GenGen.cpp` 一起编译

```cpp

// 告诉 GenGen.cpp 该 generator 的相关信息
HALIDE_REGISTER_GENERATOR(MyFirstGenerator, my_first_generator)
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

## 7.1. constructor

构造函数  

```cpp


/* Construct an undefined reduction domain.  */
Halide::RDom::RDom 	( 		) 	

/* Construct a multi-dimensional reduction domain with the given name.  */
/* If the name is left blank, a unique one is auto-generated.   */
/* 从 Region 来定义 RDom */
HALIDE_NO_USER_CODE_INLINE Halide::RDom::RDom 	( 	
  const Region &  	region,
  std::string  	name = "" 
	) 		

/* References Halide::min(). 应该是最常用的 定义方法 */
template<typename... Args>
HALIDE_NO_USER_CODE_INLINE Halide::RDom::RDom 	( 	
  Expr  	min,
  Expr  	extent,
  Args &&...  	args 
	) 		

/* Construct a reduction domain that iterates over all points in a given Buffer or ImageParam.  */
Halide::RDom::RDom 	( 	const Buffer< void, -1 > &  		) 	
Halide::RDom::RDom 	( 	const OutputImageParam &  		) 	
template<typename T , int Dims>
HALIDE_NO_USER_CODE_INLINE Halide::RDom::RDom 	( 	const Buffer< T, Dims > &  	im	) 	


/* Construct a reduction domain that wraps an Internal ReductionDomain object.  */
Halide::RDom::RDom 	( 	const Internal::ReductionDomain &  	d	) 	


```

## 7.2. example


### 7.2.1. RDom 的 reduction function

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

### 7.2.2. recursive function

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

### 7.2.3. scattering operation

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

# 8. Halide::Target

A struct representing a target machine and os to generate code for.  

用于具体的实现 Halide 的 cross-compliation


## 8.1. Halide 空间里的 API

获取 Target
* `Target Halide::get_host_target()`                  : 直接获取当前环境的 Target 信息
* `Target Halide::get_target_from_environment() 	`   : 获取环境变量 `HL_TARGET` 对应的 Target 信息, 否则调用 `get_host_target`
* `Target Halide::get_jit_target_from_environment()`  : 获取环境变量 `HL_JIT_TARGET` 对应的 Target 信息, 否则调用 `get_host_target`



环境验证:
* `bool Halide::host_supports_target_device(const Target & t)`  : 用于验证一个定义好的 Target 能不被当前的 host 环境所使用
  * 该函数只能用来查验绝对的 false, 不能保证任何 feature 的调用成功
  * 该检查函数不是线程安全的
* 

## 8.2. struct 结构体



## 8.3. 例程

```cpp
// Let's use this to compile a 32-bit arm android version of this code:
Target target;
target.os = Target::Android;                // The operating system
target.arch = Target::ARM;                  // The CPU architecture
target.bits = 32;                           // The bit-width of the architecture
std::vector<Target::Feature> arm_features;  // A list of features to set

target.set_features(arm_features);
// We then pass the target as the last argument to compile_to_file.
brighter.compile_to_file("lesson_11_arm_32_android", args, "brighter", target);

// And now a Windows object file for 64-bit x86 with AVX and SSE 4.1:
target.os = Target::Windows;
target.arch = Target::X86;
target.bits = 64;
std::vector<Target::Feature> x86_features;
x86_features.push_back(Target::AVX);
x86_features.push_back(Target::SSE41);
target.set_features(x86_features);
brighter.compile_to_file("lesson_11_x86_64_windows", args, "brighter", target);


// And finally an iOS mach-o object file for one of Apple's 32-bit
// ARM processors - the A6. It's used in the iPhone 5. The A6 uses
// a slightly modified ARM architecture called ARMv7s. We specify
// this using the target features field.  Support for Apple's
// 64-bit ARM processors is very new in llvm, and still somewhat
// flaky.
target.os = Target::IOS;
target.arch = Target::ARM;
target.bits = 32;
std::vector<Target::Feature> armv7s_features;
armv7s_features.push_back(Target::ARMv7s);
target.set_features(armv7s_features);
brighter.compile_to_file("lesson_11_arm_32_ios", args, "brighter", target);
```

### 8.3.1. find GPU 例程

```cpp
// A helper function to check if OpenCL, Metal or D3D12 is present on the host machine.

Target find_gpu_target() {
    // Start with a target suitable for the machine you're running this on.
    Target target = get_host_target();

    std::vector<Target::Feature> features_to_try;
    if (target.os == Target::Windows) {
        // Try D3D12 first; if that fails, try OpenCL.
        if (sizeof(void*) == 8) {
            // D3D12Compute support is only available on 64-bit systems at present.
            features_to_try.push_back(Target::D3D12Compute);
        }
        features_to_try.push_back(Target::OpenCL);
    } else if (target.os == Target::OSX) {
        // OS X doesn't update its OpenCL drivers, so they tend to be broken.
        // CUDA would also be a fine choice on machines with NVidia GPUs.
        features_to_try.push_back(Target::Metal);
    } else {
        features_to_try.push_back(Target::OpenCL);
    }
    // Uncomment the following lines to also try CUDA:
    // features_to_try.push_back(Target::CUDA);

    for (Target::Feature f : features_to_try) {
        Target new_target = target.with_feature(f);
        if (host_supports_target_device(new_target)) {
            return new_target;
        }
    }

    printf("Requested GPU(s) are not supported. (Do you have the proper hardware and/or driver installed?)\n");
    return target;
}
```