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



# 4. Var

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

## 4.1. implicit

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


# 5. Expr

Expr 可以理解为表达式对象, 是一种轻量化的数据结构, 表达了一个 scalar expression.
* 一个 Expr 的定义中所使用的 Var, 代表了 Var 所可以取的所有值的运算的集合.
* Expr 可以用于实现  计算, 存储常量和变量
* 通过 Expr 的结合可以实现复杂的表达式, 可以简化一个 Halide pipeline 的定义, 完全不使用 Expr 也可以定义简单 Halide 函数.  

然而, Expr 并不是一个用于存储pipeline的中间值 (intermediate values) 的数据结构, 这是因为:
* Expr 可能需要非常重的计算
* Expr 可能不符合 Halide complier 的优化策略


事实上, 很多 Halide 头文件下的 Halide 实用函数都是以 Expr 作为输入值和返回值的. 
* 对于一个 `Func`, 那么 `typeof(Func(x,y)) == Expr`, 根据库函数头的形式决定等式两边的 Func 是用 `Func` 形式还是 `Func(x,y)` 形式


## 5.1. Expr routine - 各种返回值为 Expr 的预定义 Func

### 5.1.1. 数值类型转换 - cast

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

### 5.1.2. 基础数值操作 - clamp

函数头
* clamp     : 相当于 min 和 max 的结合, 将一个输入的 Expr 限制在对应的区间以内  

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




### 5.1.3. 逻辑函数 select, mux

* select    : 类似于 numpy.where 或者 C 语言中的三元表达式, 根据条件从两个输入中选择一个到输出中
* mux       : 相当于直接把一系列 Expr 打包成由索引定位的 Expr

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

有时候我们会把某些拥有相同类型的 expressions 打包到某个 channel dimension.  这个时候可以用这个 mux 
* 一些性质与 select 一样, 当输入的 c (第一个参数, 索引) 超出了范围, 则返回表达式 list 的最后一个结果
* 该写法对比于 select 实现也有一些不方便的地方
  * Func 会定义成无限大的区域, 因此这方便很容易产生 Bug
  * 这个方法也同 select 一样有性能问题, 因此尽可能加上 `.bound(c, 0, <max_index>).unroll(c);`
  * 对比 select, 该Func 输入的 Expr 必须有相同的类型
```cpp
// 根据通道的值选择输出, 相当于一种方便的 select 写法
img(x, y, c) = select(c == 0, 100, c == 1, 50, 25);
img(x, y, c) = mux(c, {100, 50, 25});


// 直接通过花括号传入 
Expr Halide::mux 	( 	const Expr &  	id,
		const std::initializer_list< Expr > &  	values 
	) 	

// 通过 vector传入
Expr Halide::mux 	( 	const Expr &  	id,
		const std::vector< Expr > &  	values 
	) 		

// 通过 Tuple传入  
Expr Halide::mux 	( 	const Expr &  	id,
		const Tuple &  	values 
	) 	
```

### 5.1.4. 累积函数 - 需要输入的Expr 有 RDom


* sum         : 累加函数
* product     : 累乘函数
* maximum     : 求最大值函数
* minimum     : 求最小值函数



# 6. Tuple - small array of Exprs for func with multiple outputs

用于定义某个拥有一系列输出的 Func, 通过该方法定义的好处就是所有计算会使用同一个 x,y loop, 但是计算的存储结果会放在较远的内存位置  
* 定义为 Tuple 类型的 Func 不能再输入索引转为 Expr


```cpp
Func multi_valued;
multi_valued(x, y) = Tuple(x + y, sin(x * y));

// 使用该种方法定义的 Func 在实现后, 结果会以 vector<Buuffer<type>> 存在
Realization r = multi_valued.realize({80, 60});
assert(r.size() == 2);
Buffer<int> im0 = r[0];
Buffer<float> im1 = r[1];

// 直接通过花括号也可以定义 tuple 的 func
Func multi_valued_2;
multi_valued_2(x, y) = {x + y, sin(x * y)};

// 定义为 Tuple 类型的 Func 不能再输入索引直接转为 Expr. 需要在加上 索引后再加上 Expr 的索引
Func consumer;
// consumer(x, y) = multi_valued_2(x, y) + 10;  这里会报错
Expr integer_part = multi_valued_2(x, y)[0];
Expr floating_part = multi_valued_2(x, y)[1];

```

## 6.1. Tuple - 构造函数


## 6.2. 基于 Tuple 的 reductions

由于 所有计算会使用同一个 x,y loop, 因此对于需要保存状态信息的类似于 argmax 之类的运算, 使用通过 tuple 包装后的 Func 可以很快的实现计算
* halide 的内置 reduction `argmax argmin` 就是基于这种来实现的, 返回值是一个 tuple
  * the point in the reduction domain corresponding to that value
  * the value itself

```cpp
// 输入值
input_func(x) = sin(x);

// Pure definition. 定义一个 Tuple, 索引都为 0 , 则默认值都为输入的首位置值
arg_max() = {0, input(0)};
Expr old_index = arg_max()[0];
Expr old_max = arg_max()[1];
Expr new_index = select(old_max < input(r), r, old_index);
Expr new_max = max(input(r), old_max);
arg_max() = {new_index, new_max};

```

## 6.3. 基于 Tuple 的 Halide type system extend

通过定义自定义的结构体, 来实现对 Tuple的封装, 从而实现对特殊类型的实现  
```cpp
// Tuples for user-defined types.
// Tuples can also be a convenient way to represent compound
// objects such as complex numbers. Defining an object that
// can be converted to and from a Tuple is one way to extend
// Halide's type system with user-defined types.
struct Complex {
    Expr real, imag;

    // Construct from a Tuple
    Complex(Tuple t)
        : real(t[0]), imag(t[1]) {
    }

    // Construct from a pair of Exprs
    Complex(Expr r, Expr i)
        : real(r), imag(i) {
    }

    // Construct from a call to a Func by treating it as a Tuple
    Complex(FuncRef t)
        : Complex(Tuple(t)) {
    }

    // Convert to a Tuple
    operator Tuple() const {
        return {real, imag};
    }

    // Complex addition
    Complex operator+(const Complex &other) const {
        return {real + other.real, imag + other.imag};
    }

    // Complex multiplication
    Complex operator*(const Complex &other) const {
        return {real * other.real - imag * other.imag,
                real * other.imag + imag * other.real};
    }

    // Complex magnitude, squared for efficiency
    Expr magnitude_squared() const {
        return real * real + imag * imag;
    }

    // Other complex operators would go here. The above are
    // sufficient for this example.
};

```

# 7. Func

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

## 7.1. Func 的各种情报函数

布尔返回值的验证函数:
* `bool Halide::Func::defined 	( 		) 	const`              : 验证该 Func 是否 have at least a pure definition
* `bool Halide::Func::has_update_definition 	( 		) 	const`: 验证该 Func 是否有至少一个 update definition
* 

int 返回值的情报函数
* `int Halide::Func::dimensions 	( 		) 	const`            : 获取该 Func 的 dimensionality, 如果某个 Func 违背 defined, 但是被 specified 了维度, 也会正确返回值
* `int Halide::Func::num_update_definitions 	( 		) 	const`: 获取该 Func 的update definition 的个数

列表返回值
* `std::vector< Var > Halide::Func::args 	( 		) 	const`    : 获取该 Func 的 Pure arguments, 一个 Var 的 list

类型返回值
* `const Type & Halide::Func::type 	( 		) 	const`          : 获取 Func 的输出类型, 调用的 Func 不能含有 Tuple 的元素. 如果 Func 没有初始化, 则返回被 specified required type, 如果连指定都没有则会报 runtime error
* `const std::vector< Type > & Halide::Func::types 	( 		) 	const` : 对于 Func 含有 Tuple 时候的 Type 获取


## 7.2. bound 限制

静态的声明了一个 Func 的计算范围, 根据接口的不同有不同等级的条件
Statically declare that the range over which a function

* `Func & Halide::Func::bound 	( 	const Var &  	var,  Expr min, Expr extent )`  
  * should be evaluated is given by the second and third arguments.  即硬性要求
  * 如果在run 实现中 halide 的 bound 推理发现程序需要访问更多的范围, 该部分会报 runtime orror
* `Func & Halide::Func::set_estimate 	( 	const Var &  	var,   	const Expr &  	min,   	const Expr &  	extent )`
  * will be evaluated in the general case.  即在一般情况下的范围, 主要是用于给 auto scheduler 做 scheduling 参考的
* `Func & Halide::Func::set_estimates 	( 	const Region &  	estimates	) 	`
  * 同时为一个 Func 的所有维度进行 estimate, 相当于调用 n 次 set_estimate
  * 传入一个 The size of the estimates vector must match the dimensionality of the Func. 


主动的拓展一个 Func 的计算区域, 区域会自动包括不调用该接口情况下的既定区域, 因此只会做拓展, 不会加入任何 assert
Expand the region computed.  The region computed always contains the region that would have been computed without this directive, so no assertions are injected. 
* `Func & Halide::Func::align_extent 	( 	const Var &  	var,   		Expr  	modulus )`
  * 拓展计算区域, 使得实现的 extent 是 modulus 参数的倍数
  * 例如 `f.align_extent(x, 2)` 会使得 extent 使用是 偶数, 可能在向量化用得上
* `Func & Halide::Func::align_bounds 	( 	const Var &  	var,   	Expr  	modulus,   	Expr  	remainder = 0 ) ` 
  * 相比于 align_extent 加入了对 min 的控制
  *  the min coordinates is congruent to 'remainder' modulo 'modulus'.
  *  例如  `f.align_bounds(x, 2, 1)` forces the min to be odd and the extent to be even.



## 7.3. Func Var 分割重排


### 7.3.1. Func.split()

split() 接口没有重载
* Split a dimension into inner and outer subdimensions with the given names, where the inner dimension iterates from 0 to factor-1. 
* inner 会从 0 遍历到 factor -1
* 注意, 仅针对 `old` Var, 它可以同时被作为 outer 或者 inner 来实现复用.

```cpp
Func& Halide::Func::split 	( 	const VarOrRVar &  	old,
		const VarOrRVar &  	outer,
		const VarOrRVar &  	inner,
		const Expr &  	factor,
		TailStrategy  	tail = TailStrategy::Auto 
	) 	
```

* `split(x,x_outer,x_inner,factor)` : 将某个坐标轴的 loop 拆分成两个, 会影响最终生成的 psedocode
  * 具体的循环内的操作则是 `x_execute = x_outer * factor + x_inner;` 不会影响实际执行的顺序
  * 主要配合 verctorize 来使用, 例如将内循环直接向量化用于加速

### 7.3.2. Func.fuse()

Join two dimensions into a single fused dimension.
The fused dimension covers the product of the extents of the inner and outer dimensions given. 

`fuse(x,y,fused_x_y)`             : 和 split 相对比, 把某两个坐标轴整合成一个, 会减少最终的 loop 数量 
* 具体的循环内则是 `int y = fused / 根据输入自动推算的width; int x = fused % 根据输入自动推算的width;`  不会影响结果

```cpp
Func& Halide::Func::fuse 	( 	const VarOrRVar &  	inner,
		const VarOrRVar &  	outer,
		const VarOrRVar &  	fused 
	) 	
```



### 7.3.3. Func.reorder()

重排一个 var 集合, 从左开始是  innermost

```cpp
// 基础定义
Func& Halide::Func::reorder 	( 	const std::vector< VarOrRVar > &  	vars	) 	

// 超级长的返回值定义, 似乎是 args() 的内部实现
HALIDE_NO_USER_CODE_INLINE std::enable_if<Internal::all_are_convertible<VarOrRVar, Args...>::value, Func &>::type Halide::Func::reorder 	( 	const VarOrRVar &  	x,
		const VarOrRVar &  	y,
		Args &&...  	args 
	) 		
```


### 7.3.4. Func.tile()

tile() 是最常用的接口, 是 reorder 和 split 的整合

* `tile(x,y,x_outer,y_outer,x_inner,y_inner,4,4);` 相当于 split 和 reorder 的结合, 相当于以下的代码:
  * `Func.split(x, x_outer, x_inner, 4);`
  * `Func.split(y, y_outer, y_inner, 4);`
  * `Func.reorder(x_inner, y_inner, x_outer, y_outer);`
  * 具体的参数意思还是要参照文档
  * 通常情况下进行最优化并行不会显式调用 reorder 和 split , 因为他们是最为原始的设置, 通常情况下只使用 tile 就足够了

完整的函数重载定义如下  

```cpp
/* 
Split two dimensions at once by the given factors, and then reorder the resulting dimensions to be xi, yi, xo, yo from innermost outwards.
This gives a tiled traversal. 
实现的是 reorder , 并且 split, reorder 的顺序是固定的, 从内到外依次是 xi, yi, xo, yo 原来如此
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


### 7.3.5. TailStrategy  Halide::TailStrategy

enum Halide::TailStrategy 枚举类型, 总共有 7 种

用于在 tail 的时候, 定义如何处理尾部. 即 维度的宽并不能被 factor 整除的情况下, 如何对应. 因为有的时候可以进行重复计算, 而有些时候不可以.   

我还没用过 (RVar)
* RoundUp : 最简单, 对 extent 进行向上取整.  对于 `RVar` 来说不合法, 会导致算法的含义改变, 但是能够生成最快, 最简单的代码
* GuardWithIf : 在 loop 中加入 if 语句用来保证不会访问超过 extent 的范围. 能够保证算法永远合法
  * 不需要重新评估
  * 因为在分割的末尾加入了 if, 所以当向量化计算启用的时候, 尾部的计算会被还原到标量再进行计算
  * 不限制输入输出的大小
* Predicate : 在 loads and stores 的地方加入 if 语句, 似乎加入的量比上一条 GuardWithIf 少, 对 RVar 来说永远合法
* PredicateLoads, PredicateStores : 对 RVar 不合法, 只适用于最内层的情况
* ShiftInwards : 比较权衡的效果, 不仅支持完整的向量化, 会增加内部循环的代码大小, 但不会改变外部循环. 仅对 Pure defination 合法
* Auto : 
  * For pure definitions use ShiftInwards. 
  * For pure vars in update definitions use RoundUp.
  * For RVars     in update definitions use GuardWithIf. 


## 7.4. Func CPU scheduling 

同实际使用 CPU 的 scheduling 接口

对于定义好的 Func, 可以通过调用对应的句柄来让 Halide 进行并行优化, 这是一个比较有技术含量的工作, 一般的情况下交给 Halide 来自动优化就能取得比较好的效果  


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


### 7.4.1. Func.parallel()

简单明了, 将某一个 dimension 开启并行  
* 通常情况下都是要先把一个 Var split, 之后将 outer 来并行


```cpp
// 共两个重载, 分别是简要接口和 带 size 控制的接口
// Var 指的是 split 后的 outer, 即需要手动进行 split
Func& Halide::Func::parallel 	( 	const VarOrRVar &  	var	) 	

// Var 指的是原本的 Var
// dimension 可以通过 task_size 来控制执行范围, 这将在内部实现 split, 因此 inner Var是不可见的版本
Func& Halide::Func::parallel 	( 	const VarOrRVar &  	var,
		const Expr &  	task_size,
		TailStrategy  	tail = TailStrategy::Auto 
	) 	
```

### 7.4.2. Func.unroll()

* `unroll(x_inner);` : 把某一个轴的循环展开, 即从代码上取消 for 循环, 转而使用 `x=0,x=1,...,x=end` 的形式生成代码
  * 开包一个 循环, 往往针对拥有较小 extent 的 Var, 例如通过 split 获得的 inner
  * `Func.split(x, x_outer, x_inner, 2);`
  * `Func.unroll(x_inner);`


```cpp
// 有两种重载
// 简易版本, 需要手动提前 split
Func& Halide::Func::unroll 	( 	const VarOrRVar &  	var	) 	

// 自动版本, 包含了 split
// 会在 内部将 Var 进行 split, 同时 var 本身会被自动转化成 outer
Func& Halide::Func::unroll 	( 	const VarOrRVar &  	var,
		const Expr &  	factor,
		TailStrategy  	tail = TailStrategy::Auto 
	) 		

```

### 7.4.3. Func.vectorize()

将一个 dimension 转化成 single vector, 使得通过一次计算即可遍历完
硬性条件是该 dimension 必须是 innermost one

* `vectorize(x);`       : 在计算的时候将某一个 loop 直接向量化加速
  * 通常和 split 来结合使用, 即 split_factor 即为硬件所支持的最大向量长度
  * `Func.split(x, x_outer, x_inner, 4);` 
  * `Func.vectorize(x_inner);`

```cpp
// 两种重载
// 需要通过手动 split
Func& Halide::Func::vectorize 	( 	const VarOrRVar &  	var	) 	

// 自动, 同理 会在 内部将 Var 进行 split, 同时 var 本身会被自动转化成 outer
Func& Halide::Func::vectorize 	( 	const VarOrRVar &  	var,
		const Expr &  	factor,
		TailStrategy  	tail = TailStrategy::Auto 
	) 	
```


## 7.5. Func GPU scheduling

使用 GPU 时候的 pipeline scheduling 接口同纯 CPU 的接口不太一致  

可以在编译的时候为 Target 结构体添加 `Debug` flag 来显式提示所有的 GPU pipeline
`target.set_feature(Target::Debug);`


`Func.gpu_*` 相关接口:
* `gpu_blocks(var block)`     : 等同于 CUDA 里的 block 
* `gpu_threads(var thread)`   : 等同于 CUDA 里的 thread
* `gpu_tile`

姑且在教程里使用的顺序是
```cpp
lut.gpu_blocks(block).gpu_threads(thread);
```

### 7.5.1. Func.gpu_threads()

GPU 里 threads 是最小的执行单元, 多个 threads 组成一个 block

该接口告诉 Halide 哪个 dimension 要利用 GPU 的 threads

该接口主要用于在 block 中控制计算的时候, 详细的控制 how that function's dimensions map to GPU threads.
如果该接口被执行, 然而 target 并没有 GPU 设备, 则会被当作普通的 cpu parallel


共同参数
*  		DeviceAPI  	device_api = DeviceAPI::Default_GPU 
```cpp
// 所有参数都是 const VarOrRVar &, 一共三种, 分别对应 1~3 个 Var
thread_x 
thread_x, thread_y
thread_x,thread_y,thread_z
```

### 7.5.2. Func.gpu_blocks()

告诉 GPU 如何调用 block indices
* 对于在各个 block 里串行运行的 计算 stage 很有用
* If the selected target is not ptx, this just marks those dimensions as parallel. 
* 什么是 ptx ?? 

共同参数
*  		DeviceAPI  	device_api = DeviceAPI::Default_GPU 

```cpp
// 所有参数都是 const VarOrRVar &, 一共三种, 分别对应 1~3 个 Var

block_x
block_x,block_y
block_x,block_y,block_z
```

### 7.5.3. Func.gpu_tile()

同 CPU 的 tile 一样, 也是一个 short-hand 函数
* tiling a domain and mapping the tile indices to GPU block indices 
* 这个函数相当于 tile, gpu_blocks, gpu_threads 三个函数的集合 
* the coordinates within each tile to GPU thread indices. 
传入该函数的 Var 会被 消耗 `consumes`, 因此文档中说要先执行其他的 scheduling 


共同参数:
* (TailStrategy) tail = TailStrategy::Auto
* (DeviceAPI) device_api = DeviceAPI::Default_GPU 
* (const Expr & ) _size

```cpp
// 6 种格式的重载, 分别是 1~3 个Var, 以及是否有 block
// Func& Halide::Func::gpu_tile

// 所有 Var 输入都是 const VarOrRVar &
// _size 的格式都是  		const Expr & 
x, bx, tx, x_size
x, tx, x_size

x,y,bx,by,tx,ty, x_size, y_size
x,y,tx,ty, x_size, y_size

x,y,z,bx,by,bz,tx,ty,tz,x_size,y_size,z_size
x,y,z,tx,ty,tz,x_size,y_size,z_size
```

* 相当于 `gpu_blocks` 和 `gpu_threads` 和 `tile`/`(split+reorder)` 的结合, 在实际使用的时候可以进行等价替换

## 7.6. Statical declaration 静态声明

静态声明接口 (Statical declaration) `Func.*`:
* `bound(var, Expr min, Expr extent)`     : 用于静态指定某一个 Var 的 range, 最经典的莫过于 color channel, 来方便 Halide 执行某些特殊优化
  * 如果在 pipeline 生成的时候推算出需要限制以上的范围的话, 则会产生 runtime error
* 


## 7.7. loop 与 store 结构, specialize

`compute_*, store_*` 系列: 它用于调整整个 Halide pipeline 管线的循环嵌套逻辑: 该逻辑管理与 CPU 或 GPU 执行的设置相互独立   


* 说明:
  * **对于默认完全不加修饰的 Func stage Pure Definition, 靠前定义的 Func 会作为内联函数完全嵌入后面的 Func**
  * The default schedule fully inlines 'producer' into 'consumer'
  * 这在某些时候会导致 producer 的一些基础资源被重复计算, 因此适时的手动将一些 Func 添加计算顺序是很有必要的
* `Func.compute_*`   : 调整某个 Func 的计算循环级
* `Func.store_*`     : 调整某个 Func 的存储循环级, 该接口一般作为附加选项添加到 compute_ 上, 用以实现 存储和计算的分离, 达到更好的效果

### 7.7.1. specialize 分支外迁

`Stage Halide::Func::specialize 	( 	const Expr &  	condition	) 	`

对于 Halide 的 select 操作, 正如说明里那样, 会分别计算 True False 的状态, 并在最终才进行选择, 而 specialize 接口可以让这个 分支外迁到循环外围, 是的针对某一个 condition 从最初开始就分别执行不同的代码, 这也导致可以针对不同的 condition 来执行不同的 schedule  

```cpp
// 定义需要用 select 的 Func 
f(x) = x + select(cond, 0, 1);

// 在此基础上直接将 cond 作为该接口的参数传入即可  
f.specialize(cond)

// 每个 specialize 会有自己的 schedule, 想要重新对某个 cond 进行 scheduling, 则重新调用该 specialze 并传入对应的 cond 即可
f.specialize(width == 1); // Creates a copy of the schedule so far.
f.unroll(x, 2); // Only applies to the unspecialized case.
f.specialize(width > 1).unroll(x, 2);

// 每个 Func 在执行的时候会进入第一个匹配的 specialize, 因此如果涉及到多个 cond 的时候在代码的编排上要注意
```

对于 compute_at 类的 schedule, 在和 schedule 一起使用的时候更是要特别注意 
* `The Var in the compute_at call to must exist in all paths`
* 对 prior 函数调用 compute_at 的时候, 如果使用的是 后继函数所独有的 Var, 那么需要在代码上通过 splits, fuses, renames 进行一下创造

```cpp
// prior 函数和后继函数
g(x, y) = 8*x;
f(x, y) = g(x, y) + 1;

// 对后继函数进行 sepcialize
f.compute_root().specialize(cond);

// 特殊 Var
Var g_loop;
// f 本身不存在 g_loop, 因此对 g 使用 g_loop 来优化 f 的时候, 需要让 f 在每个 path 中都出现名为 g_loop 的 var
f.specialize(cond).rename(y, g_loop);
f.rename(x, g_loop);
g.compute_at(f, g_loop);

```


### 7.7.2. compute_* 系列接口

* `Func::compute_root 	()`
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

### 7.7.3. store_* 系列接口

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



### 7.7.4. Func.update()

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



## 7.8. Func realize

和 Halide 函数的 JIT 实例化相关   

## 7.9. Func compile_to   - AOT/JIT

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

## 7.10. Func Debug

通过一系列的 Func 对象方法, 可以实现对多个环节的 dump 以及打印, 从而实现多种量级的 debug


### 7.10.1. trace

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
#### 7.10.1.1. HTML 输出底层编译结果代码
通过将输出以 HTML 的形式表示, 方便查看和理解 Halide 的最佳化结果
```cpp
Func gradient("gradient");
Var x("x"), y("y");
gradient(x, y) = x + y;

// 带有语法高亮的 HTML 的 Halide 编译结果
gradient.compile_to_lowered_stmt("gradient.html", {}, HTML);
```

### 7.10.2. print_loop_nest();

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

### 7.10.3. debug_to_file  dump图像

`void Halide::Func::debug_to_file 	( 	const std::string &  	filename	) 	`

会把一个 Func 当前的 值 dump 出来.
* 如果提供的 filename 以 `.tif` `.tiff` 结尾, 则在 dump 的时候会直接存储为 TIFF 图像格式  
* 如果提供了 `.tmp`, 则该文件可以直接被 `ImageStack`  读取
* 否则的话 会以特定的二进制格式 `byte-order` 存储
  * 为首的`20 byte-header` 
    * 前 4 个 float 存储了该 Func 的前 4 个维度值
    * 第 5 个为 32-bit int 存储了数据格式

数据格式对照表  :
* float = 0
* double = 1
* uint8_t = 2
* int8_t = 3
* uint16_t = 4
* int16_t = 5
* uint32_t = 6
* int32_t = 7
* uint64_t = 8
* int64_t = 9


## 7.11. Halide::BoundaryConditions - 边界条件

用于自动应对超出边界的访问, 根据设定自动生成边界外的数值
* 所有接口都接受一个 Func 并返回一个 Func
* 根据需要设定 Boundary 的维度, 因为有些时候不需要对所有维度都设置 Boundary, 例如 RGB 图像的 Channel

目前 Halide 所提供的边界种类有 5种 :
* constant_exterior
* repeat_edge
* repeat_image
* mirror_image
* mirror_interior


## 7.12. Func update() 与 stage

`Stage Halide::Func::update 	( 	int  	idx = 0	) 	`
* 不同于 pure definition, 后继的 update 运算在 Halide 中会被记为 Stage
* 通过 update 接口可以获取 对应 update 的 Stage 来分别进行 schedule

Stage:  A single definition of a Func.  May be a pure or update definition. 

### 7.12.1. rfactor

属于 Halide 自动性能优化的一个功能  

`Func Halide::Stage::rfactor 	( 	std::vector< std::pair< RVar, Var >>  	preserved	) 	`
* 通过调用 rfactor, 可以在指定维度上将 原本的 update definition 分割为子定义  
* 接口返回一个 Func, 这也就可以直接对子定义进行 schedule


# 8. Halide::Runtime Buffer 

Halide::Runtime 命名空间下定义了一些莫名奇妙的结构体, 以及一个类 Buffer 

## 8.1. Halide::Runtime::Buffer\<T\>

Buffer 在一定程度上也是一个虚拟的 Buffer, 在定义的时候需要指定  大小或者维度  
主要用来从 C++ 向 Halide 空间传递数据使用

作为 类的 Buffer 主要是 
A templated Buffer class that wraps `halide_buffer_t` and adds functionality
具体的数据相关的信息仍然需要参照 struct halide_buffer_t

在使用 C++ 的时候, 管理 Halide Buffer 的最好的形式, 在 stack 上仅仅只会额外使用 16 bytes

定义
`template<typename T = void, int Dims = AnyDims, int InClassDimStorage = (Dims == AnyDims ? 4 : std::max(Dims, 1))> `
`class Halide::Runtime::Buffer< T, Dims, InClassDimStorage >`

含义
* typename T    : buffer 的元素类型, 如果元素类型位置或者不唯一, 则使用 void 或者 const void
* int Dims      : 维度数, 如果维度 位置, 或者可能会变化, 则使用默认值 AnyDim
* int InClassDimStorage : 在类的内部的数据存储维度数, 设置成期望该缓冲区的最大维度. 如果超过, 则会分配堆存储来动态跟踪缓冲区的形状  
* 该类可以通过提供的分配器分配的共享指针来构建, 如果不提供指针则通过 malloc 和 free 来单独管理内存. 且只有 host 端分配了内存后, 主机端才会被视为拥有内存.  

### 8.1.1. 构造函数

目前 Doc 上显式 Buffer 有 22 个构造函数重载, 进行一下分类  

默认系列
* 默认构造 `Buffer 	( 		)`
* 拷贝构造, 只拷贝对象, 不拷贝数据  `Buffer 	( 	const Buffer< T, Dims, InClassDimStorage > &  	other	) `
* 移动构造  `Buffer 	( 	Buffer< T, Dims, InClassDimStorage > &&  	other	) 	`

共享指针


简单自定维度, 分配内存并填入数据 0
* 一维                  `Buffer 	( 	int  	first	)`
* 二维                  `Buffer 	( 	int  	first, int  	second, Args...  	rest )`
* 多维                  `Buffer 	( 	const std::vector< int > &  	sizes	) 	`
* 未知类型, 多维        `Buffer 	( 	halide_type_t  	t, const std::vector< int > &  	sizes )`


### 8.1.2. 内存管理方法

* device_dirty()      获取device内存是否 dirty 的 flag
* host_dirty()        获取 host 内存是否 ditry
* set_host_dirty(bool  	v = true	)  Methods for managing any GPU allocation.  管理任何 GPU 分配? 没懂
* set_device_dirty(bool  	v = true	)  
* copy_to_host(void *  	ctx = nullptr)      拷贝到 主内存 


```cpp
bool Halide::Runtime::Buffer< T, Dims, InClassDimStorage >::host_dirty 	( 		) 	const
HALIDE_ALWAYS_INLINE bool Halide::Runtime::Buffer< T, Dims, InClassDimStorage >::device_dirty 	( 		) 	const

HALIDE_ALWAYS_INLINE void Halide::Runtime::Buffer< T, Dims, InClassDimStorage >::set_host_dirty 	( 	bool  	v = true	)
void Halide::Runtime::Buffer< T, Dims, InClassDimStorage >::set_device_dirty 	( 	bool  	v = true	) 	

int Halide::Runtime::Buffer< T, Dims, InClassDimStorage >::copy_to_host 	( 	void *  	ctx = nullptr	) 	
```
### 8.1.3. 杂

set_min
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

## 8.2. halide_buffer_t Struct Reference

The raw representation of an image passed around by generated Halide code.

包括了一些 stuff 用于追踪 image 是否在主内存上. 如果要使用更加方便的 C++ 封装, 就使用上一章的 `Halide::Buffer<T>`


# 9. Generator - Halide 精髓

是一种更加结构化的书写 Filter 的方法, Generator is a class used to encapsulate the building of Funcs in user pipelines. 
* 比起将 pipeline 定义在 main() 中, 这种方法将 pipeline 实现为函数, 更加贴合实际使用
* 具体的使用需要将所有 pipeline 定义在一个从 `Halide::Generator` 继承的自定义类中
* 使用 Generator 和 AOT 或 JIT 无关, 都可以使用, 但是对于 AOT 模式特别的方便


## 9.1. 定义书写

将 pipeline 定义为一个 class  `class myGenerator : public Generator<myGenerator> `
* 具体的处理管道需要定义在 `generate()` 函数里
* 对于管道中的各种 `Func` 可以在一开始全部定义在 `private` 里
  * 保留的函数名 `generate()` 只关注算法, 如果不定义 `schedule()` 的话将 schedule 写在 generate() 里面也没问题
  * 保留的函数名 `schedule()`, which (if present) should contain all scheduling code.
  * 保留的函数名 `configure()`, 会在 `generator()` 之前被调用, 用于动态地将输入和输出添加到生成器, 可以用该函数来实现 GeneratorParams 的验证
    * 该函数中应该书写的是
    * `add_input` 和 `add_output`  来添加输入和输出, 在代码生成的时候会附在 预先声明的成员之后  (输入总是在输入之前)
    * `set_type` `set_dimensions` `set_array_size` 用于在未指定类型的输入输出上 动态的设置

对于 public 中的成员, 需要按照如下顺序进行定义, 所有的成员都应该一起定义一个与变量名相同的 显式名称    
```cpp
GeneratorParam(s)
Input<Func>(s)
Input<non-Func>(s)
Output<Func>(s)
```

完整的示例代码  
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

## 9.2. 定义输入输出  

定义输出的时候一般会定义为 Buffer, 但是 Halide 本质上是支持把 Func 定义为 Input 的, 以下是原话:

在这种情况下 编译好的 C 函数声明的输入类型是 halide_buffer_t, 而其他类型则是 适当的 C++ 标量类型 

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

## 9.3. GeneratorParam   - 编译时候的动态参数  

用于在 Halide 库编译生成时候的参数指定, 在 Generator 生成的时候半动态的调整程序的行为

GeneratorParams 所支持的数据种类
* any float or int type, 数字种类是支持设置最大最小值的.  
* bool
* enum
* Halide::Type
  * 本质上仍然是一个 enum, 只不过是库内部预定义的
  * Halide::Type is treated as though it were an enum, with the mappings:
    * "int8" Halide::Int(8) "int16" Halide::Int(16) "int32" Halide::Int(32) "uint8" Halide::UInt(8) "uint16" Halide::UInt(16) "uint32" Halide::UInt(32) "float32" Halide::Float(32) "float64" Halide::Float(64)
* Halide::Target
* std::string : 应该尽量避免直接使用 string , 转而 同 enum 来代替

预定义的两个 GeneratorParams : 
* `GeneratorParam<Target> target{"target", Target()};`  用于实现 Generator 的 cross-compile, 在Generator输出的时候该参数是必须指定的 
* `GeneratorParam<AutoschedulerParams> autoscheduler{"autoscheduler", {}}` 用于指定 是否/如何 使用 autoscheduler 来生成代码


```cpp
// You can define GeneratorParams of all the basic scalar types. 

// For numeric types you can optionally provide a minimum and maximum value.
// [2/4]
template<typename T >
Halide::GeneratorParam< T >::GeneratorParam 	( 	const std::string &  	name,
		const T &  	value,
		const T &  	min,
		const T &  	max 
	) 	
// 带有范围限定的参数
GeneratorParam<float> scale{"scale",
                            1.0f /* default value */,
                            0.0f /* minimum value */,
                            100.0f /* maximum value */};




// 枚举类型的参数  [3/4]
// To make this work you must provide a mapping from strings to your enum values.
template<typename T >
Halide::GeneratorParam< T >::GeneratorParam 	( 	const std::string &  	name,
		const T &  	value,
		const std::map< std::string, T > &  	enum_map 
	) 	

// 要使用的枚举类
enum class Rotation { None,
                      Clockwise,
                      CounterClockwise };
GeneratorParam<Rotation> rotation{"rotation",
                                  /* default value */
                                  Rotation::None,
                                  /* map from names to values */
                                  {{"none", Rotation::None},
                                    {"cw", Rotation::Clockwise},
                                    {"ccw", Rotation::CounterClockwise}}};


// [1/4] 
template<typename T >
template<typename T2 = T, typename std::enable_if<!std::is_same< T2, std::string >::value >::type * = nullptr>
Halide::GeneratorParam< T >::GeneratorParam 	( 	const std::string &  	name,
		const T &  	value 
	) 	
// [4/4]
template<typename T >
Halide::GeneratorParam< T >::GeneratorParam 	( 	const std::string &  	name,
		const std::string &  	value 
	) 	
//  bool 类型的参数
GeneratorParam<bool> parallel{"parallel", /* default value */ true};


// 对于 Buffer 在编译的时候动态指定数值类型   为  <buffer的名字>.type=<Halide::Type>
Output<Buffer<void, 2>> output{"output"};
output(x, y) = cast(output.type(), before_cast(x, y));
/* ./lesson_15_generate -g my_second_generator -f my_second_generator_1 -o . \
target=host parallel=false scale=3.0 rotation=ccw output.type=uint16 */

```

## 9.4. 编译与库生成

* 在使用的时候和 halide 目录下的 `tools/GenGen.cpp` 一起编译, GenGen 里面包含了 main 函数以及对应的 CLI , 用于在后续中生成对应的库 
* 需要在代码中告诉 Halide 要生成为 generator 的信息
```cpp
// 告诉 GenGen.cpp 该 generator 的相关信息
HALIDE_REGISTER_GENERATOR(MyFirstGenerator, my_first_generator)
```

# 10. RDom 

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

## 10.1. constructor

构造
函数  

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
## 10.2. RDom where

首先 RDom 只能在指定维度上进行范围指定, 最终的效果从 2维上看只能是矩形.  通过 where 的设置即可让 RDom 实现更加复杂的范围设定  
* `void Halide::RDom::where 	( 	Expr  	predicate	) 	`
* 传入的 predicate 应该是一个 值为 boolean 的 Expr
* 在 RDom 的定义里, 虽然没有硬性要求, 但对于最终的 where 范围定义尽可能小的 RDom 有助于性能优化
* where 可以多次调用, 从而实现及其复杂的范围选定


```cpp
RDom r(0, 7, 0, 7);
r.where((r.x - 3) * (r.x - 3) + (r.y - 3) * (r.y - 3) <= 10);


// Next, let's add the three predicates to the RDom using
// multiple calls to RDom::where
r.where(r.x + r.y > 5);
r.where(3 * r.y - 2 * r.x < 15);
r.where(4 * r.x - r.y < 20);

```


## 10.3. RDom example

### 10.3.1. RDom 的 reduction function

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

### 10.3.2. recursive function

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

### 10.3.3. scattering operation

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
# 11. Type : The Halide type system 数据类型系统

Halide 专有的数据类型其实不多, 类型通过 `Halide::Type` 类来进行管理  

```cpp
Type valid_halide_types[] = {
    UInt(8), UInt(16), UInt(32), UInt(64),
    Int(8), Int(16), Int(32), Int(64),
    Float(32), Float(64), Handle()};
```

基础规则:
* Var 一般通过 int32 来表示
* 大部分 内置的 halide 计算函数, 例如三角函数都是返回 float32 的
* `cast` 函数可以直接以 Halide::Type 对象作为参数, 来动态的确定 cast 的目标, 具体查看 cast 的函数定义
* 通过 Func.type() 或者 Func.types() 来验证 Func 的返回值类型

隐式转换规则:
* 在 Handle() 类型上运行算数运算符或者强制转换是 error
* 如果 types 是相同的, 则不会发生 cast
* 如果运算符的单方是小数, 则 整数类型会被转化成小数, 这有可能会导致大整数的精度丢失, 因为 float32 的精度只有24位
* 如果运算符双方都是小数, 则低精度会被转化成高精度小数
* 如果运算发生在 halide Expr 和 C++ 的整数, 则 C++ int 会被转化成 Expr 的数据类型, 这有可能会导致整个程序出错, 例如 `uint8 + 257` 
* 如果双方都是 uint, 则低精度会被转化成高精度的 uint
* 如果双方分别是 uint 和 int, 则会同时发生 uint2int 以及精度提升
  * 对于大的 uint 数, 转化为同样精度的 int 有可能会 overflow
  * 对于 uint2int 以及精度提升, 总是会先进行精度调整, 在进行符号转换
  * uint8 255 -> int32  255  先精度后符号
  * int8 -1 -> uint16 65535  先精度后符号

Handle() 类型
* 对任何 指针 应用 `type_of` 都会返回 `Handle()` 
* Handle() 总是会以 64 位来存储, 不论 compilation 的目标是否是别的位数
* handle 类型 Expr 的主要作用就是将 Expr 传递给外部的代码  




## 11.1. Halide::Type 类

各种 Type 的成员函数可以用于对类型进行更改或者验证, type 的类可以使得 halide 的函数能够动态的处理各种类型的输入数据, 而无需定义重载或者 template  

## 11.2. Halide 空间下的类型相关函数

* `template<typename T >  Type Halide::type_of 	( 		) 	`     : 构造等价于一个 C 类型的 halide Typle 对象

cast 函数
* `Expr Halide::cast 	(Type t, Expr  	a )`                      : 不使用 template 的 cast, 将 Expr 转化为某个 Halide::Type() 类型
* ` template<typename T > Expr Halide::cast 	( 	Expr  	a	)`  : 使用 template, 将 Expr 转化为某个 C++ 的数据类型

# 12. Halide::Internal

位于 Internal 空间下的成员都算是 Halide 的内部构造, 了解一些相关类可以快速的理解代码

## 12.1. Halide::Internal::Dimension


* `Expr Halide::Internal::Dimension::min 	( 		) 	const`    : 获取一个 Expr 代表图像的该 dimension 的最小坐标
* `Expr Halide::Internal::Dimension::max 	( 		) 	const`    : 获取一个 Expr 代表图像的该 dimension 的最大坐标
* `Expr Halide::Internal::Dimension::extent 	( 		) 	const`  : 获取 Expr 表示图像在该 dimension 的 extent
* `Expr Halide::Internal::Dimension::stride 	( 		) 	const`  : 获取一个 Expr 代表该图像在该 dimension 下的步长
* `Dimension Halide::Internal::Dimension::set_min 	( 	Expr  	min	) 	`   : 将给定维度的最小值设置成 min, 一般设置成 0 可以简化寻址数学
* `Dimension Halide::Internal::Dimension::set_extent 	( 	Expr  	extent	) 	` : 设置对应维度的 extent
  * 主动设置 extent 可以让 halide简化一些边界检查
  * `im.dim(0).set_extent(100);`
  * `im.dim(0).set_extent(im.dim(1).extent());`  方形 extent
  * `im.dim(0).set_extent((im.dim(0).extent()/32)*32);` extent 是 32 的倍数
* `Dimension Halide::Internal::Dimension::set_bounds 	( 	Expr  	min, Expr  	extent )` : 同时设置 min 和 extent 		
* `Dimension Halide::Internal::Dimension::set_estimate 	( 	Expr  	min, Expr  	extent ) ` : 同其他 estimate 一样, 主要用于 auto-scheduler , 没有实际上的限制作用  
  * 两个分开的单独 estimate 命令
  * `Expr Halide::Internal::Dimension::min_estimate 	( 		) 	const`
  * `Expr Halide::Internal::Dimension::extent_estimate 	( 		) 	const`
* `Dimension Halide::Internal::Dimension::set_stride 	( 	Expr  	stride	) 	` : 设置对应维度的 stride
  * 主要用于更好的进行 vectoring.  
  * Known strides for the vectorized dimension generate better code. 

* `Dimension Halide::Internal::Dimension::dim 	( 	int  	i	) 	const`    : 获取该 dimension 所属的 buffer 的其他 dimension



# 13. Target

A struct representing a target machine and os to generate code for.  

用于具体的实现 Halide 的 cross-compliation


## 13.1. Halide 空间里的 API

获取 Target
* `Target Halide::get_host_target()`                  : 直接获取当前环境的 Target 信息
* `Target Halide::get_target_from_environment() 	`   : 获取环境变量 `HL_TARGET` 对应的 Target 信息, 否则调用 `get_host_target`
* `Target Halide::get_jit_target_from_environment()`  : 获取环境变量 `HL_JIT_TARGET` 对应的 Target 信息, 否则调用 `get_host_target`



环境验证:
* `bool Halide::host_supports_target_device(const Target & t)`  : 用于验证一个定义好的 Target 能不被当前的 host 环境所使用
  * 该函数只能用来查验绝对的 false, 不能保证任何 feature 的调用成功
  * 该检查函数不是线程安全的
* 

## 13.2. struct 结构体



## 13.3. 例程

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

### 13.3.1. find GPU 例程

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