# GNU Octave 


# 1. octave language basic

`help 指令` 调出关于命令的帮助文档  
`ctrl + c` 中止正在运行的程序  
octave 的下标从`1`开始  

## 1.1. 命令行 文件操作  

在交互界面下,同 linux 一致的命令  
| 命令                               | 功能                          |
| ---------------------------------- | ----------------------------- |
| `pwd`                              | 打印当前目录,默认是程序的目录 |
| `cd '' `                           | 转移到目标目录, 需要用单引号  |
| `ls`          w                     | 打印当前目录的文件            |
| `addpath('c:\Users\long\Desktop')` | 添加路径到搜索目录            |

其他命令  
| 命令                      | 功能                                               |
| ------------------------- | -------------------------------------------------- |
| `who`                     | 输出当前工作空间存在的变量名                       |
| `whos`                    | 输出当前工作空间存在的变量名, 以及size bytes class |
| `clear`                   | 清楚当前工作空间的所有变量                         |
| `load 文件名`             | 读取文件                                           |
| `load('文件名')`          | 文件名放在字符串里读取                             |
| `save 文件名 变量`        | 将变量的数据以`文件名`保存输出                     |
| `save 文件名 变量 -ascii` | 将变量的数据以`文件名`保存输出, 以text形式         |

## 1.2. 基础运算符

| 运算符   | 功能           |
| -------- | -------------- |
| +-*/     | 加减乘除       |
| ^        | 指数           |
| ==       | 等于,返回0/1   |
| ~=       | 不等于,返回0/1 |
| &&       | 与运算         |
| `||`     | 或运算         |
| xor(a,b) | 异或ab         |

## 1.3. 基本命令

字符串用单引号括起来表示  

`;` 分号加在交互界面的命令末尾, 会隐藏输出  
`disp()`  更高级的输出方法  
`sprintf()` 类似于传统C的格式化输出函数  

例:  
`disp(sprintf('6 decimals %0.6f',a))`   将会把变量a以小数点后6位的形式输出  

`format`  在交互式界面下更改默认输出格式  
例:  
`format long`  长格式输出小数  
`format short` 短格式输出小数  
`hist(a,b)`  根据数列a生成直方图 , b 为可选参数, 指定分成多少柱  
`size(A)`  可以返回一个矩阵的维度 , 返回值同时也是一个 1 X 2 的矩阵  
`size(A,1)`  可以返回一个矩阵的行数 , 返回值是整数  
`size(A,2)`  可以返回一个矩阵的列数 , 返回值是整数  
`length(A)`  返回一个数组或者向量的长度  

## 1.4. 变量定义

`a=1;`  基础定义变量
`A=[1 2;3 4; 5 6]` 定义矩阵  
`V=[1 2 3]` 定义向量 (横向量)  
`V= 1:0.1:2 ` 自动定义等间距数组  `起始值:间距:末尾值`  或者默认间距为一 `起始值:末尾值`


`C= eye (a)` 定义(a X a)单位矩阵 
`C= ones (a,b)` 定义(a X b)全一矩阵 
`C= zeros (a,b)` 定义(a X b)全零矩阵 
`C= rand (a,b)` 定义(a X b)随机数矩阵 随机数0~1   
`C= randn (a,b)` 定义(a X b)随机数矩阵  服从高斯分布 平局值和标准差都是0    

## 1.5. 矩阵操作  

(:)表示切片     `v= a(1:10)`  取出向量a 中的前10个元素赋值给v  
                `v=a(:)`     取出矩阵a的所有元素赋值给一个数组
(,)表示索引     `b= a(3,2)`   取出a的3行2列的元素赋值给b  
(,:)`:`表示整行 `b= a(3,:)`   取出a的3行所有元素复制给b  
()中加入数组    `b= a([1 2],:)`取出a的第一行和第三行所有元素给b  


矩阵修改  
`A(:,2) = [1;2;3]` 将矩阵A的第二列更改为 1,2,3  

矩阵组合, `,` 逗号表示同行, `;` 分号表示下一列, `[]` 方括号表示组合  
`A=[A,[100;101;102]]` , 在A的右边加一列,加一个 100,101,102 的竖向量  
`C=[A B]`或者`C=[A,B]`  B在A的右边  
`C=[A;B]`               B在A的下边    

矩阵的单独元素运算 `.`  点运算  
`A .* B`  A和B是相同维度的矩阵,各自对应的元素相乘  
`A .^ 2`  A矩阵的所有元素各自平方  
使用 `.` 运算符可以进行多种不同维度数据的运算  
`1 ./ A` 对每个A的元素都被1除   

对于布尔运算,默认会对矩阵的每个元素都进行比较  
`b= A<3`  结果是只含有 0或1 的与A相同纬度的矩阵  
对于布尔运算, 可以使用 find()来找到结果为真的元素的索引  
`find(a<3)`  返回数列a中元素小于三的索引  

矩阵的转置符 `'`  `A'` 转置矩阵A  
矩阵的逆运算 `pinv()`  


其他运算,都可以直接对矩阵运算  
| 运算符        | 功能                                 |
| ------------- | ------------------------------------ |
| abs           | 绝对值                               |
| exp           | e指数                                |
| log           | 求对数                               |
| sum           | 求和                                 |
| prod          | 求全部元素乘积                       |
| ceil/floor    | 取整                                 |
| max(a)        | 获取最大值,对于矩阵默认是对列        |
| max(a,b)      | 两个矩阵对应位置的最大值组成的新矩阵 |
| max(a,[],1/2) | 获取每行或者每列的最大值,1表列,2表行 |
| sum(a,1/2)    | 获取每行或者每列的和,1表列,2表行     |
若要获取整个矩阵的最大值可以用 `max(max(A))` 或者 `max(a(:))`  
同理,对于求和也一样  

## 1.6. 条件循环操作  

基本上同python和 Matlab 一致 

```m
index=1:10
<!-- for 循环 -->
for i=1:10,
  v(i)=1:10
end;
```
## 1.7. while 循环

```m
while i<=5,
  v(i)=100;
  i=i+1;
<!-- if语句不用加括号, 逗号结尾 -->
  if i==6,
    break;
  elseif i==5,
    disp('The value is 5');
  end;
end;
```

## 1.8. 函数 function

```octave
function y = squareThisNumber(x)
  y1=x^2;
function [y1,y2] = squareThisNumber(x)


```


# 2. 数据可视化  

## 2.1. plot()

头两个参数  x 是横坐标数据 y是纵坐标数据   
`plot(x,y,'rx')`  指定点的标识为红色叉,    
`hold on;`  用于保持图片, 用于在图中画多条线  

`xlabel('字符串')`  
`ylabel('字符串')`  指定横纵坐标名称  
`title('字符串')`  指定图标题  
`legend(字符串,字符串,...)`  按顺序给线添加标识  

把图输出到文件  
`print -dpng '名称.png'`   
`close`  关闭所有图像  
`clf`  清除缓存里的所有图像  

使用figure() 命名不同图像  
`figure(1); plot(...)`    
`figure(2); plot(...)`  建立两个不同的图  

使用subplot(a,b,c)建立子图, 将图分成 aXb个子图,并选定c子图  
`subplot(1,2,1); plot(...)`    
`subplot(1,2,2); plot(...)`    

使用axis([])更改 坐标轴的范围,即选择图的范围  
`axis([x轴起始 x轴终止 y轴起始 y轴终止])`    


## 2.2. imagesc()

`imagesc(A)`  可视化一个矩阵A,根据数据分配颜色  
`colorbay`  为上面建立的图像增加一个数据说明条  
`colormap gray` 只是用灰度颜色,用深浅来表示数据的大小  


## 2.3. Regression 数据分析

```python
function [jVal, gradient] = costFunction(theta)
  jVal = [...code to compute J(theta)...]
  gradient = [...code to compute derivative of J(theta)...]
end

options = optimset('GradObj', 'on', 'MaxIter', 100)
initialTheta = zeros(2,1)
   [optTheta, functionVal, exitFlag] = fminunc(@costFunction, initialTheta, options)

```

# 3. CourseaML 实现代码

## 3.1. 单特征线性回归 Linear Regression

### 3.1.1. 数据处理
```octave

% 生成5*5单位矩阵
A = eye(5)

% 提取数据 , 从 txt 文件
data = load('ex1data1.txt');
X = data(:, 1); y = data(:, 2);
m = length(y); % number of training examples

% Add a column of ones to x
X = [ones(m, 1), data(:,1)]; 
% initialize fitting parameters
theta = zeros(2, 1); 

% Some gradient descent settings
iterations = 1500;
alpha = 0.01;

% run gradient descent
theta = gradientDescent(X, y, theta, alpha, iterations);

% 作图画出训练数据和拟合线
hold on; % keep previous plot visible
plotData(X, y);
plot(X(:,2), X*theta, '-')
legend('Training data', 'Linear regression')
```

### 3.1.2. Gradient　Descent 函数实现
```octave
% cost function 实现
function J = computeCost(X, y, theta)

% number of training examples
m = length(y);
hx= X * theta;

J=1/(2*m)*sum((hx-y).^2);
end

% gradientDescent
function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    % 计算迭代量
    cost=X*theta-y;
    theta=theta-(alpha/m*(X'*cost));

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
end 
end

```

### 3.1.3. 作图画出 cost function 二维变化图表

```octave

% Grid over which we will calculate J
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);

% initialize J_vals to a matrix of 0's
J_vals = zeros(length(theta0_vals), length(theta1_vals));

% Fill out J_vals
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];
	  J_vals(i,j) = computeCost(X, y, t);
    end
end

% Because of the way meshgrids work in the surf command, we need to
% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';
% Surface plot
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1');

% Contour plot
figure;
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);

```

## 3.2. 多特征线性回归 Linear Regression with Multiple Variables

### 3.2.1. Feature Normalization

```
% 读取数据 , 特征有两项
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

[X mu sigma] = featureNormalize(X);

function [X_norm, mu, sigma] = featureNormalize(X)

% 标准化后特征值的 平局值为0  标准差为1
X_norm = X;

% mu 保存各个特征的平局值 行向量
mu = zeros(1, size(X, 2));
% sigma 保存各个特征的标准差
sigma = zeros(1, size(X, 2));

% 标准化处理步骤一 
mu=mean(X);

for i=1:size(X,2),
    X_norm(:,i)=X(:,i)-mu(i);
end;

% 标准化处理步骤二
sigma=std(X);
for i=1:size(X,2),
    X_norm(:,i)=X_norm(:,i)/sigma(i);
end

end



% 标准化后 在使用模型的时候也要传入同样标准后的特征

% 重新获得原始数据的 mean 和 std
X = data(:, 1:2);
mu=mean(X);
sigma=std(X);

% 对传入的数据标准化 
house=[1650 3];
house=house.-mu;
house=house./sigma;

% 加入常数量 计算预测值
price = [1 house]*theta; % You should change this
```

### 3.2.2. Gradient Descent

```octave
% 计算 costFunction 的函数大体一样

function J = computeCostMulti(X, y, theta)
m = length(y); % number of training examples
hx= X * theta;

J=1/(2*m)*sum((hx-y).^2);

end

% 使用 Gradient Descent 计算 theta 以及每次迭代的 cost 

function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)

m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    cost=X*theta-y;
    theta=theta-(alpha/m*(X'*cost));

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end
end
```

### 3.2.3. Normal Equations

一个式子搞定
```
function [theta] = normalEqn(X, y)
  theta = zeros(size(X, 2), 1);
  theta=pinv(X'*X)*X'*y;
end

```


## 3.3. Logistic Regression

### 打印数据图

在进行分类模型前, 将源数据以图的形式打印出来很有帮助

```octave

function plotData(X, y)

% X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% 先分别找到 正负类别
pos=find(y==1);
neg=find(y==0);

% 画图

% k+ 是 + 号标记
plot(X(pos,1),X(pos,2),'k+','LineWidth',2,'MarkerSize',7);W
% o 是圆形标记
plot(X(neg,1),X(neg,2),'ko','MarkerFaceColor','y','MarkerSize',7);

```

### Sigmoid Function

针对logistic 分类训练需要使用该函数
```octave
function g = sigmoid(z)
g = zeros(size(z));


% 注意这里要加两个 '.'  , 不然g的维度会颠倒
g=(1./(1+e.^(-z)));

end
```

### Logistic 的 costFunction

```octave

function [J, grad] = costFunction(theta, X, y)

m = length(y);
J = 0;
grad = zeros(size(theta));


% 这里的代码分步计算了
z=X*theta;
w=(-y)'*log(sigmoid(z)) - (1-y)'*log(1-sigmoid(z));

J=(1/m)*sum(w);
grad=(1/m)*((sigmoid(z)-y)'*X);

end
```

### octave 的自动迭代收敛函数 fminunc

通过该方法, 可以省去自己写收敛函数的过程, 而且参数都可以指定

```octave

%  Set options for fminunc  相关参数设置
options = optimset('GradObj', 'on', 'MaxIter', 400);


[theta, cost] = ...
	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);
% ------------
% fminunc(    @( 要优化的参数 )( 函数以及传入的参数 , 初始参数 , 设置变量 ) )
% t 就是传入的 theta , 最后将最终的值返回给 theta 和 cost

```

### Logistic 的 预测

```octave

function p = predict(theta, X)
m = size(X, 1); 
p = zeros(m, 1);


% 注意这里, 对于预测的 hx 先执行 sigmoid 在归化到 0和1
p=round(sigmoid(X*theta));
end

% 在主函数中 对整个测试数据集进行预测 
% Compute accuracy on our training set
p = predict(theta, X);

fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
fprintf('Expected accuracy (with lambda = 1): 83.1 (approx)\n');
```

### mapFeature 

通过 feature mapping 将二位的特征映射到高维, 来拟合非线性边界

```octave
function out = mapFeature(X1, X2)

% 最高将特征创建到6次方
degree = 6;
out = ones(size(X1(:,1)));
for i = 1:degree
    for j = 0:i
        % 这里用 end+1 来插入新的一列
        out(:, end+1) = (X1.^(i-j)).*(X2.^j);
    end
end

end
```

### regularized logistic regression

计算cost和梯度下降

注意在计算标准化的逻辑回归时, 常数项 theta 0 和剩余的 theta 的梯度计算式子不同 

```octave
function [J, grad] = costFunctionReg(theta, X, y, lambda)
m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

hx=sigmoid(X*theta);
w=(-y).*log(hx) - (1-y).*log(1-hx);

J=(  sum(w)+sum(theta.^2)*(lambda/2) )/m;

% grad=(1/m).*(X'*(hx-y)).+(lambda/m).*theta;
% grad(1)=(X'(1,:)*(hx-y))/m;

grad=(X'*(hx-y))/m;
grad(2:end)=grad(2:end)+(lambda/m).*theta(2:end);

end

```

### 打印训练好的边界

分类模型训练完成后 在数据图上打印边界

```octave



function plotDecisionBoundary(theta, X, y)

% 先打印数据 使用之前定义的函数
plotData(X(:,2:3), y);
hold on

% 如果特征只是一维的 即 1,x ,y
if size(X, 2) <= 3
    % Only need 2 points to define a line, so choose two endpoints
    
    % 定义坐标系的宽度
    plot_x = [min(X(:,2))-2,  max(X(:,2))+2];

    % Calculate the decision boundary line
    % 定义 y轴的高度 = plot_x (x轴的左右顶点带入式子 [ (-1*theta+x*theta2)/theta3 ] )
    plot_y = (-1./theta(3)).*(theta(2).*plot_x + theta(1));

    % Plot, and adjust axes for better viewing  最好看的宽度高度
    plot(plot_x, plot_y)
    
    % Legend, specific for the exercise
    legend('Admitted', 'Not admitted', 'Decision Boundary')
    axis([30, 100, 30, 100])

% 如果特征是多项式的
else
    % Here is the grid range
    u = linspace(-1, 1.5, 50);
    v = linspace(-1, 1.5, 50);

    z = zeros(length(u), length(v));
    % Evaluate z = theta*x over the grid
    for i = 1:length(u)
        for j = 1:length(v)
            z(i,j) = mapFeature(u(i), v(j))*theta;
        end
    end
    z = z'; % important to transpose z before calling contour

    % Plot z = 0
    % Notice you need to specify the range [0, 0]
    % 使用 countor 来根据 z 带入训练模型后得出的值来画出边界
    contour(u, v, z, [0, 0], 'LineWidth', 2)
end
hold off

end

```

## Multi-class Classification and Neural Networks

### 对于原生矩阵数据的读取

```octave
load('ex3data1.mat'); % training data stored in arrays X, y
```
对于二进制数据, 只能通过`load`命令直接读入内存, 而不能使用ASCII方法来查看或者修改  
读取后数据变量自动存储在内存, 不需要再分配变量  


### 将矩阵数据以图表现出来

```octave
function [h, display_array] = displayData(X, example_width)

% Set example_width automatically if not passed in
if ~exist('example_width', 'var') || isempty(example_width) 
	example_width = round(sqrt(size(X, 2)));
end


% Gray Image
colormap(gray);

% Compute rows, cols
[m n] = size(X);
example_height = (n / example_width);


% Compute number of items to display
display_rows = floor(sqrt(m));
display_cols = ceil(m / display_rows);

% Between images padding
% 各个样例之间的空白
pad = 1;

% Setup blank display  
% 用数列来存储最终的图像数据
display_array = - ones(pad + display_rows * (example_height + pad), ...
                       pad + display_cols * (example_width + pad));

% 将数据拷入显示内存
% Copy each example into a patch on the display array
curr_ex = 1;
for j = 1:display_rows
	for i = 1:display_cols
    % 如果定义的画板太大 数据不够 则退出循环
		if curr_ex > m, 
			break; 
		end
		% Copy the patch
		
		% Get the max value of the patch
    % 获取灰度图的最大值
		max_val = max(abs(X(curr_ex, :)));

    % 将图像数据拷贝到最终整体的数列对应位置的部分
		display_array(pad + (j - 1) * (example_height + pad) + (1:example_height), ...
		              pad + (i - 1) * (example_width + pad) + (1:example_width)) = ...
						reshape(X(curr_ex, :), example_height, example_width) / max_val;
		curr_ex = curr_ex + 1;
	end
	if curr_ex > m, 
		break; 
	end
end

% Display Image
% 将图像显示出来  值的区间设置成 [-1 1]
h = imagesc(display_array, [-1 1]);

% Do not show axis
% 不打印横纵坐标
axis image off

drawnow;

end


```