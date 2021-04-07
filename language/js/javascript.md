# 1. javascript 介绍

JavaScript 是 Web 的编程语言。  
所有现代的 HTML 页面都使用 JavaScript。  

web 开发语言:
1. HTML 定义网页的内容
2. CSS 定义网页的布局
3. JS 实现网络的行为

JS的特点:
1. 脚本语言, 轻量级
2. 可以插入 HTML 页面, 并由所有现代浏览器执行

## 1.1. 版本时间
* 2009 	ECMAScript 5	添加 "strict mode"，严格模式, 添加 JSON 支持
* 2015 	ECMAScript 6	添加类和模块
* 2016 	ECMAScript 7	增加指数运算符 (**)增加 Array.prototype.includes


## 1.2. HTML 事件

事件是发生在 HTML 元素上的事情, 可以触发 javascript 脚本  

* `<button onclick="displayDate()">现在的时间是?</button>`  
* `<button onclick="this.innerHTML=Date()">现在的时间是?</button>`




## 1.3. 简单示例

1. 直接写入到 HTML 输出流
* `document.write("<p>这是一个段落。</p>");`
* 只能在 `HTML 输出中使用 document.write` 
* 如果在文档加载后使用该方法 (比如在函数中), 会覆盖整个文档


2. 对事件进行反应
* `<button type="button" onclick="alert('欢迎!')">点我!</button>`
* 在HTML标签中嵌入 js 函数


3. 动态页面: 改变HTML的内容
* `x=document.getElementById("demo");  //查找元素`
* `x.innerHTML="Hello JavaScript";    //改变内容`
* 通过修改属性来改变 HTML 页面的内容
* `element=document.getElementById('myimage') // 查找元素`
* `element.src="/images/pic_bulbon.gif";      // 更改图片`


## 1.4. 使用方法

内嵌在 HTML 中:
1. HTML 中的脚本必须位于 `<script>  </script>` 标签之间
2. 允许被放置在 HTML 页面的 `<body>  <head>` 部分中
   * 通常函数放在 `<head>` 中, 或者 `<body>` 的最底部


外部 js 文件:
1. 文件扩展名 `.js`
2. 在 HTML 页面文件中的 `<script>` 标签的 "src" 属性中设置该 .js 文件
   * `<body><script src="myScript.js"></script></body>`


## 1.5. 基础输出方法

1. `window.alert()` 弹出警告框
2. `document.write()` 方法将内容写到 HTML 文档中
3. `element.innerHTML` 写入到 HTML 元素
4. `console.log()` 写入到浏览器的控制台


# 2. javascript 语言

## 2.1. 保留字总结

* var       : 定义变量
* function  : 定义函数
* return    : 函数返回值

## 2.2. 语法

* js 的代码可以使用分号分割 `;`   (非必须)
* 字符串中, 反斜杠 `\` 可以用来进行代码跨行
* 注释是双斜杠 `//` , 多行注释 `/**/`
* 大小写明


### 2.2.1. 数据类型

1. 值类型(基本类型)：
   * 字符串 (String)
   * 数字(Number)   js 不区分整数和小数
   * 布尔(Boolean)
   * 对空 (Null)
   * 未定义 (Undefined)
     * 注意 `Undefined` 代表没有值
     * 和 `null` 代表空值是不一样的
   * Symbol
2. 引用数据类型：
   * 对象(Object)
   * 数组(Array)
   * 函数(Function)


### 2.2.2. 字面量与变量

字面量即固定值:
1. 数字, 可以是整数或者小数, 科学计数(e)
2. 字符串, 单引号和双引号都可以用
3. 字面量的表达式
4. 数组, js数组是 0下标
5. 对象
6. 函数

与字面量相对的就是变量:
* 变量名称遵循 c语言规则
* 未赋值的变量的值统一是 `undefined`
* 重新声明一个变量不会改变原有的值
* var 变量
  * 使用 `var 标识符` 来定义变量, 不必须指定数据类型 (变量拥有动态类型)
  * 使用 `new` 可以执行变量的类型 `var carname=new String;`
* 无 var 变量 `carname="Volvo";`
  * 会自动作为 window 的一个变量属性 (全局性) 不论是不是在函数中
    * 在任何地方通过 `window.carName` 访问
  * 可以被 `delete` 手动删除
  * var变量不能被 `delete` 删除


### 2.2.3. 操作符

* `=` 赋值运算符     : 可以用来给变量赋值
* `+-*/` 数学运算符  : 算数运算
* `== != <>`        : 标准比较运算符


### 2.2.4. 对象

* 对象由花括号定义
* 对象的属性以名称和值对的形式 (name : value) 来定义
  * 对象属性的访问 `person.lastName;  person["lastName"]; `
* 对象的方法也算一个属性
  * 通过添加 `()` 来调用

* 变量: `var car = "Fiat"; `
* 对象: `var car = {type:"Fiat", model:500, color:"white"}; `

### 2.2.5. 函数

* 函数的局部变量生命周期与C语言相同
```js
function functionname(argument1,argument2)
{
    // 执行代码
    return value;
}
```

# 3. 内置对象

## 浏览器对象

### window

* Window 对象表示浏览器中打开的窗口
* 虽然没有应用于 window 对象的公开标准, 但是所有浏览器都支持该对象



# 4. js异步编程

* async  Asynchronous
* sync   Synchronous

## 4.1. promise

* Promise 是 ECMAScript 6 加入的`类`
* 可以优雅的书写复杂的异步任务

```js
new Promise(
   function (resolve, reject) {
    console.log("Run");
   }
);
```

使用方法:
1. 在构造后会立刻异步运行
2. promise 构造函数只有一个参数, 就是一个函数(起始函数)
3. 起始函数需要包含两个参数 `resolve, reject`, 这两个参数都是函数
4. 调用 resolve 代表一切正常
5. 调用 reject  代表出现异常


## 4.2. 导入模块

使用 `require` 来导入一个模块, 并将实例化的模块赋值给一个变量  
`var http = require("http");`

## 4.3. 

http.createServer() 方法可以创建服务器  

```js
var http = require('http');

http.createServer(function (request, response) {

    // 发送 HTTP 头部 
    // HTTP 状态值: 200 : OK
    // 内容类型: text/plain
    response.writeHead(200, {'Content-Type': 'text/plain'});

    // 发送响应数据 "Hello World"
    response.end('Hello World\n');
}).listen(8888);

// 应用会被自动绑到 localhost 对应端口
console.log('Server running at http://127.0.0.1:8888/');
```



