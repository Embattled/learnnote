# 1. SQL 语言

对数据库进行查询和修改操作的语言叫做 SQL（Structured Query Language) 结构化查询语言
* SQL 是目前广泛使用的 `关系数据库` 的标准语言, 也是数据库脚本文件的拓展名
* SQL 是一种数据库查询和程序设计语言, 由很少的关键字组成
* SQL的特点:
  * 一体化: 集数据定义, 数据操作, 数据控制为一体, 可以完成全部数据库工作
  * 灵活  : 可以以命令形式执行, 也可以嵌入其他编程语言中使用
  * 非过程化: 具体的操作是由 数据库管理系统 DBMS 来完成
  * 语言简单
* SQL 语言是完整不区分大小写的语言
  * 为了便于阅读, 通常情况下: 
  * 开发人员会对 SQL关键字进行大写
  * 对 表或者列名使用小写
* 除了标准SQL语言, 不同的数据库一般都有自己的特性SQL语句
  * Microsoft SQL Server : T-SQL
  * Oracle  : PL/SQL
  * MySQL   : MySQL


标准SQL发展的简要历史：
* 1986年，ANSI X3.135-1986，ISO/IEC 9075:1986，SQL-86
* 1989年，ANSI X3.135-1989，ISO/IEC 9075:1989，SQL-89
* 1992年，ANSI X3.135-1992，ISO/IEC 9075:1992，SQL-92（SQL2）
* 1999年，ISO/IEC 9075:1999，SQL:1999（SQL3）
* 2003年，ISO/IEC 9075:2003，SQL:2003
* 2008年，ISO/IEC 9075:2008，SQL:2008
* 2011年，ISO/IEC 9075:2011，SQL:2011


SQL语言具体可以分成4部分
1. 数据定义语言 Data Definition Language, DDL
2. 数据操作语言 Data Manipulation Language, DML
3. 数据查询语言 Data Query Language, DQL
4. 数据控制语言 Data Control Language DCL




## 1.1. SQL语法

* SQL语句以 `;` 结尾, 换行不会中断语句
* SQL语句不区分大写, 包括表名, 列名 (当然具体的表项数据是区分的)
* 常数的书写方法 (字符串, 日期, 数字)
  * 字符串用单引号括起来 `'abc'`
  * 日期用单引号括起来, 同时需要满足任意一种日期格式, eg `'2020-01-26'`
  * 数字直接书写即可





# 2. Data Query Language, DQL

用来查询表中的记录, 主要包含 SELECT 命令, 来查询表中的数据. 

* 主关键字
  * SELECT
  * JOIN
* 前加关键字
  * DISTINCT, 返回唯一不同的值
* 后加关键字
  * AS , 即 Alias 添加别名, 可以作用于 列名或者表名
    * 添加列的 Alias 可以使结果表中显示的列名易读
    * 添加表的 Alias 可以使得其他子句中的逻辑式容易书写
  * WHERE子句, 用于附加选择数据的条件
    * AND OR 运算符用于结合多个 WHERE 子句, 形成复杂的逻辑判断, 搭配括号使用
  * ORDER BY, 用于对结果进行排序
  * GROUP BY, 对一个或者多个列的结果集进行分组
* 非标准子句
  * TOP 字句

## 2.1. TOP 字句
* TOP (非标准), 用于指定返回的记录的数目, 用于大型数据库
* 不同的数据库对于该功能有不同的语法

* SQL Server 语法： TOP
  * `TOP 数字` 数字为行数
  * `TOP 数字 PERCENT` 数字为百分比   
  * `SELECT TOP number|percent`
* MySQL 语法：LIMIT 
  * `LIMIT number` 写在最后
  * `SELECT column_name(s) FROM table_name LIMIT number`
* Oracle 语法: ROWNUM关键字
  * `WHERE ROWNUM <= number`


## 2.2. WHERE

* `WHERE 列 运算符 值  [AND/OR 列 运算符 值 ...]`  
* 用于筛选只满足条件的数据项


运算符表:
| 操作符            | 描述                                 |
| ----------------- | ------------------------------------ |
| `=`               | 等于                                 |
| `<>`              | 不等于                               |
| `>`               | 大于                                 |
| `<`               | 小于                                 |
| `>=`              | 大于等于                             |
| `<=`              | 小于等于                             |
| `BETWEEN A AND B` | 在某个范围内, 用于数字, 日期, 字符串 |
| `IN (a,b,c..)`    | 在某些候补项中                       |
| `LIKE`            | 搜索某种模式                         |
* 注意, 后三种字符关键字可以前接 `NOT`

* `LIKE/NOT LIKE` 通配搜索
通配符列表, SQL中通配符只用在 LIKE 语句中  
| 通配符                       | 描述                       |
| ---------------------------- | -------------------------- |
| %                            | 代表零个或多个字符         |
| _                            | 仅替代一个字符             |
| `[charlist]`                 | 字符列中的任何单一字符     |
| `[^charlist]`  `[!charlist]` | 不在字符列中的任何单一字符 |


## 2.3. ORDER BY

* `ORDERBY 列 顺序[, 列 顺序...]`
* 顺序可以省略, 默认是升序 `ASC` , 可以指定成降序 `DESC`
* 可以添加多个排列对象, 用逗号分割, 按照顺序决定优先级
* `ORDER BY Company DESC, OrderNumber ASC` 

## 2.4. SELECT

从表中选取数据, 结果存储在一个 结果表(结果集)  
* `SELECT [DISTINCT] 列名/* FROM 表名称 [WHERE 子句] [ORDER BY 子句]`  
* 多个列名用逗号分隔 `LastName,FirstName`


## 2.5. JOIN

JOIN 运算是根据多个表中的关系来输出结果
* 主键 (Primary Key), 在每个行都是唯一的值
* 使用 WHERE 可以达到相同的效果
* 语法 `JOIN 表a,表b ON a.id=b.id`
* JOIN 的区分
  * (INNER) JOIN : 只有匹配上的行才会被返回
  * LEFT JOIN: 即使右表中没有匹配的行, 也返回左表的所有行
  * RIGHT JOIN: 即使左表没有匹配的行, 也会返回右表的所有行
  * FULL JOIN: 没有匹配的行都会被列出
```sql
SELECT Persons.LastName, Persons.FirstName, Orders.OrderNo
FROM Persons, Orders
WHERE Persons.Id_P = Orders.Id_P 

-- INNER JOIN
SELECT column_name(s)
FROM table_name1
INNER JOIN table_name2 
ON table_name1.column_name=table_name2.column_name
```

## 2.6. SQL 查询函数



# 3. Data Manipulation Language, DML

用来变更表中的记录, 主要包含以下几种命令: 

* INSERT: 向表中插入新数据
* UPDATE: 更新表中的数据
* DELETE: 删除表中的数据

## 3.1. INSERT

* 插入新的表项(行)
* `INSERT INTO 表名 [列1, 列2,...] VALUES (值1, 值2,....)` 

## 3.2. UPDATE

* 用于修改表中的数据
* `UPDATE 表名称 SET 列1 = 值1[, 列2 = 值2] [WHERE子句指定行]`


## 3.3. DELETE

* 用于删除表中的行
* `DELETE FROM 表名称 WHERE子句`
* `DELETE * FROM 表名称` 删除整个表的数据, 但是保留表的存在和结构

# 4. Data Definition Language, DDL

用来创建或删除数据库以及表等对象, 主要包含以下几种命令: 
* DROP  : 删除数据库和表等对象
* CREATE: 创建数据库和表等对象
* ALTER : 修改数据库和表等对象的结构


    CREATE DATABASE - 创建新数据库
    ALTER DATABASE - 修改数据库
    CREATE TABLE - 创建新表
    ALTER TABLE - 变更（改变）数据库表
    DROP TABLE - 删除表
    CREATE INDEX - 创建索引（搜索键）
    DROP INDEX - 删除索引





# 5. Data Control Language, DCL

用来确认或者取消对数据库中的数据进行的变更.   
除此之外, 还可以对数据库中的用户设定权限.  主要包含以下几种命令: 
* GRANT：赋予用户操作权限
* REVOKE：取消用户的操作权限
* COMMIT：确认对数据库中的数据进行的变更
* ROLLBACK：取消对数据库中的数据进行的变更

