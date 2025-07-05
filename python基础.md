# python基础

2024/10/14

#### 优点

Python为我们提供了非常完善的基础代码库，覆盖了网络、文件、GUI、数据库、文本等大量内容，被形象地称作“内置电池（batteries included）”。用Python开发，许多功能不必从零编写，直接使用现成的即可。

Python的定位是“优雅”、“明确”、“简单”

#### 缺点

第一个缺点就是运行速度慢

第二个缺点就是代码不能加密（凡是编译型的语言，都没有这个问题，而解释型的语言，则必须把源码发布出去。）

## python解释器安装

https://www.python.org/downloads/

![{E97C8BAE-7138-4B64-B42F-0B49E5C67D33}.png](https://github.com/Davereminisce/image/blob/master/%7BE97C8BAE-7138-4B64-B42F-0B49E5C67D33%7D.png?raw=true)

## python使用

#### 命令行模式和Python交互模式

直接输入`python`进入交互模式，相当于启动了Python解释器，但是等待你一行一行地输入源代码，每输入一行就执行一行。

直接运行`.py`文件相当于启动了Python解释器，然后一次性把`.py`文件的源代码给执行了，你是没有机会以交互的方式输入源代码的。（执行`.py`需要在所在的目录下，才能正常执行）

#### 文本编辑器写Python程序

用文本编辑器写Python程序，然后保存为后缀为`.py`的文件，就可以用Python直接运行这个程序了。

#### 输出

用`print()`在括号中加上字符串,遇到逗号“,”会输出一个空格

```python
print('hello, world')
print('The quick brown fox', 'jumps over', 'the lazy dog')
print(name)
```

#### 输入

`input()`，可以让用户输入字符串，并存放到一个变量里。

`input()`返回的数据类型是`str`(需要强制转换)

```python
name = input('please enter your name: ')
```

## python基础

#### 注释

以`#`开头的语句是注释，其他每一行都是一个语句，当语句以冒号`:`结尾时，缩进的语句视为代码块。

```python
# print absolute value of an integer:
a = 100
```

Python程序是*大小写敏感*

#### 缩进

Python使用缩进来组织代码块，请务必遵守约定俗成的习惯，坚持使用4个空格的缩进。

#### 数据类型（与c相近）

整数

浮点数

字符串：字符串是以单引号`'`或双引号`"`括起来的任意文本，比如`'abc'`，`"xyz"`等等

布尔值：可以用`and`、`or`和`not`运算。

空值：空值是Python里一个特殊的值，用`None`表示。`None`不能理解为`0`，因为`0`是有意义的，而`None`是一个特殊的空值。

#### 变量（动态语言）

变量本身类型不固定的语言称之为***动态语言***

**在Python中，等号`=`是赋值语句，可以把任意数据类型赋值给变量，同一个变量可以反复赋值，而且可以是不同类型的变量

```python
a = 123 # a是整数
print(a)
a = 'ABC' # a变为字符串
print(a)
```

#### 常量

在Python中，通常用全部大写的变量名表示常量

#### 两种除法

在Python中，有两种除法，一种除法是`/`：

```plain
>>> 10 / 3
3.3333333333333335
```

`/`除法计算结果是浮点数，即使是两个整数恰好整除，结果也是浮点数：

```plain
>>> 9 / 3
3.0
```

还有一种除法是`//`，称为地板除，两个整数的除法仍然是整数：

```plain
>>> 10 // 3
3
>>> 10.0 // 3
3.0
>>> 9.0 // 2
4.0
>>> 9.000 // 2.0
4.0
```

#### 字符串和编码

`ord()`函数获取字符的整数表示

`chr()`函数把编码转换为对应的字符

```
>>> ord('中')
20013
>>> chr(66)
'B'
```

`len()`函数：计算`str`包含多少个字符

#### 格式化

%

在Python中，采用的格式化方式和C语言是一致的，用`%`实现

```python
print('%2d-%02d' % (3, 1))
print('%.2f' % 3.1415926)
print('growth rate: %d %%' % 7)
```

format()

```plain
>>> 'Hello, {0}, 成绩提升了 {1:.1f}%'.format('小明', 17.125)
'Hello, 小明, 成绩提升了 17.1%'
```

f-string

```plain
>>> r = 2.5
>>> s = 3.14 * r ** 2
>>> print(f'The area of a circle with radius {r} is {s:.2f}')
The area of a circle with radius 2.5 is 19.62
```

### 使用list和tuple

#### list（类似c数组）

Python内置的一种数据类型是列表：list。类似c数组但list也是**动态**。

```plain
>>> classmates = ['Michael', 'Bob', 123]
>>> classmates
['Michael', 'Bob', 123]
>>> classmates[0]
'Michael'
>>> classmates[1]
'Bob'
>>> classmates[-1] #用-1做索引，直接获取最后一个元素
123
```

`len()`函数：获得list元素的个数

```plain
>>> len(classmates)
3
```

`pop()`方法：删除list末尾的元素

`pop(i)`方法：删除指定位置的元素

#### tuple

另一种有序列表叫元组：tuple。和list非常类似，但是tuple一旦初始化就不能修改。

```plain
>>> classmates = ('Michael', 'Bob', 'Tracy')
```

定义一个空的tuple，可以写成`()`：

```plain
>>> t = ()
>>> t
()
```

定义一个只有1个元素的tuple：

```plain
>>> t = (1)
>>> t
1    #这里定义的是数1因为括号()既可以表示tuple，又可以表示数学公式中的小括号
>>> t = (1,)
>>> t
(1,)  #只有1个元素的tuple定义时必须加一个逗号，
```

“可变的”tuple：

```plain
>>> t = ('a', 'b', ['A', 'B'])
>>> t[2][0] = 'X'
>>> t[2][1] = 'Y'
>>> t
('a', 'b', ['X', 'Y'])  #变的不是tuple的元素，而是list的元素
```

### 条件判断

`if`语句

```python
if <条件判断1>:		#注意不要少写了冒号:
    <执行1>
elif <条件判断2>:	#elif是else if的缩写
    <执行2>
elif <条件判断3>:
    <执行3>
else:
    <执行4>
```

### 模式匹配

`match`语句（类似c的switch）

```python
age = 15
match age:
    case x if x < 10:
        print(f'< 10 years old: {x}')		# x=15已经被赋值
    case 10:
        print('10 years old.')
    case 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18:
        print('11~18 years old.')
    case 19:
        print('19 years old.')
    case _:									# _表示匹配到其他任何情况
        print('not sure.')
```

```python
args = ['gcc', 'hello.c', 'world.c']
# args = ['clean']
# args = ['gcc']

match args:
    # 如果仅出现gcc，报错:
    case ['gcc']:
        print('gcc: missing source file(s).')
    # 出现gcc，且至少指定了一个文件:
    case ['gcc', file1, *files]:
        print('gcc compile: ' + file1 + ', ' + ', '.join(files))
    # 仅出现clean:
    case ['clean']:
        print('clean')
    case _:
        print('invalid command.')
```

第一个`case ['gcc']`表示列表仅有`'gcc'`一个字符串，没有指定文件名，报错；

第二个`case ['gcc', file1, *files]`表示列表第一个字符串是`'gcc'`，第二个字符串绑定到变量`file1`，后面的任意个字符串绑定到`*files`（符号`*`的作用将在[函数的参数](https://liaoxuefeng.com/books/python/function/parameter/index.html)中讲解），它实际上表示至少指定一个文件；

第三个`case ['clean']`表示列表仅有`'clean'`一个字符串；

最后一个`case _`表示其他所有情况。

### 循环

#### `for in`循环

依次把list或tuple中的每个元素迭代出来

```python
names = ['Michael', 'Bob', 'Tracy']
for name in names:
    print(name)
```

```python
sum = 0
for x in range(101):
    sum = sum + x
print(sum)
```

##### `range()`函数

可以生成一个整数序列。`range()` 函数在 Python 中生成的对象既不是 `tuple` 也不是 `list`，而是一个 `range` 对象。这是一个惰性序列（lazy sequence），它不会直接生成所有的值，而是按需计算并在迭代时逐步生成值。

```python
r = range(5)
print(r)  # 输出: range(0, 5)

# 可以将 range 对象转换为 list 或 tuple
print(list(r))  # 输出: [0, 1, 2, 3, 4]
print(tuple(r))  # 输出: (0, 1, 2, 3, 4)
```

#### `while`循环

```python
sum = 0
n = 99
while n > 0:
    sum = sum + n
    n = n - 2
print(sum)
```

#### break

`break`的作用是提前结束循环。

#### continue

`continue`语句，跳过当前的这次循环，直接开始下一次循环。

### 使用dict和set

#### dict

`dict`是一种字典数据结构，它是键-值对的集合。字典中的键是唯一的，而值可以是任何类型的对象（包括数字、字符串、列表、甚至另一个字典）。字典的键必须是不可变的对象，比如字符串、数字或元组。

可以通过几种方式创建字典：

1. 使用花括号 `{}`

   ```python
   my_dict = {"name": "Alice", "age": 25, "city": "New York"}
   ```

2. 使用 `dict()` 函数

   ```python
   my_dict = dict(name="Alice", age=25, city="New York")
   ```

3. 通过键值对的列表创建

   ```python
   my_dict = dict([("name", "Alice"), ("age", 25), ("city", "New York")]
   ```

```python
#访问字典元素
my_dict = {"name": "Alice", "age": 25}
print(my_dict["name"])  			# 输出: Alice
#添加和修改元素
my_dict["city"] = "New York"  		# 添加新的键值对
my_dict["age"] = 26           		# 修改现有键的值
#删除元素
del my_dict["age"]  				# 删除键 "age"
my_dict.pop("city") 				# 删除并返回键 "city" 对应的值
#遍历字典
for key in my_dict:					#遍历键
    print(key)
for value in my_dict.values():		#遍历值
    print(value)
for key, value in my_dict.items():	#遍历键值对
    print(f"{key}: {value}")
#字典方法
my_dict.keys() 		# dict_keys(['name', 'age'])返回所有的键
my_dict.values() 	# dict_values(['Alice', 25])返回所有的值
my_dict.items()  	# dict_items([('name', 'Alice'), ('age', 25)]
					#返回所有的键值对
```

#### set

`set` 是一种 无序、不重复的集合数据类型。它类似于数学中的集合，主要用于存储唯一的元素。`set` 不允许重复项，并且元素的顺序是不确定的。

可以通过以下几种方式创建集合：

1. 使用花括号 `{}`

   ```python
   my_set = {1, 2, 3, 4}
   ```

2. 使用 `set()` 函数

   ```python
   my_set = set([1, 2, 3, 4])
   ```

元素唯一性：`set` 中的每个元素都是唯一的，重复的元素会被自动去重。

无序性：`set` 中的元素是无序的，**不能通过索引访问**。

```python
#添加元素
my_set.add(5)
#删除元素
my_set.remove(3)	# 如果元素不存在，会抛出 KeyError。
my_set.discard(3)	# 如果元素不存在，不会抛出异常。
#清空集合
my_set.clear()
#检查元素是否存在
print(2 in my_set)  # 输出: True
#集合操作
set1 = {1, 2, 3}
set2 = {3, 4, 5}
print(set1 | set2)  # 并集 (union 或 |)输出: {1, 2, 3, 4, 5}
print(set1 & set2)  # 交集 (intersection 或 &)输出: {3}
print(set1 - set2)  # 差集 (difference 或 -)输出: {1, 2}
print(set1 ^ set2)  # 对称差集 (symmetric_difference或^)输出: {1, 2, 4, 5}
```

## 函数

### 调用函数

Python内置了很多有用的函数，我们可以直接调用。

要调用一个函数，需要知道函数的名称和参数，比如求绝对值的函数`abs`，只有一个参数。

查看函数

1.从Python的官方网站查看[文档](http://docs.python.org/3/library/functions.html#abs)

2.在交互式命令行通过`help(abs)`查看`abs`函数的帮助信息。

#### 数据类型转换

Python内置的常用函数还包括数据类型转换函数，即数据类型转换是函数。

### 定义函数

#### 自定义函数

定义一个函数要使用`def`语句，依次写出函数名、括号、括号中的参数和冒号`:`，然后，在缩进块中编写函数体，函数的返回值用`return`语句返回。

```python
def my_abs(x):			#不需要显式定义函数参数的类型。
    if x >= 0:
        return x
    else:
        return -x

print(my_abs(-99))  
```

如果想为参数和返回值提供类型提示，可以使用类型注解：

```python
def add(x: int, y: int) -> int:
    return x + y
```

`x: int` 和 `y: int` 表示 `x` 和 `y` 期望是 `int` 类型的参数。

`-> int` 表示函数的返回值应当是 `int` 类型。

没有`return`语句，函数执行完毕后也会返回结果，只是结果为`None`。`return None`可以简写为`return`。

#### 空函数（pass用法）

定义一个什么事也不做的空函数，可以用`pass`语句：

```python
def nop():
    pass
```

实际上`pass`可以用来作为占位符，比如现在还没想好怎么写函数的代码，就可以先放一个`pass`，让代码能运行起来。

`pass`还可以用在其他语句里，缺少了`pass`，代码运行就会有语法错误。

#### 返回多个值

```python
import math

def move(x, y, step, angle=0):
    nx = x + step * math.cos(angle)
    ny = y - step * math.sin(angle)
    return nx, ny

x, y = move(100, 100, 60, math.pi / 6)
print(x, y)			#151.96152422706632 70.0
r = move(100, 100, 60, math.pi / 6)
print(r)			#(151.96152422706632, 70.0)
```

返回值是一个tuple

返回一个tuple可以省略括号，而多个变量可以同时接收一个tuple，按位置赋给对应的值，所以，Python的函数返回多值其实就是返回一个tuple

### 函数的参数

#### 位置参数

如c

#### 默认参数

当函数有多个参数时，把变化大的参数放前面，变化小的参数放后面。变化小的参数就可以作为默认参数。

```python
def power(x, n=2):			#n=2为默认参数
    s = 1
    while n > 0:
        n = n - 1
        s = s * x
    return s
```

```plain
>>> power(5)
25
>>> power(5, 2)
25
>>> power(5, 3)
125
```

#### 可变参数

在Python函数中，还可以定义可变参数。顾名思义，可变参数就是传入的参数个数是可变的，可以是1个、2个到任意个，还可以是0个。

Python 提供了两种方式来定义可变参数的函数：

1. `\*args`：用于接收任意数量的位置参数。

2. `\**kwargs`：用于接收任意数量的关键字参数。

   `*args` 和 `**kwargs` 只是约定俗成的名字，你完全可以使用其他名字来替代它们。关键是符号 `*` 和 `**`，它们分别用于收集位置参数和关键字参数。

##### 1.`*args`（可变位置参数）

使用 `*args` 可以让函数接受任意数量的位置参数。它将这些参数收集到一个元组中。

```python
def my_function(*args):
    for arg in args:
        print(arg)
```

```python
my_function(1, 2, 3)  
# 输出:
# 1
# 2
# 3

#允许你在list或tuple前面加一个*号，把list或tuple的元素变成可变参数传进去
nums = [1, 2, 3]
my_function(*nums)	#*nums表示把nums这个list的所有元素作为可变参数传进去。
```

这里，`args` 是一个元组，包含了传递的所有参数。

##### 2.`**kwargs`（可变关键字参数）

使用 `**kwargs` 可以让函数接受任意数量的关键字参数。它将这些参数收集到一个字典中。

```python
def my_function(**kwargs):
    for key, value in kwargs.items():
        print(f"{key} = {value}")
```

```python
my_function(name="Alice", age=25, city="New York")
# 输出:
# name = Alice
# age = 25
# city = New York
```

这里，`kwargs` 是一个字典，包含了所有的关键字参数。

##### 3.同时使用 `*args` 和 `**kwargs`

可以同时使用 `*args` 和 `**kwargs`，这样函数可以同时接受任意数量的位置参数和关键字参数。

```python
def my_function(*args, **kwargs):
    print("Positional arguments:", args)
    print("Keyword arguments:", kwargs)
```

```python
my_function(1, 2, 3, name="Alice", age=25)
# 输出:
# Positional arguments: (1, 2, 3)
# Keyword arguments: {'name': 'Alice', 'age': 25}
```

可以用简化的写法把dict写入：

```plain
>>> extra = {'city': 'Beijing', 'job': 'Engineer'}
>>> person('Jack', 24, **extra)
name: Jack age: 24 other: {'city': 'Beijing', 'job': 'Engineer'}
```

`**extra`表示把`extra`这个dict的所有key-value用关键字参数传入到函数的`**kw`参数，`kw`将获得一个dict，注意`kw`获得的dict是`extra`的一份拷贝，对`kw`的改动不会影响到函数外的`extra`。

##### 4.使用默认参数与可变参数

可以将默认参数与可变参数结合使用，但需要注意顺序：默认参数必须放在 `\*args` 和 `\**kwargs` 之前。

```
def my_function(a, b=10, *args, **kwargs):
    print("a:", a)
    print("b:", b)
    print("args:", args)
    print("kwargs:", kwargs)
```

```
my_function(5, 20, 1, 2, 3, name="Alice", age=25)
# 输出:
# a: 5
# b: 20
# args: (1, 2, 3)
# kwargs: {'name': 'Alice', 'age': 25}
```

### 递归函数

同C

## 高级特性

### 切片

```plain
>>> L[0:3]
['Michael', 'Sarah', 'Tracy']
```

`L[0:3]`表示，从索引`0`开始取，直到索引`3`为止，但不包括索引`3`。即索引`0`，`1`，`2`，正好是3个元素。

如果第一个索引是`0`，还可以省略：L[:3]

同样支持倒数切片（倒数第一个元素的索引是`-1`）后10个数：L[-10:]

所有数，每5个取一个：L[::5]

`[:]`原样复制一个list：L[:]

tuple也是一种list，唯一区别是tuple不可变。因此，tuple也可以用切片操作，只是操作的结果仍是tuple：

```plain
>>> (0, 1, 2, 3, 4, 5)[:3]
(0, 1, 2)
```

字符串`'xxx'`也可以看成是一种list，每个元素就是一个字符。因此，字符串也可以用切片操作，只是操作结果仍是字符串：

```plain
>>> 'ABCDEFG'[:3]
'ABC'
>>> 'ABCDEFG'[::2]
'ACEG'
```

### 迭代

如果给定一个`list`或`tuple`，我们可以通过`for`循环来遍历这个`list`或`tuple`，这种遍历我们称为迭代（Iteration）。（即c语言中的遍历）

### 列表生成式

**列表生成式**（List Comprehension）是 Python 中一种简洁、优雅的创建列表的方式。它可以从一个已有的可迭代对象（如列表、元组、字符串等）中通过表达式快速生成新的列表。

##### 基本语法

```python
[expression for item in iterable]
```

expression：生成列表元素的表达式，可以是对 `item` 的某种操作。

for item in iterable：遍历一个可迭代对象中的每个 `item`。

**基本用法**：生成一个平方的列表

```python
#基本用法
squares = [x**2 for x in range(5)]
print(squares) 				# 输出: [0, 1, 4, 9, 16]
#嵌套循环
cartesian_product = [(x, y) for x in [1, 2, 3] for y in ['a', 'b', 'c']]
print(cartesian_product)  	# 输出: [(1, 'a'), (1, 'b'), (1, 'c'), (2, 'a'), (2, 'b'), (2, 'c'), (3, 'a'), (3, 'b'), (3, 'c')]
#字符串处理
uppercase_letters = [char.upper() for char in "hello"]
print(uppercase_letters)  	# 输出: ['H', 'E', 'L', 'L', 'O']
#使用两个变量来生成list
d = {'x': 'A', 'y': 'B', 'z': 'C' }
list_1=[k + '=' + v for k, v in d.items()]
print(list_1)				#输出: ['y=B', 'x=A', 'z=C']

#加上条件判断if		跟在for后面的if是一个筛选条件，不能带else
even_squares = [x**2 for x in range(10) if x % 2 == 0]
print(even_squares)  		# 输出: [0, 4, 16, 36, 64]
#加上条件判断if-else	for前面的表达式x if x % 2 == 0 else -x才能根据x计算出确定的结果
result = [x if x % 2 == 0 else -x for x in range(10)]
print(result)  # 输出: [0, -1, 2, -3, 4, -5, 6, -7, 8, -9]
```

if情况下，if x % 2 == 0在for后面，筛选出满足的x

if-else情况下，x if x % 2 == 0 else -x在for前面，是决定表达式不是来选x的

### 生成器

在Python中，这种一边循环一边计算的机制，称为生成器：generator。

第一种方法很简单，只要把一个列表生成式的`[]`改成`()`，就创建了一个generator：

```plain
>>> L = [x * x for x in range(10)]
>>> L
[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
>>> g = (x * x for x in range(10))
>>> next(g)
0
>>> next(g)
1
#省略
>>> next(g)
81
>>> next(g)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
StopIteration
```

通过`next()`函数获得generator的下一个返回值

使用`for`循环，因为generator也是可迭代对象，并且不需要关心`StopIteration`的错误。

```plain
>>> g = (x * x for x in range(10))
>>> for n in g:
...     print(n)
```

#### `yield`语句

我认为yield就像断点一样，这次在这里断下，下次从这里开始。

变成generator的函数，在每次调用`next()`的时候执行，遇到`yield`语句返回，再次执行时从上次返回的`yield`语句处继续执行。

```python
def odd():
    print('step 1')
    yield 1
    print('step 2')
    yield(3)
    print('step 3')
    yield(5)
```

```plain
>>> o = odd()
>>> next(o)
step 1
1
>>> next(o)
step 2
3
>>> next(o)
step 3
5
>>> next(o)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
StopIteration
```

### 迭代器

可以直接作用于`for`循环的对象统称为可迭代对象：`Iterable`。

可以使用`isinstance()`判断一个对象是否是`Iterable`对象：

```plain
>>> from collections.abc import Iterable
>>> isinstance([], Iterable)
True
>>> isinstance({}, Iterable)
True
>>> isinstance('abc', Iterable)
True
>>> isinstance((x for x in range(10)), Iterable)
True
>>> isinstance(100, Iterable)
False
```

可以被`next()`函数调用并不断返回下一个值的对象称为迭代器：`Iterator`它们表示一个惰性计算的序列

可以使用`isinstance()`判断一个对象是否是`Iterator`对象：

```plain
>>> from collections.abc import Iterator
>>> isinstance((x for x in range(10)), Iterator)
True
>>> isinstance([], Iterator)
False
>>> isinstance({}, Iterator)
False
>>> isinstance('abc', Iterator)
False
```

集合数据类型如`list`、`dict`、`str`等是`Iterable`但不是`Iterator`，不过可以通过`iter()`函数获得一个`Iterator`对象。

```plain
>>> isinstance(iter([]), Iterator)
True
>>> isinstance(iter('abc'), Iterator)
True
```

## 函数式编程

特点：允许把函数本身作为参数传入另一个函数，还允许返回一个函数。

### 高阶函数 

函数本身也可以赋值给变量，即：变量可以指向函数。

```plain
>>> x = abs(-10)
>>> x
10
>>> f = abs
>>> f
<built-in function abs>
```

变量`f`现在已经指向了`abs`函数本身。直接调用`abs()`函数和调用变量`f()`完全相同。

```plain
>>> f = abs
>>> f(-10)
10
```

函数名是指向函数的变量，对于`abs()`这个函数，完全可以把函数名`abs`看成变量，它指向一个可以计算绝对值的函数

```plain
>>> abs = 10
```

这个变量已经不指向求绝对值函数而是指向一个整数`10`

#### 传入函数

既然变量可以指向函数，函数的参数能接收变量，那么一个函数就可以接收另一个函数作为参数，这种函数就称之为高阶函数。

```python
def add(x, y, f):
    return f(x) + f(y)
```

#### map()

`map()`函数接收两个参数，一个是函数，一个是`Iterable`，`map`将传入的函数依次作用到序列的每个元素，并把结果作为新的`Iterator`返回。

```plain
>>> def f(x):
...     return x * x
...
>>> r = map(f, [1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> list(r)
[1, 4, 9, 16, 25, 36, 49, 64, 81]
```

`map()`传入的第一个参数是`f`，即函数对象本身。由于结果`r`是一个`Iterator`，`Iterator`是惰性序列，因此通过`list()`函数让它把整个序列都计算出来并返回一个list。

```plain
>>> list(map(str, [1, 2, 3, 4, 5, 6, 7, 8, 9]))
['1', '2', '3', '4', '5', '6', '7', '8', '9']
```

#### reduce()

`reduce`把一个函数作用在一个序列`[x1, x2, x3, ...]`上，这个函数必须接收两个参数，`reduce`把结果继续和序列的下一个元素做累积计算，其效果就是：

```python
reduce(f, [x1, x2, x3, x4]) = f(f(f(x1, x2), x3), x4)
```

比方说对一个序列求和，就可以用`reduce`实现：

```plain
>>> from functools import reduce
>>> def add(x, y):
...     return x + y
...
>>> reduce(add, [1, 3, 5, 7, 9])
25
```

#### filter()

`filter()`也接收一个函数和一个序列,`filter()`把传入的函数依次作用于每个元素，然后根据返回值是`True`还是`False`决定保留还是丢弃该元素。

```python
def not_empty(s):
    return s and s.strip()

list(filter(not_empty, ['A', '', 'B', None, 'C', '  ']))
# 结果: ['A', 'B', 'C']
```

注意到`filter()`函数返回的是一个`Iterator`，也就是一个惰性序列，所以要强迫`filter()`完成计算结果，需要用`list()`函数获得所有结果并返回list。

#### sorted()

Python内置的`sorted()`函数就可以对list进行排序：

```plain
>>> sorted([36, 5, -12, 9, -21])
[-21, -12, 5, 9, 36]
```

`sorted()`函数也是一个高阶函数，它还可以接收一个`key`函数来实现自定义的排序，例如按绝对值大小排序：

```plain
>>> sorted([36, 5, -12, 9, -21], key=abs)
[5, 9, -12, -21, 36]
```

### 返回函数

如果不需要立刻求和，而是在后面的代码中，根据需要再计算

不返回求和的结果，而是返回求和的函数：

```python
def lazy_sum(*args):
    def sum():
        ax = 0
        for n in args:
            ax = ax + n
        return ax
    return sum

f = lazy_sum(1, 3, 5, 7, 9)		#返回的并不是求和结果，而是求和函数，即f是一个求和函数
x = f()								#此时才是求和
```

请再注意一点，当我们调用`lazy_sum()`时，每次调用都会返回一个新的函数，即使传入相同的参数：

```plain
>>> f1 = lazy_sum(1, 3, 5, 7, 9)
>>> f2 = lazy_sum(1, 3, 5, 7, 9)
>>> f1==f2
False
```

`f1()`和`f2()`的调用结果互不影响。

#### 闭包

返回的函数并没有立刻执行，而是直到调用了`f()`才执行。

```python
def count():
    fs = []
    for i in range(1, 4):
        def f():
             return i*i
        fs.append(f)
    return fs

f1, f2, f3 = count()
```

```plain
>>> f1()
9
>>> f2()
9
>>> f3()
9
```

全部都是`9`！原因就在于返回的函数引用了变量`i`，但它并非立刻执行。等到3个函数都返回时，它们所引用的变量`i`已经变成了`3`，因此最终结果为`9`。

思考：相当与开始的f1, f2, f3 = count()只是把函数名字（相当于地址）给了f1，f2，f3，最后调用时返回的是变化后的count（）了。

**返回闭包时牢记一点：返回函数不要引用任何循环变量，或者后续会发生变化的变量。**

#### nonlocal

使用闭包，就是内层函数引用了外层函数的局部变量。如果只是读外层变量的值，我们会发现返回的闭包函数调用一切正常：

```python
def inc():
    x = 0
    def fn():
        # 仅读取x的值:
        return x + 1			#此处认为x仍是外层函数局部变量，x值不变
    return fn

f = inc()
print(f()) # 1
print(f()) # 1
```

但是，如果对外层变量赋值，由于Python解释器会把`x`当作函数`fn()`的局部变量，它会报错：

```python
def inc():
    x = 0
    def fn():
        # nonlocal x
        x = x + 1
        return x				#此处认为x是内层函数局部变量，同时x值被改变
    return fn

f = inc()
print(f()) # 1
print(f()) # 2
```

原因是`x`作为局部变量并没有初始化，直接计算`x+1`是不行的。但我们其实是想引用`inc()`函数内部的`x`，所以需要在`fn()`函数内部加一个`nonlocal x`的声明。加上这个声明后，解释器把`fn()`的`x`看作外层函数的局部变量，它已经被初始化了，可以正确计算`x+1`。

### 匿名函数

即没有函数名

以`map()`函数为例，计算f(x)=x2时，除了定义一个`f(x)`的函数外，还可以直接传入匿名函数：

```plain
>>> list(map(lambda x: x * x, [1, 2, 3, 4, 5, 6, 7, 8, 9]))
[1, 4, 9, 16, 25, 36, 49, 64, 81]
```

关键字`lambda`表示匿名函数，冒号前面的`x`表示函数参数。

匿名函数有个限制，就是只能有一个表达式，不用写`return`，返回值就是该表达式的结果。

匿名函数是一个函数对象，也可以把匿名函数赋值给一个变量，再利用变量来调用该函数：

```python
f = lambda x: x * x
n = f(5)
```

也可以把匿名函数作为返回值返回，比如：

```python
def build(x, y):
    return lambda: x * x + y * y
```

### 装饰器

装饰器（Decorator）是 Python 中的一种设计模式，用于在不修改原有函数或类的前提下，增强或扩展其功能。装饰器本质上是一个返回函数的函数，它允许你在函数执行前后添加代码逻辑。

装饰器的定义和使用有两部分：

1. 定义装饰器函数：接受一个函数作为参数，并返回一个新函数。
2. 使用装饰器：通过 `@decorator_name` 语法来应用装饰器。

#### 基本装饰器

```python
def my_decorator(func):
    def wrapper():
        print("Something before the function is called.")
        func()
        print("Something after the function is called.")
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

say_hello()
```

```sql
Something before the function is called.
Hello!
Something after the function is called.
```

在这个例子中：

`my_decorator` 是一个装饰器函数，它接受函数 `func` 作为参数，返回一个包装器函数 `wrapper`。

`say_hello` 函数在运行时被 `my_decorator` 包装起来，因此 `say_hello()` 会先输出装饰器中定义的消息，再执行原函数的内容。

#### 带参数的装饰器

如果被装饰的函数有参数，装饰器中的包装器函数也必须接受这些参数。

带参数的装饰器

```python
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("Function is called with:", args, kwargs)
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def greet(name, greeting="Hello"):
    print(f"{greeting}, {name}!")

greet("Alice")
greet("Bob", greeting="Hi")
```

```sql
Function is called with: ('Alice',) {}
Hello, Alice!
Function is called with: ('Bob',) {'greeting': 'Hi'}
Hi, Bob!
```

### 偏函数

`functools.partial`就是帮助我们创建一个偏函数的，不需要我们自己定义

`functools.partial`的作用就是，把一个函数的某些参数给固定住（只是设置为默认值可以更改），返回一个新的函数，调用这个新函数会更简单。

```plain
>>> import functools
>>> int2 = functools.partial(int, base=2)	#将base默认值设为2
>>> int2('1000000')
64
>>> int2('1010101')
85
>>> int2('1000000', base=10)				#更改base=10
1000000
```

## 模块

模块是一组Python代码的集合，可以使用其他模块，也可以被其他模块使用。（相当与.h文件）

以内建的`sys`模块为例，编写一个`hello`的模块：

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-   
###第1行和第2行是标准注释，第1行注释可以让这个hello.py文件直接在Unix/Linux/Mac上运行，第2行注释表示.py文件本身使用标准UTF-8编码；###

' a test module '

__author__ = 'asd'

import sys

def test():
    args = sys.argv
    if len(args)==1:
        print('Hello, world!')
    elif len(args)==2:
        print('Hello, %s!' % args[1])
    else:
        print('Too many arguments!')

if __name__=='__main__':
    test()
```

使用`sys`模块的第一步：导入该模块

```python
import sys
```

`sys`模块有一个`argv`变量，用list存储了命令行的所有参数。`argv`至少有一个元素，因为第一个参数永远是该.py文件的名称，例如：

运行`python3 hello.py`获得的`sys.argv`就是`['hello.py']`；

运行`python3 hello.py Michael`获得的`sys.argv`就是`['hello.py', 'Michael']`。

注意到这两行代码：

```python
if __name__=='__main__':
    test()
```

当我们在命令行运行`hello`模块文件时，Python解释器把一个特殊变量`__name__`置为`__main__`，而如果在其他地方导入该`hello`模块时，`if`判断将失败，因此，这种`if`测试可以让一个模块通过命令行运行时执行一些额外的代码，最常见的就是运行测试。

我们可以用命令行运行`hello.py`看看效果：

```plain
$ python3 hello.py
Hello, world!
$ python hello.py Michael
Hello, Michael!
```

如果启动Python交互环境，再导入`hello`模块：

```plain
$ python3
Python 3.4.3 (v3.4.3:9b73f1c3e601, Feb 23 2015, 02:52:03) 
[GCC 4.2.1 (Apple Inc. build 5666) (dot 3)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import hello
>>>
```

导入时，没有打印`Hello, word!`，因为没有执行`test()`函数。

调用`hello.test()`时，才能打印出`Hello, word!`：

```plain
>>> hello.test()
Hello, world!
```

### 作用域

正常的函数和变量名是公开的（public），可以被直接引用，比如：`abc`，`x123`，`PI`等；

类似`__xxx__`这样的变量是特殊变量，可以被直接引用，但是有特殊用途，比如上面的`__author__`，`__name__`就是特殊变量，`hello`模块定义的文档注释也可以用特殊变量`__doc__`访问，我们自己的变量一般不要用这种变量名；

类似`_xxx`和`__xxx`这样的函数或变量就是非公开的（private），不应该被直接引用，比如`_abc`，`__abc`等；

我们在模块里公开如`greeting()`函数，而把内部逻辑用private函数隐藏起来了，这样，调用`greeting()`函数不用关心内部的private函数细节，这也是一种非常有用的代码封装和抽象的方法，即：

外部不需要引用的函数全部定义成private，只有外部需要引用的函数才定义为public。

（也相当于封装）

2024/10/14