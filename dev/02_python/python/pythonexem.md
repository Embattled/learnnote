# 1. Extending and Embedding the Python Interpreter

主要面向C/C++的Python解释器扩展和嵌入  

# Extending Python with C or C++

通过 C 往 Python 里追加 模组, 可以实现两样在 Python 里无法直接做到的事情
* 追加 built-in 目标类型
* 调用 C library 以及系统调用

为了实现 C 的extensions, Python API 定义了一系列 函数, 宏, 变量用于访问 Python run-time 的各种方面.
在 C 程序中通过包含 `Python.h` 头文件来使用

Note: 通过 C 写的 Python 拓展只能用于 CPython, 而无法用于其他语言实现的 Python 解释器. 
Python 官方更加推荐使用 `ctypes` or `cffi` 来实现与 C 库接口的交互, 这样可以最大程度的确保不同 python 实现见的兼容性



# 2. Write Module In C/C++

1.  You must include Python.h before any standard headers are included.
2.  在Python调用和 C函数中存在名称转换

```cpp
#define PY_SSIZE_T_CLEAN
#include <Python.h>

//  python 调用 status = spam.system("ls -l")
// C中的函数名写作 spam_system
static PyObject * spam_system(PyObject *self, PyObject *args)
{
    const char *command;
    int sts;

    if (!PyArg_ParseTuple(args, "s", &command))
        return NULL;
    sts = system(command);
    return PyLong_FromLong(sts);
}
```


## 2.1. Python.h

All user-visible symbols defined by Python.h have a prefix of `Py` or `PY`.

