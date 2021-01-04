# Extending and Embedding the Python Interpreter

主要面向C/C++的Python解释器扩展和嵌入  


# Write Module In C/C++

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


## Python.h

All user-visible symbols defined by Python.h have a prefix of `Py` or `PY`.

