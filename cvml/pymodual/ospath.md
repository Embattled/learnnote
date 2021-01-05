# 1. File System

os, glob,  shutil
## 1.1. pathlib
The pathlib module was introduced in Python 3.4 to deal with these challenges. It gathers the necessary functionality in one place and makes it available through methods and properties on an easy-to-use Path object.  

### 1.1.1. pathlib.Path
The best way to construct a path is to join the parts of the path using the special operator `/`.


#### 1.1.1.1. 使用Path来定义路径
You can use `Path.cwd()` or `Path('.') `to refer to your currently working directory.

```py
from pathlib import Path

print("Getting 'text1.txt'")
path1 = Path('.') / 'folder1' / 'text1.txt'
print(path1)

```

#### 1.1.1.2. 使用Path来获取路径的属性

.name, .parent, .stem, .suffix, .anchor 

The pathlib.Path is represented by either a `WindowsPath` or a `PosixPath`.

```py
path1 = Path('.') / 'folder1' / 'text1.txt'
print([path1, path1.name, path1.stem, path1.suffix, path1.parent, path1.parent.parent, path1.anchor])
# [PosixPath('folder1/text1.txt'), 'text1.txt', 'text1', '.txt', PosixPath('folder1'), PosixPath('.'), '']

```

#### 1.1.1.3. 获取文件列表

Using `.iterdir()` you can get all the files in a folder.   
By list comprehension, you can convert this into a list object.  


```py
path2 = Path('.') / 'folder1'
path_list = list(path2.iterdir())
print(f'List of files: {path_list}')
'''
List of files: [PosixPath('folder1/text1.txt'), PosixPath('folder1/text2.txt'), PosixPath('folder1/text3.txt')]
'''

print(f'Number of files: {len(path_list)}')

```

#### 1.1.1.4. Path 的有用方法

```py

# Path.exists()
# Checks if a path exists or not. Returns boolean value.
file_path = Path('.') / 'folder1' / 'text2.txt'
print(file_path.exists())


# Path.glob()   Globs and yields all file paths matching a specific pattern. 
#  mark (?), which stands for one character.
print("\nGetting all files with .csv extension.")
dir_path = Path('.') / 'folder1'
file_paths = dir_path.glob("*.csv")
print(list(file_paths))

# Path.rglob()
# This is like Path.glob method but matches the file pattern recursively.
print("\nGetting all .txt files starts with 'other' in all directories.")
dir_path = Path('.')
file_paths = dir_path.rglob("other?.txt")
print(list(file_paths))

# Path.mkdir()
# Creates a new directory at this given path. 
dir_path = Path('.') / 'folder_new' / 'folder_new_1'
# parents:(boolean) If parents is True, 
#         any missing parents of this path are created as needed. 
#          Otherwise, if the parent is absent, FileNotFoundError is raised.
dir_path.mkdir(parents=True)
# exist_ok: (boolean) 
#       If False, FileExistsError is raised if the target directory already exists. 
#       If True, FileExistsError is ignored.

 
# Path.rename(target)   This will raise FileNotFoundError if the file is not found
dir_path = Path('.') / 'folder_new' / 'folder_new_1'
dir_path.rename(dir_path.parent / 'folder_n1')


# Replaces a file or directory to the given target. Returns the new path instance. 
dir_path = Path('.') / 'folder_new' / 'folder_n1'
dir_path2 = Path('.') / 'folder1'  
dir_path.replace(dir_path.parent / dir_path2)



# Path.rmdir()
# Removes a path pointing to a file or directory. The directory must be empty, otherwise, OSError is raised.
```
