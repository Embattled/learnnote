# trimesh 

一个很受欢迎的轻量化 3D mesh 处理库

只依赖于 numpy

库本身没有特别清晰的层级结构

# quick start



# trimesh

trimesh 最顶层的接口

包括了最上层的类对象和高级 IO 函数


## basic I

* trimesh.load          : Load mesh or vectorized path into objects like Trimesh, Path2D, Path3D, Scene.
* trimesh.load_mesh     : Load a mesh file into a Trimesh object.
* trimesh.load_path     : Load a file to a Path file_object.


```py
# 通用的多功能 load
trimesh.load(file_obj: str | Path | IO | BytesIO | StringIO | BinaryIO | TextIO | dict | None, 
                file_type: str | None = None, 
                resolver: Resolver | dict | None = None, 
                force: str | None = None, 
                **kwargs) → Geometry | list[Geometry]

# mesh 专用 load
trimesh.load_mesh(file_obj: str | Path | IO | BytesIO | StringIO | BinaryIO | TextIO | dict | None, 
                    file_type: str | None = None, 
                    resolver: Resolver | dict | None = None, 
                    **kwargs) → Geometry | list[Geometry]

# path 专用 load
trimesh.load_path(file_obj, file_type=None, **kwargs)

# 从远程 url 读取 mesh
trimesh.load_remote(url, **kwargs)
```

## class trimesh.Geometry

Geometry 是所有高级类的父类 (虚基类) Bases: ABD

内部定义有很多的 abstract


## class trimesh.PointCloud

Bases: Geometry3D

Hold 3D points in an object which can be visualized in a scene.

## class trimesh.Scene

Bases: Geometry3D

A simple scene graph which can be rendered directly via pyglet/openGL or through other endpoints such as a raytracer. Meshes are added by name, which can then be moved by updating transform in the transform tree.


## class trimesh.Trimesh

Bases: Geometry3D
A Trimesh object contains a triangular 3D mesh.

最核心最顶层的 Mesh 类

构造函数参数 : array 支持的类型写的超级详细所以省略
* vertices      ((n, 3) float)          – Array of vertex locations, 顶点的 3D 坐标
* faces         ((m, 3) or (m, 4) int)  – Array of triangular or quad faces (triangulated on load), 在读取的适合统一转成 三角面
* face_normals  ((m, 3) float)          – Array of normal vectors corresponding to faces, 面的法线向量
* vertex_normals ((n, 3) float)         – Array of normal vectors for vertices, 顶点的法线向量
* metadata      (dict)                  – Any metadata about the mesh, 应该是用来存储用户自定义数据的
* process       (bool)  : 用于消除输入数据的 NaN 和重复的 faces
* use_embree    (bool)  : 如果为 True 的话则会尝试着使用 pyembree(?) 光线追踪器, 如果 其不可用的话则返回去调用 rtree/numpy
* initial_cache 
* visual 

### is_* property


* is_convex : bool, 该网格是否是 凸的
* is_watertight : bool, 该网格是否不漏水(?), 每条边都包含在两个面中, 即封闭的 surface
* is_winding_consistent : bool, 绕序一致性(?), 所有面的顶点顺序都遵顼相同的绕序规则, 保证面的法向量方向一致 
* is_volume : bool, 该网格是否有表示 volume 的所需要的所有属性, 包括了
  * is_watertight
  * consistent winding
  * outward facing normals

