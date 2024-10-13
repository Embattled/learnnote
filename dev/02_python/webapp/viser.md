# 1. viser - library for interactive 3D visualization in Python

https://github.com/nerfstudio-project/viser

受  Pangolin, rviz, meshcat, and Gradio 启发的, 用于开发 交互式 3D 可视化 工具的 python 库

是 nerfstudio 的基石


## 1.1. Frame Conventions - 框架的约定

说明在 viser 使用的坐标轴的约定



### 1.1.1. Scene tree naming

所有添加到 scene 的 object 都会被初始化为 node in a scene tree

tree 的结构由分配给节点的名称来确定

假如说添加一个 object 名为 `/base_link/shoulder/wrist` 则其代表了由三个节点构成的树, 其中 `/base_link` 是父节点

在操作的时候, 对于任意节点的操作都会影响其子节点, 而不会影响其父节点


### 1.1.2. Poses

viser 里的 Pose 是使用  a pair of fields 定义的
* `wxyz`        : 单位四元数方向 term, 始终是 4D 的数据, 可以由 `SO(3)` 矩阵转换而来
* `position`    : 位置 term, 始终是 3D 数据

这两项数据提供了 从 local 坐标框架到 父框架的坐标转换信息

$$p_{\text{parent}}=
\begin{bmatrix}
    R&t
\end{bmatrix}
\begin{bmatrix}
    p_{\text{local}}\\
    1
\end{bmatrix}$$



### World coordinates

世界坐标系

在 viser 的定义里 +z 默认指向上方, 而 xy 在文档里没有书写

而在库中可以通过函数接口来变更

`viser.SceneApi.set_up_direction(direction: Literal[‘+x’, ‘+y’, ‘+z’, ‘-x’, ‘-y’, ‘-z’] | tuple[float, float, float] | ndarray) → None`

### Cameras 

相机的方向, 在 viser 里使用的是 COLMAP/OpenCV 的约定方向

* Forward: +Z   相机射线方向为 +Z
* Up: -Y        相机正上方为 -Y, 正下方为 +Y 类似于图像平面坐标
* Right : +X    同样类似于图像平面

而要注意, 在 NerfStudio 中, 相机坐标轴的方向由于采用了 `OpenGL/Blender` 的约定, 因此需要做更改
* `Forward: -Z   相机射线方向为 -Z`
* `Up: +Y        相机正上方为 +Y`
* Right : +X    同样类似于图像平面

因此要在两种相机约定之间进行转换, 可以简单的围绕 X 轴旋转 180 度

## Development

简要概述 viser 开发的实例和流程

viser 的单元测试比较难以实现, 可以通过一个命令行 `viser-dev-checks` 工具来快速进行 
* runs linting, formatting, and type-checking


### Message updates

通过一组共享的消息定义 来实现 前后端的通信
* 服务器端, 定义为 Python 的数据类 `~/viser/src/viser/_messages.py`
* 客户端, 定义为 TypeScript 接口 `~/viser/src/viser/client/src/WebsocketMessages.tsx`
两个类的消息类型是完全 1对1 对应的

在开发的过程中, 不应该直接修改 TypeScript 的定义, 而是应该在 Python 的类中进行修改, viser 提供了 `sync_message_defs.py` 来同步对应的更改

### Client development


启动书写好的 viser 服务端代码 `python 05_camera_commands.py`

当 viser 脚本的服务启动后
* An HTTP URL, like `http://localhost:8080`, which can be used to open a pre-built version of the React frontend.
* A websocket URL, like ws://localhost:8080, which client applications can connect to.


如果客户端的源文件发生更改, viser 会自动检测到, 并重启重新构建对应的客户端. 但是为了更快速的反应更改而跳过重新构建, 可以使用 客户端的 开发版本
* 安装 nodejs, yarn, 以及 `~/viser/src/viser/client` 下的 yarn 依赖
* 在 `cd ~/viser/src/viser/client` 下面运行 `yarn start`


格式化? (没看懂, 前端代码的格式化?)
* 使用 `prettier` 库
* `from ~/viser/src/viser/client.`
  * `prettier -w .`
  * `npx prettier -w .`


# API (Basics) - 基本的 API

基本 API 里的内容非常简洁, 所有接口都应以在了对应的3个类里

## Viser Server

`class viser.ViserServer`
全局的核心 API， 在该类的实例初始化的时候, 会在一个线程里建立 web server, 并提供对应的高级 API 用于交互式 3D 可视化

客户端连接到 web browser 后, 会被提供两个组件
* 3D scene  : `ViserServer.scene` 用于管理 3D 图元
* 2D GUI panel  : `ViserServer.gui` 用户管理 2D GUI 面板里的元素

状态共享: 通过特定服务器的 API 添加的 3D 或 2D 原件会在连接到服务器的所有客户端之间自动共享和同步, 例如
* `SceneApi.add_point_cloud()` `GuiApi.add_button()`

若要操作客户端的本地元素, 使用
* `ClientHandle.scene`
* `ClientHandle.gui`


构造函数参数:
* host  : Host to bind server to
* port  : Port to bind server to
* label : Label shown at the top of the GUI panel

属性:
* `scene: SceneApi`  : Handle for interacting with the 3D scene.
* `gui: GuiApi`      : Handle for interacting with the GUI.



## Scene API

`class viser.SceneApi`
Interface for adding 3D primitives to the scene.

用于向 3D 场景中添加图元(primitives), 所有客户端同步的 Server 和 客户端独立的 Client 的 句柄 (Handles) 都使用该类


### Scene add 系列方法

向场景中添加图元




## GUI API

`class viser.GuiApi`
同  SceneAPI 一样, 只不过只负责 2D GUI


### GUI add 系列方法

方法列表
* add_folder
* add_number
* add_slider

`add_folder(label: str, order: float | None = None, expand_by_default: bool = True, visible: bool = True) → GuiFolderHandle`
添加一个GUI元件 folder, 返回一个 Handle 用于控制句柄的结束, 通常使用 with 的上下文管理器来使用, 根据 GUI 内容对各个组件进行分类, 在 folder 里进行定义即可
* 参数:
  * label(str)  : folder 的标签, 用来展示说明
  * order(float|None)   : 在GUI的顺序, 可以通过一个浮点数来指定
  * expand_by_default (bool) : GUI中默认是否展开
  * visible (bool) : 元件的可见性
* returns : GuiFolderHandle, A handle that can be used as a context to populate the folder.


`add_number(label: str, initial_value: IntOrFloat, min: IntOrFloat | None = None, max: IntOrFloat | None = None, step: IntOrFloat | None = None, disabled: bool = False, visible: bool = True, hint: str | None = None, order: float | None = None) → GuiNumberHandle[IntOrFloat]`
添加一个 数字输入 组件, 参数超级多


`add_slider(label: str, min: IntOrFloat, max: IntOrFloat, step: IntOrFloat, initial_value: IntOrFloat, marks: tuple[IntOrFloat | tuple[IntOrFloat, str], …] | None = None, disabled: bool = False, visible: bool = True, hint: str | None = None, order: float | None = None) → GuiSliderHandle[IntOrFloat]`






# API (Advanced)

## Client Handles

## Camera Handles

## GUI Handles

GUI 的各个组件进行独自管理的内容, 实例一般都是作为 add 方法的返回值获取

# Examples 示例


## 00_coordinate_frames

```python
import viser
import time

# 启动默认参数的服务器
server = viser.ViserServer()

# 添加固定内容
server.scene.add_frame

# 死循环是确保服务器持续启动的关键, 是必需品
# 在死循环里书写的场景内容是需要不断更新的内容
while True:
    server.scene.add_frame
    leaf = server.scene.add_frame

    # 移除定义的 元件
    leaf.remove()

    # 类似于 OpenCV 的 waitkey()???
    time.sleep(0.5)
```

##  01_image

```python
# 添加全局固定的背景图片
server.scene.set_background_image

# 添加图片
server.scene.add_image
```