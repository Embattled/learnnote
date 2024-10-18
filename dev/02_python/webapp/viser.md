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



### 1.1.3. World coordinates

世界坐标系

在 viser 的定义里 +z 默认指向上方, 而 xy 在文档里没有书写

而在库中可以通过函数接口来变更

`viser.SceneApi.set_up_direction(direction: Literal[‘+x’, ‘+y’, ‘+z’, ‘-x’, ‘-y’, ‘-z’] | tuple[float, float, float] | ndarray) → None`

### 1.1.4. Cameras 

相机的方向, 在 viser 里使用的是 COLMAP/OpenCV 的约定方向

* Forward: +Z   相机射线方向为 +Z
* Up: -Y        相机正上方为 -Y, 正下方为 +Y 类似于图像平面坐标
* Right : +X    同样类似于图像平面

而要注意, 在 NerfStudio 中, 相机坐标轴的方向由于采用了 `OpenGL/Blender` 的约定, 因此需要做更改
* `Forward: -Z   相机射线方向为 -Z`
* `Up: +Y        相机正上方为 +Y`
* Right : +X    同样类似于图像平面

因此要在两种相机约定之间进行转换, 可以简单的围绕 X 轴旋转 180 度

## 1.2. Development

简要概述 viser 开发的实例和流程

viser 的单元测试比较难以实现, 可以通过一个命令行 `viser-dev-checks` 工具来快速进行 
* runs linting, formatting, and type-checking


### 1.2.1. Message updates

通过一组共享的消息定义 来实现 前后端的通信
* 服务器端, 定义为 Python 的数据类 `~/viser/src/viser/_messages.py`
* 客户端, 定义为 TypeScript 接口 `~/viser/src/viser/client/src/WebsocketMessages.tsx`
两个类的消息类型是完全 1对1 对应的

在开发的过程中, 不应该直接修改 TypeScript 的定义, 而是应该在 Python 的类中进行修改, viser 提供了 `sync_message_defs.py` 来同步对应的更改

### 1.2.2. Client development


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


# 2. API (Basics) - 基本的 API

基本 API 里的内容非常简洁, 所有接口都应以在了对应的3个类里

## 2.1. Viser Server

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

属性: 只有两个
* `scene: SceneApi`  : Handle for interacting with the 3D scene.
* `gui: GuiApi`      : Handle for interacting with the GUI.

### 2.1.1. get_* 函数

`get_clients() → dict[int, ClientHandle]`
* 获取 IDs 和对应的 clientHandle



## 2.2. Scene API

`class viser.SceneApi`
Interface for adding 3D primitives to the scene.

用于向 3D 场景中添加图元(primitives), 所有客户端同步的 Server 和 客户端独立的 Client 的 句柄 (Handles) 都使用该类


### 2.2.1. Scene add 系列方法

向场景中添加图元

* add_camera_frustum  : 添加相机图片视锥
* add_frame         : 添加坐标轴
* add_image         : 添加静态图
* add_point_cloud   : 添加点云
* 


Scene 的元件通用参数:
* `name (str)` – Scene node name. 节点名称
* `wxyz (Tuple[float, float, float, float] | ndarray)` – R_parent_local transformation. 从父节点坐标系到当前节点坐标系的旋转四元数
* `position (Tuple[float, float, float] | ndarray)` – t_parent_local transformation. 父节点坐标系到当前节点坐标系的位移
* `visible (bool)` – Initial visibility of scene node. 控制节点的可见性


`add_camera_frustum(name: str, fov: float, aspect: float, scale: float = 0.3, color: Tuple[int, int, int] | Tuple[float, float, float] | ndarray = (20, 20, 20), image: ndarray | None = None, format: Literal[‘png’, ‘jpeg’] = 'jpeg', jpeg_quality: int | None = None, wxyz: tuple[float, float, float, float] | ndarray = (1.0, 0.0, 0.0, 0.0), position: tuple[float, float, float] | ndarray = (0.0, 0.0, 0.0), visible: bool = True) → CameraFrustumHandle`
* Add a camera frustum to the scene for visualization.
* 





`add_frame(name: str, show_axes: bool = True, axes_length: float = 0.5, axes_radius: float = 0.025, origin_radius: float | None = None, wxyz: tuple[float, float, float, float] | ndarray = (1.0, 0.0, 0.0, 0.0), position: tuple[float, float, float] | ndarray = (0.0, 0.0, 0.0), visible: bool = True) → FrameHandle`
* 添加坐标轴, 可以作为相机坐标, 但是可视化效果不太好



`add_point_cloud(name: str, points: ndarray, colors: ndarray | tuple[float, float, float], point_size: float = 0.1, point_shape: Literal[‘square’, ‘diamond’, ‘circle’, ‘rounded’, ‘sparkle’] = 'square', wxyz: tuple[float, float, float, float] | ndarray = (1.0, 0.0, 0.0, 0.0), position: tuple[float, float, float] | ndarray = (0.0, 0.0, 0.0), visible: bool = True) → PointCloudHandle`
* 添加点云
* 参数:
  * points  : (n, 3) 的点云坐标
  * colors  : (n, 3) 的点的颜色
 



`add_mesh_skinned(name: str, vertices: ndarray, faces: ndarray, bone_wxyzs: tuple[tuple[float, float, float, float], …] | ndarray, bone_positions: tuple[tuple[float, float, float], …] | ndarray, skin_weights: ndarray, color: Tuple[int, int, int] | Tuple[float, float, float] | ndarray = (90, 200, 255), wireframe: bool = False, opacity: float | None = None, material: Literal[‘standard’, ‘toon3’, ‘toon5’] = 'standard', flat_shading: bool = False, side: Literal[‘front’, ‘back’, ‘double’] = 'front', wxyz: Tuple[float, float, float, float] | ndarray = (1.0, 0.0, 0.0, 0.0), position: Tuple[float, float, float] | ndarray = (0.0, 0.0, 0.0), visible: bool = True) → MeshSkinnedHandle`
* 添加 skinned mesh , 可以使用 bone transformations 来进行变形



`add_mesh_simple(name: str, vertices: ndarray, faces: ndarray, color: Tuple[int, int, int] | Tuple[float, float, float] | ndarray = (90, 200, 255), wireframe: bool = False, opacity: float | None = None, material: Literal[‘standard’, ‘toon3’, ‘toon5’] = 'standard', flat_shading: bool = False, side: Literal[‘front’, ‘back’, ‘double’] = 'front', wxyz: tuple[float, float, float, float] | ndarray = (1.0, 0.0, 0.0, 0.0), position: tuple[float, float, float] | ndarray = (0.0, 0.0, 0.0), visible: bool = True) → MeshHandle`
* 添加普通的 mesh



`add_mesh_trimesh(name: str, mesh: trimesh.Trimesh, scale: float = 1.0, wxyz: tuple[float, float, float, float] | np.ndarray = (1.0, 0.0, 0.0, 0.0), position: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 0.0), visible: bool = True) → GlbHandle`
* 添加 trimesh 库格式的 mesh `trimesh.Trimesh`


`add_gaussian_splats(name: str, centers: ndarray, covariances: ndarray, rgbs: ndarray, opacities: ndarray, wxyz: Tuple[float, float, float, float] | ndarray = (1.0, 0.0, 0.0, 0.0), position: Tuple[float, float, float] | ndarray = (0.0, 0.0, 0.0), visible: bool = True) → GaussianSplatHandle`
* 添加 高斯飞溅模型. 该功能正处于试验阶段
* 参数列表:
  * `centers (ndarray)` : Gaussians 的中心坐标 (N,3)
  * `covariances (ndarray)` : Gaussians 的协方差矩阵 (N, 3, 3)
  * `rgbs (ndarray)`  : Gaussian 的渲染颜色. (N, 3)
  * `opacities (ndarray)` : Gaussian 的不透明度. (N, 1)




`add_image(name: str, image: ndarray, render_width: float, render_height: float, format: Literal[‘png’, ‘jpeg’] = 'jpeg', jpeg_quality: int | None = None, wxyz: tuple[float, float, float, float] | ndarray = (1.0, 0.0, 0.0, 0.0), position: tuple[float, float, float] | ndarray = (0.0, 0.0, 0.0), visible: bool = True) → ImageHandle`
* 添加静态图片
* 参数:
  * render_width, render_height : 在坐标系中的渲染长宽
  * wxyz  : 旋转四元数
  * position  : 坐标


### 2.2.2. set_* 函数

`set_up_direction(direction: Literal[‘+x’, ‘+y’, ‘+z’, ‘-x’, ‘-y’, ‘-z’] | tuple[float, float, float] | ndarray) → None`
* 用于调整坐标系的方向
* 默认为 +Z-up



## 2.3. GUI API

`class viser.GuiApi`
同  SceneAPI 一样, 只不过只负责 2D GUI


### 2.3.1. GUI add 系列方法

方法列表
* add_folder    : 用于元件管理
* add_number    : 数字输入组件
* add_button    : 按钮
* add_checkbox  : 勾选框
* add_dropdown  : 下拉菜单组件
* add_slider    : 滑动条组件
* 

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



`add_dropdown(label: str, options: Sequence[TLiteralString], initial_value: TLiteralString | None = None, disabled: bool = False, visible: bool = True, hint: str | None = None, order: float | None = None) → GuiDropdownHandle[TLiteralString]`
* 下拉组件



`add_slider(label: str, min: IntOrFloat, max: IntOrFloat, step: IntOrFloat, initial_value: IntOrFloat, marks: tuple[IntOrFloat | tuple[IntOrFloat, str], …] | None = None, disabled: bool = False, visible: bool = True, hint: str | None = None, order: float | None = None) → GuiSliderHandle[IntOrFloat]`







# 3. API (Advanced)


## 3.1. Handle 通用方法

回调函数  `on_*`
回调函数可以作为 装饰器, 来快速的定义回调

## 3.2. Client Handles

`class viser.ClientHandle`
* 用以管理每个 ClientHandle 链接
* 有各自的几大属性
  * scene  : SceneApi
  * gui     : GuiApi
  * client_id : int
  * camera  : CameraHandle, 管理 client 的 viewport


## 3.3. Camera Handles

## 3.4. GUI Handles

GUI 的各个组件进行独自管理的内容, 实例一般都是作为 add 方法的返回值获取

## 3.5. Events

事件 Event 类型, 这些类型的实例会作为 参数 传递给 callback functions
前两个比较高阶, 


* class viser.ScenePointerEvent   : scene 的点击回调
  * client
  * client_id
  * event_type
  * ray_origin
  * ray_direction
  * screen_pos
* class viser.SceneNodePointerEvent   : scene nodes 的点击回调
  * 
* class viser.GuiEvent  : GUI 的 update or click 
  * client      : ClientHandle
  * client_id   : int
  * target      : TGuiHandle, 受影响的 GUI 元素 (不太懂)





# 4. API (Auxiliary) : 辅佐库

## 4.1. Transforms

viser 内部实现的 李群库, 基于 jaxlie, 被用于 viser 内部以及示例.

Implements SO(2), SO(3), SE(2), and SE(3) Lie groups. Rotations are parameterized via S^1 and S^3.

有一些特殊的构造函数是通过类的静态函数实现的, 学到了 



### 4.1.1. 虚基类 class viser.transforms.MatrixLieGroup

用于定义  matrix Lei groups

`Bases: ABC`


### 4.1.2. class viser.transforms.SOBase

Base class for special orthogonal groups.

`Bases: MatrixLieGroup`




### 4.1.3. class viser.transforms.SO3

用于表示 3D 旋转的 special orthogonal group 
拥有与 numpy 相同的 broadcasting rules


内部参数为 `qw, qx, qy, qz`
切线参数为  `omega_x, omega_y, omega_z`


属性:
`wxyz: onpt.NDArray[onp.floating]`
* Internal parameters. (w, x, y, z) quaternion. Shape should be `(*, 4)`.





# 5. Examples 示例


## 5.1. 00_coordinate_frames

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

##  5.2. 01_image

```python
# 添加全局固定的背景图片
server.scene.set_background_image

# 添加图片
server.scene.add_image
```



## 5.3. 05_camera_commands 

```py

# Move the camera when we click a frame., 把相机的视角移动到对应的坐标轴上
@frame.on_click
def _(_):
    # 从相机的 wxyz 和 位置坐标 构建 相机位置的 SE3
    T_world_current = tf.SE3.from_rotation_and_translation(
        tf.SO3(client.camera.wxyz), client.camera.position
    )

    # 目标坐标系的 wxyz 和 坐标的 SE3
    # 目标坐标系稍微往坐标系的后面一点, 这样视野里能看到坐标系本身
    T_world_target = tf.SE3.from_rotation_and_translation(
        tf.SO3(frame.wxyz), frame.position
    ) @ tf.SE3.from_translation(np.array([0.0, 0.0, -0.5]))

    # 从当前到目标的相对旋转
    T_current_target = T_world_current.inverse() @ T_world_target

    for j in range(20):
      # 微分相对旋转
      # 涉及到 对李群转为李代数的正向和逆向变换
      # log() 转为 李代数, exp 从李代数变回李群
      # 在李代数上可以直接进行线性运算
        T_world_set = T_world_current @ tf.SE3.exp(
            T_current_target.log() * j / 19.0
        )

        # We can atomically set the orientation and the position of the camera
        # together to prevent jitter that might happen if one was set before the
        # other.
        # 原子设定, viser 内部应该是有一些隐藏的并行化处理, 
        with client.atomic():
            client.camera.wxyz = T_world_set.rotation().wxyz
            client.camera.position = T_world_set.translation()

        client.flush()  # Optional!
        time.sleep(10.0 / 60.0)

    # Mouse interactions should orbit around the frame origin.
    # 在旋转的时候 client.camera 始终注视要移动的目标点
    client.camera.look_at = frame.position
```



## 5.4. 11_colmap_visualizer


```py
# viser 的视角有一定的限制, 导致一定程度视角的不自由, 通过重新相机的正上方向可以调整相机视角的自由度

gui_reset_up = server.gui.add_button(
    "Reset up direction",
    hint="Set the camera control 'up' direction to the current camera's 'up'.",
)

@gui_reset_up.on_click
def _(event: viser.GuiEvent) -> None:
    client = event.client
    assert client is not None
    client.camera.up_direction = tf.SO3(client.camera.wxyz) @ np.array(
        [0.0, -1.0, 0.0]
    )



# 注意这里的 inverse 
T_world_camera = tf.SE3.from_rotation_and_translation(tf.SO3(img.qvec), img.tvec)
.inverse()
frame = server.scene.add_frame(
    f"/colmap/frame_{img_id}",
    wxyz=T_world_camera.rotation().wxyz,
    position=T_world_camera.translation(),
    axes_length=0.1,
    axes_radius=0.005,
)
frames.append(frame)


# 基于相机参数, 创建 colmap 的图片视锥
# 注意这里的 name tips, 因为名字是在上面 frame 的节点的下面, 因此不需要指定坐标和旋转, 会直接继承 frame 的
H, W = cam.height, cam.width
fy = cam.params[1]
image = iio.imread(image_filename)
image = image[::downsample_factor, ::downsample_factor]
frustum = server.scene.add_camera_frustum(
    f"/colmap/frame_{img_id}/frustum",
    fov=2 * np.arctan2(H / 2, fy),
    aspect=W / H,
    scale=0.15,
    image=image,
)


```