# The Mesa 3D Graphics Library

https://www.mesa3d.org/

Mesa, also calledMesa3D and The Mesa 3D Graphics Library, is an open source software implementation of OpenGL,Vulkan, and other graphics API specifications.

Mesa translates these specifications to vendor-specific graphics hardware drivers.

Mesa 实现了多种图形API的各种硬件实现, 其中包括最著名的 OpenGL
* Mesa 事实上不是一个新的 API
* 而是一种转换层, 提供了图形API(OpenGL) 到各种图形硬件 (GPU) 驱动的一个转换
* 还包括 GLSL 到 GPU 指令的转换  

Mesa 同时还和其他几个开源项目有深度合作: Direct Rendering Infrastructure, X.org, Wayland

## Platforms and Drivers  

Mesa 主要基于 Linux 开发, 但同时支持其他操作系统.  
主要在维护的 API 为 OpenGL, 但 Mesa 项目也支持其他标准: OpenGL ES, Vulkan, EGL, OpenMAX, OpenCL, VDPAU, VA-API  

MESA 提供支持的硬件有:
* Intel 核显: GMA, HD Graphics, Iris
* AMD Radeon 独显
* NVIDIA GPUs 独显, 从 GeForce 5/FX 往后
* Qualcomm Adreno 2xx-6xx SoC核显
* Broadcom VideoCore 4,5 博通视频核心? VC4 V3D
* ARM Mali Utgard. (Lima)  Midgard., Bifrost. (Panfrost)
* NVIDIA Tegra (Switch 上用的移动芯片)
* Vivante GCxxx. (没听过)

Layered driver include: 作为基础硬件层的驱动项目
* D3D12     : 基于 Direct3D 12 API 的 OpenGL 支持
* SVGA3D    : 基于 VMware virtual GPU 的支持
* VirGL     : 用于 QEMU 
* Zink      : 提供 基于 Khronos'Vulkan 的 OpenGL API 支持

Software drivers : 软件编写的图形驱动?
* LLVMpipe  : uses LLVM for JIT code generation and is multi-threaded
* Softpipe  : a reference Gallium driver



# Drivers


## LLVMpipe (Gallium LLVMpipe)

是一个 软件光栅器 (software rasterizer), 使用 LLVM 来实现实时 代码生成.  
通过 LLVM IR 来实现 Shaders, point/line/triangle rasterization, vertex process, 并转换为 x86, x86-64 或者 ppc64le 的机器码.  

LLVMpipe 支持多线程, 目前最多支持到 32 个, 是 Mesa 项目中速度最快的 software rasterizer. 

 
