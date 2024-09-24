# 1. Colmap

COLMAP is a general-purpose Structure-from-Motion (SfM) and Multi-View Stereo (MVS) pipeline with a graphical and command-line interface.

COLMAP 是一个通用目的的 Structure-from-Motion (SfM) 和 Multi-View Stereo (MVS) pipline  
提供了完整的 GUI 和 CLI 界面, 以及能够应用于顺序或者无序输入的图像重构技术.   

基于 BSD License 的开源, 论文源于  
* Structure-from-Motion Revisited  CVPR 2016
`Johannes L. Schönberger` and `Frahm, Jan-Michael`
* Pixelwise View Selection for Unstructured Multi-View Stereo  ECCV 2016
`Johannes L. Schönberger` and Zheng, Enliang and Pollefeys, Marc and `Frahm, Jan-Michael`

此外, 图像检索的内容也基于相同作者的论文
* A Vote-and-Verify Strategy for Fast Spatial Verification in Image Retrieval  ACCV 2016
`Schönberger, Johannes Lutz` and Price, True and Sattler, Torsten and `Frahm, Jan-Michael` and Pollefeys, Marc

简易的Pipeline 包括
* 使用 SfM 技术从 输入图像中获取相机的 pose
* 获取到的相机 pose 和图像一起作为 MVS 的输入

## 1.1. Camera Model

Colmap 本身提供了多种相机模型, 如果模型实现不知道参数, 则 Colmap 会自动选择最简单 但足够建模畸变的模型  



# 2. 使用指南 


## 2.1. Feature Detection and Extraction

colmap 运行的第一步, 特征提取



## 2.2. Feature Matching and Geometric Verification

第二步, 对各个图片中提取的图像进行匹配  

匹配过程有多种模式可以选择  
* Exhaustive Matching 穷举匹配: 
  * 每一张图片都和其他所有的图进行匹配, 在图片数量不过百张以内的时候能后获得最好的重建效果
  * block size 决定了 同时从磁盘中加载到内存中的图像数量
* Sequential Matching 顺序匹配: 
  * 对于图像的名称和图像的采集顺序相关的情况下, 此模式最有用
  * 连续帧具有视觉重叠, 不用彻底匹配所有图像对, 而是匹配相邻的
  * 在该模式下, 图像文件名必须按照顺序排列
* Spatial Matching 空间匹配:
  * 需要输入图像本身已经有了大体的空间位置, 将会匹配空间最近的图像
  * 空间位置可以在数据库管理中手动设置, 也可以从 Exif 中的 GPS 信息里获取
  * 如果有 prior 位置的话, 此模式最推荐
* Transitive Matching 传递匹配:
  * 基于特征的传递匹配, A-B, B-C 则会尝试直接将 A-C 匹配
  * 文档中没有更多的说明, 应该是特征压倒一切的匹配模式
* 

## 2.3. Sparse Reconstruction - 稀疏重建

特征提取以及匹配过后, 就可以进入 incremental reconstruction 增量重构模式   

从最开始的 initial image pair 开始, 增量的将别的 Images 的 triangulating new point 加入到图里面.  
该过程可通过 GUI 来实时观看进度.

如果不是所有图像都注册到同一个地图模型中, COLMAP 会生成多个模型.  如果不同的模型具有相同的注册图像, 则可以在之后的处理用别的 可执行文件 `model_converter` 来手动合并为单个重建  


理想情况下, 所有图像都会被 registered, 如果在实际中没能成功, 则
* 执行附加 matching, 例如使用穷举匹配, 启用引导匹配, 增加词汇数, 或者增加顺序匹配中的重叠等
* 手动选择构图的初始图象对

# 3. Command-line Interface


## 3.1. feature_extractor
```sh
$ colmap feature_extractor -h

    Options can either be specified via command-line or by defining
    them in a .ini project file passed to `--project_path`.

      -h [ --help ]
      --project_path arg
      --database_path arg
      --image_path arg
      --image_list_path arg
      --ImageReader.camera_model arg (=SIMPLE_RADIAL)
      --ImageReader.single_camera arg (=0)
      --ImageReader.camera_params arg
      --ImageReader.default_focal_length_factor arg (=1.2)
      --SiftExtraction.num_threads arg (=-1)
      --SiftExtraction.use_gpu arg (=1)
      --SiftExtraction.gpu_index arg (=-1)
      --SiftExtraction.max_image_size arg (=3200)
      --SiftExtraction.max_num_features arg (=8192)
      --SiftExtraction.first_octave arg (=-1)
      --SiftExtraction.num_octaves arg (=4)
      --SiftExtraction.octave_resolution arg (=3)
      --SiftExtraction.peak_threshold arg (=0.0066666666666666671)
      --SiftExtraction.edge_threshold arg (=10)
      --SiftExtraction.estimate_affine_shape arg (=0)
      --SiftExtraction.max_num_orientations arg (=2)
      --SiftExtraction.upright arg (=0)
      --SiftExtraction.domain_size_pooling arg (=0)
      --SiftExtraction.dsp_min_scale arg (=0.16666666666666666)
      --SiftExtraction.dsp_max_scale arg (=3)
      --SiftExtraction.dsp_num_scales arg (=10)

The available options can either be provided directly from the command-line or through a .ini file provided to --project_path.
```

## 3.2. sequential_matcher



## model_converter  - 模型格式转换



