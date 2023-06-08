# COLMAP 

COLMAP 是一个通用目的的 Structure-from-Motion (SfM) 和 Multi-View Stereo (MVS) pipline  
提供了完整的 GUI 和 CLI 界面, 以及能够应用于顺序或者无序输入的图像重构技术.   

基于 BSD License 的开源, 论文源于  

Structure-from-Motion Revisited  CVPR 2016
Johannes L. Schönberger and Frahm, Jan-Michael

Pixelwise View Selection for Unstructured Multi-View Stereo  ECCV 2016
Johannes L. Schönberger and Zheng, Enliang and Pollefeys, Marc and Frahm, Jan-Michael

此外, 图像检索的内容也基于相同作者的论文
A Vote-and-Verify Strategy for Fast Spatial Verification in Image Retrieval  ACCV 2016
Schönberger, Johannes Lutz and Price, True and Sattler, Torsten and Frahm, Jan-Michael and Pollefeys, Marc

简易的Pipeline 包括
* 使用 SfM 技术从 输入图像中获取相机的 pose
* 获取到的相机 pose 和图像一起作为 MVS 的输入



