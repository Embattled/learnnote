# Implicit Neural Representation

通过神经网络参数 来不同于传统方式的 隐式表示一些东西

# Neural Radiance Fields (NeRF)

通过 神经网络表示 Radiance Fields

## Topics

* mip-NeRF 360 consistently produces fewer artifacts and higher reconstruction quality. 
* low-dimensional generative latent optimization (GLO) vectors introduced in NeRF in the Wild, learned real-valued latent vectors that embed appearance information for each image. the model can capture phenomena such as lighting changes without resorting to cloudy geometry, a common artifact in casual NeRF captures. 
* exposure conditioning as introduced in Block-NeRF, 


* NeRF's baked representations


## Practical Concerns


* 输入数据 : a dense collection of photos from which 3D geometry and color can be derived, every surface should be observed from multiple different directions.
* For example, most of the camera’s properties, such as white balance and aperture, are assumed to be fixed throughout the capture.
* scene itself is assumed to be frozen in time: lighting changes and movement should be avoided. 
* As photos may inadvertently contain sensitive information, we automatically scan and blur personally identifiable content.


## Rest questions


* with scene segmentation, adding semantic information to the scenes
* Adapting NeRF to outdoor photo collections
* Real time render

## Referance


Google Blog
Reconstructing indoor spaces with NeRF
Wednesday, June 14, 2023
https://ai.googleblog.com/2023/06/reconstructing-indoor-spaces-with-nerf.html


# 2D - Implicit Image Function

通过 神经网络表达 2D 图像


