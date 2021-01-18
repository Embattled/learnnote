# 1. torchvision

The torchvision package consists of :
1. popular datasets
2. model architectures
3. common image transformations for computer vision


# 2. torchvision.io

* The torchvision.io package provides functions for performing IO operations. 
* They are currently specific to reading and writing video and images

## 2.1. Image part

1. torchvision.io.read_image(path: str) → torch.Tensor
2. torchvision.io.decode_image(input: torch.Tensor) → torch.Tensor
3. torchvision.io.encode_jpeg(input: torch.Tensor, quality: int = 75) → torch.Tensor
4. torchvision.io.write_jpeg(input: torch.Tensor, filename: str, quality: int = 75)
5. torchvision.io.encode_png(input: torch.Tensor, compression_level: int = 6) → torch.Tensor
6. torchvision.io.write_png(input: torch.Tensor, filename: str, compression_level: int = 6)




# 3. torchvision.models

保存了预定义的模型用于不同任务
* image classification
* video classification
* pixelwise semantic segmentation
* object detection, instance segmentation , person keypoint detection

