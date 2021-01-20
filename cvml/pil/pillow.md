# 1. Pillow 

Pillow is the friendly PIL fork

所有子模块都定义在 `PIL` 里  


# 2. Image

`from PIL import Image`  
定义在 Image 包中的同名类  

基本操作 `im 为实例`:
* im.show()
* im.save() : 第一个参数是 path, 第二个参数是格式, 默认通过文件名推断保存的格式

## 2.1. 定义

### 2.1.1. open
```py
im = Image.open("hopper.ppm")
```

### 2.1.2. new

```py
PIL.Image.new(mode, size, color=0)
```

## 2.2. Information

class 拥有的情报:
* format    : 来源的格式, 如果是从内存中创建的图像则为 `None`
* size
* mode
```py
print(im.format, im.size, im.mode)
# PPM (512, 512) RGB
```

## 2.3. transform



# Ima#geDraw

## 类方法
### .text

```py
ImageDraw.text(
    xy, 
    text, 
    fill=None, 
    font=None, 
    anchor=None, 
    spacing=4, 
    align='left', 
    direction=None, 
    features=None, 
    language=None, 
    stroke_width=0, 
    stroke_fill=None, 
    embedded_color=False)
```
# 3. Others
## 3.1. ImageFont

定义了一个同名的 ImageFont 类, 可以保存 bitmap 类型的 fonts  

### 3.1.1. truetype

```py

PIL.ImageFont.truetype(font=None, size=10, index=0, encoding='', layout_engine=None)
```