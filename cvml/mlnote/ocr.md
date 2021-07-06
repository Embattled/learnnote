# Scene text detection and recognition

Constructing a high-quality scene text recognition system is a non-trivial task.

## Project link

* MASTER
  * [link-Tensorflow](https://github.com/jiangxiluning/MASTER-TF)
  * [link-Torch](https://github.com/wenwenyu/MASTER-pytorch)
* lmdb dataset
  * [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark)
## Survey

* (2016) Scene text detection and recognition: recent advances and future trends 


* Word as basic unit.
* Character as basic unit.


### Future


* Multi-orientation
* Multi-laguage
* Deep learning big data.

### Application

* Signboard for autonomous driving.
* ID card scan for a bank.
* Key information extraction in Robotic Process Automation.

### Difficulty


1. Diversity of scene text
   * Document images usually with fegular font, single colour, consistent size, uniform arrangement.
   * Texts in natural scenes may bear entirely different fonts, colours, scales, orientations. 
2. Complexity of background
   * Signs, fences, bricks, grasses are easily to cause conusions and errors.
3. Interference factors
   * Noise, blur, distortion, low resolution, nonuniform illumination and partial occlusion.
   * Complex clutter.


### Direction

1. Text detection
   * Discover and locate the regions possibly contaioning text from natural images.
   * Not to preform recognition.
2. Text recognition
   * Supposes that text have beed decected.
   * Only focus on the process of converting the detected text regions into computer readable and editable symbols.
3. End-to-end text recognition
   * Constructing end-to-end text recognition systens that accomplish both the detection and recognition task.


Regular scene text recognition
* Recognize a sequence of characters from an almost straight text image.
* Can considered as an image-based sequence recognition problem.
* Method:
  * Human-designed features
  * CTC based method
  * Attention-based methods.

Irregular scene text recognition
* Various curved shapes and perspective distortions.
* Method:
  * Rectification based.
  * multi-direction encoding based
  * attention-based



# Conventional methods


## Text detection：  

* Texture based method
  * Treat texts as a special type of texture.
  * Make use of their textural properities.
    * local intensities
    * filter responses
    * wavelet coefficients.
  * Usually
    * Computationally expensive.
    * All locations and scales should be scanned.
    * Sensitive to rotation and scale change.
* Component based method (mainstream)
  * Firstly, extract candidate components through a variety of ways. (color clustering or extreme region extraction.)
  * Secondly, filter out non-text components using manually designed rules, or automatically trained classifiers.

### Representation

Keywords:
* Color similarity
* Spatial distance
* Relative size of regions.

Text Detection:  
* SWT. Stroke Width Transform.
* MSER. Maximally Stable Extremal Regions.
* Sparse representation.



### Title Memo

* (2012) Detecting texts of arbitrary orientations in natural images.
* Robust scene text detection with convolu- tion neural network induced Mser trees.


## Text Recognition

Keywords:  
* Automatically create character templates according to che characteristics of natural images.
* Surface fitting classifier and specifically designed character recognition algorith,.


## End-to-end 


# Deep learning based methods


## Attention-based


