# neural_style
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Supported Python version](http://dswami.freevar.com/git_icons/pyversions.svg)](https://www.python.org/downloads/)

An implementation of [neural style][paper] in TensorFlow. Special thanks to MatConvNet team for using their pretrained VGG 19 model.<br>

## Running

To run this project, pass in the minimum required arguments (contentFile and styleFile) as:

```
python neural_style.py --content contentFile --style styleFile
```
Use `--width` and `--heigth` to set height and width of generated images (default same as that of content image).

Use `--iterations` to change the number of iterations (default 200).  For a 640×640 pixel content file, 200 iterations take 70 seconds on a Tesla K40, 15 seconds on GTX 1080 Ti, or 20 minutes on Macbook Pro'18 (2.3GHz quad-core Intel i5). 

Use `--checkpoint-iterations` to save checkpoint images.


**Parameters:**
```
Required:
--style : style image
--content : content image

Optional:
--vgg : path to pretrained vgg weights (default: './imagenet-vgg-verydeep-19.mat')
--output-folder : path to store output/checkpoint images (default: ./output/)
--width : width of generated images (default: same as content image)
--height : height of generated images (default: same as content image)
--content-weight : Beta (default: 40)
--style-weight : Alpha (default:10)
--style-layer-weight-factor : style layer weight increase factor (default : 1)
--iterations : total iterations (default: 200)
--print-iterations : print statistics every # iterations (default:20)
--checkpoint-iterations : save generated image every # iterations (default:20)
--learning-rate : learning rate in Adam Optimizer
```
**Requirements**
* [Pre-trained VGG network][net] - put it in the top level of this repository, or specify its location using the `--vgg` option. 
* You can install Python dependencies using `pip install -r requirements.txt`.

## Hyperparameters Modification
`--learning-rate` command line argument could be used to adjust how "crude"
the style transfer should be. Lower values mean that style transfer of a finer features
will be favored over style transfer of a more coarse features, and vice versa. Default
value is 2.0. Somewhat extreme examples of what you can achieve:

<img src = "/images/sample_1.jpg" width="300" height ="300"><img src = "/output/sample_1_1_p.png" width="300" height ="300"><img src = "/output/sample_1_10_p.png" width="300" height ="300">

(**left**: 1.0 - original image; **center**: 1.0 - finer features style transfer; **right**: 10.0 - coarser features style transfer)

[paper]: https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf
[net]: http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat
