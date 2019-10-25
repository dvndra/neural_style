# neural_style
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Supported Python version](http://dswami.freevar.com/git_icons/pyversions.svg)](https://www.python.org/downloads/)

An implementation of [neural style][paper] in TensorFlow. <br>

## Running

To run this project, pass in the minimum required arguments (contentFile and styleFile) as:

```
python neural_style.py --content contentFile --style styleFile
```
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

[paper]: https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf
[net]: http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat
