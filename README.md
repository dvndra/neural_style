# neural_style
[![License: MIT](http://dswami.freevar.com/git_icons/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Supported Python version](http://dswami.freevar.com/git_icons/pyversions.svg)](https://www.python.org/downloads/)

An implementation of [neural style][paper] in TensorFlow.<br> 
Special thanks to MatConvNet team for using their pretrained VGG 19 model in this implementation.<br>

## Running

To run this project, pass in the minimum required arguments (contentFile and styleFile) as:

```
python neural_style.py --content contentFile --style styleFile
```
Use `--width` and `--heigth` to set height and width of generated images (default same as that of content image).

Use `--iterations` to change the number of iterations (default 200).  For a 640×640 pixel content file, 200 iterations take 70 seconds on a Tesla K40, 15 seconds on GTX 1080 Ti, or 20 minutes on Macbook Pro'18 (2.3GHz quad-core Intel i5). 

Use `--checkpoint-iterations` to save checkpoint images.

***Note***: Random noises are added to content image to generate initial image for training. Thus, the training starts with very low content error and very high style error. Style error will then decrease on expense of content error to minimize total error. You can start seeing interesting results from 100+ iterations in most settings. 

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
`--learning-rate` command line argument could be used to adjust to what extent
the style transfer should be applied to the content. Higher values mean that style transfer of finer features
will be favored over style transfer of more coarse features, and vice versa. Default
value is 2.0. Somewhat extreme examples of what you can achieve:

&nbsp;&nbsp;&nbsp;<img src = "/images/sample_1.jpg" width="250" height ="250">&nbsp;&nbsp;&nbsp;<img src = "/output/sample_1_1_p.png" width="250" height ="250">&nbsp;&nbsp;&nbsp;<img src = "/output/sample_1_10_p.png" width="250" height ="250">

(**left**: my image at heritage site; **center**: 1.0 - coarser features style transfer; **right**: 10.0 - finer features style transfer) Style image used: picasso.jpg<br><br>

`--style-weight` or `--content-weight` command line argument could be supplied to explicitly set how close the generated image to the style and content images. Higher values mean that generated image is closer to the corresponding style/ content. Default value for style_weight and content_weight is 40 and 10 respectively. Please find below images generated with variation in style_weight:

&nbsp;&nbsp;&nbsp;<img src = "/images/sample_2.jpg" width="250" height ="250">&nbsp;&nbsp;&nbsp;<img src = "/output/sample_2_beta_10.png" width="250" height ="250">&nbsp;&nbsp;&nbsp;<img src = "/output/sample_2_beta_100.png" width="250" height ="250">

(**left**: my image at Mt.Fuji in summer'16; **center**: 10.0 - content favored over style; **right**: 100.0 - style favored) Style image used: van_gogh.jpg<br><br>

`--style-layer-weight-factor` parameter adjusts the granularity of style transfer application. Lower values favors style transfer of finer features over more coarse features, and vice versa. Default value is 1.0 - all layers treated equally. Thus, it gives results similar to what can be achieved with tweaking learning rate. An extreme example acieved for this parameter:

&nbsp;&nbsp;&nbsp;<img src = "/images/sample_3.jpg" width="250" height ="250">&nbsp;&nbsp;&nbsp;<img src = "/output/sample_3_02.png" width="250" height ="250">&nbsp;&nbsp;&nbsp;<img src = "/output/sample_3_2.png" width="250" height ="250">

(**left**: my image at Gangtok, India; **center**: 0.2 - coarser features style transfer; **right**: 2.0 - finer features style transfer) Style image used: van_gogh.jpg<br><br>

[paper]: https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf
[net]: http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat
