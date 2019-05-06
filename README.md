# Picasso Colorful World

## Fast style transfer

Neural style transfer is the process of taking the style of one image then applying it to the content of another image. There is <a href="https://arxiv.org/abs/1705.04058">a review about Neural Style Transfer </a> that is worked out by Yongcheng Jing, Yezhou Yang, Zunlei Feng, Jingwen Ye, Yizhou Yu, Mingli Song

<img src="images/neural-style-transfer-a-view.png"/>

In this practice, we choose "Model-Optimisation-Based Offline Neural Methods", also called as "fast style transfer". It's Per-Style-Per-Model Neural Methods. 

Following describes the details about how we cook Picasso Colorful World. Following this practice, you can learn how to build same scenario to Web, UWP(Windows), Android, and NCS that Intel announced for AIOT usage.

## About Pablo Picasso

For more Pablo Picasso, please refer to [wiki](https://en.wikipedia.org/wiki/Pablo_Picasso)

## Tensorflow

The source code in this project is written in [Tensorflow](https://www.tensorflow.org/)

You can find related instruction to train your own model file by this sample code.

Following also brief some tips about the experience while we were working on this project.


## Features

In this project, it will provide the following packages
* Training Van Gogh gallery with Python
* Inference with real time camera and still images and 
  * Deployment on <a href="https://github.com/acerwebai/PicassoColorfulWorld-Windows">Windows applications </a>
  * Deployment on <a href="https://github.com/acerwebai/PicassoColorfulWorld-Android">Android applications </a>
  * Deployment on <a href="https://github.com/acerwebai/PicassoColorWorld-Web">Web pages </a>
  * Deployment on <a href="https://github.com/acerwebai/PicassoColorfulWorld-NCS"> NCS </a> that Intel announced for AIOT usage.



## Getting Started


### Getting the Code

```
git clone https://github.com/acerwebai/PicassoColorfulWorld.git
```

### Get Pre-Trained Model

You can download the pre-trained models from here and should find the checkpoint files for each models

### Prerequisites

* Python 3.6

* (Optional) If your machine support [nVidia GPU with CUDA](https://developer.nvidia.com/cuda-gpus), please refer to the installation from nVidia 
	* CUDA v9.0: https://developer.nvidia.com/cuda-90-download-archive
	* cuDNN v7.3.0 for CUDA 9.0: https://developer.nvidia.com/rdp/cudnn-archive
	* Note: CUDA and cuDNN has [dependencies](https://docs.nvidia.com/deeplearning/sdk/cudnn-support-matrix/index.html)

* Tensorflow 1.12.0 
  * pip install tensorflow==1.12.0 for CPU
  * pip install tensorflow-gpu==1.12.0 for GPU

* Pillow 3.4.2, scipy 0.18.1, numpy 1.11.2, ffmpeg 3.1.3  or later version

### Create Virtual Environment

In creating a virtual environment you will create a directory containing a python binary and everything needed to run PicassoColorfulWorld.

Firstly, you need to install virtualenvwrapper via

```
pip install virtualenvwrapper
```

Then you can create a virtual environment using this command:

```
virtualenv -p python3 $HOME/tmp/PicassoColorfulWorld-venv/
```

And activate the virtual environment like this 

```
source $HOME/tmp/PicassoColorfulWorld-venv/Scripts/activate
```

In that, you can isolate the working environments project by project.

So, please work on this virtual environment for the following installations.

### Installing

Change directory to PicassoColorfulWorld, where the git clone goes

```
cd PicassoColorfulWorld
```

We have already make all required packages listing in the requirements.txt, all you need to do is just to pip install the dependencies

```
pip install -r requirements.txt
```

Note: If your machine do not support nVidia GPU, please replace Tensorflow-gpu as Tensorflow inside the requirements.txt

### Get Dataset & VGG19

Before training, you need get dataset from [COCO](http://images.cocodataset.org/zips/test2014.zip) and VGG19 from [matconvnet](http://www.vlfeat.org/matconvnet/), or execute **setup.sh** to get dataset and VGG19

```
./setup.sh
 
```

### Run Pre-Trained Models

Now, you have all the packages for running the pre-trained models
You can have a trial run the starrynight style model that we have pre-trained, from the as following 

for example: you want to evaluate images in examples/content with starrynight-300-255-NHWC_nbc8_bs1_7e00_1e03_0.01the instruction is as here.
```
python evaluate.py --data-format NHWC --num-base-channels 4 --checkpoint tf-models/a_muse_1935-NHWC_nbc4_bs1_7e00_1e02_0.001 \
  --in-path examples/content  \
  --out-path examples/results  \
  --allow-different-dimensions
 
```

where
* --data-format: NHWC is for tensorflow sieries framework, NCHW is for non-tensorflow series, ex. ONNX that WinML required.
* --num-base-channels: it's used to reduce model size to improve inference time on Web, and other lower compute platform.
* --checkpoint: is the path where you place the pre-trained model checkpoint
* --in-path: is the path to input images, can be a folder or a file
* --out-path: is the path to output images, can be a folder or a file

## Training

Let's start to do the training

```
python style.py --data-format NHWC --num-base-channels 4 --style examples/style/a_muse.jpg \
  --checkpoint-dir ckpts \
  --test examples/content/butterfly.jpg \
  --test-dir examples/result \
  --content-weight 7e0 \
  --style-weight 1e02
  --checkpoint-iterations 1000 \
  --learning-rate 1e-3
  --batch-size 1
```

where

you need create a folder "ckpts" in the root of this project to save chackpoint files.
* --data-format: NHWC is for tensorflow sieries framework, NCHW is for non-tensorflow series, ex. ONNX that WinML required.
* --num-base-channels: it's used to reduce model size to improve inference time on Web, and other lower compute platform.
* --checkpoint-dir: is the path to save checkpoint in
* --style: style image path
* --train-path: path to training images folder
* --test: test image path
* --test-dir: test image save dir
* --epochs: number of epochs
* --batch-size: number of images feed for a batch
* --checkpoint-iterations: checkpoint save frequency
* --vgg-path: path to VGG19 network
* --content-weight: content weight
* --style-weight: style weight
* --tv-weight: total variation regularization weight
* --learning-rate: learning rate


## Evaluating

You can evaluate the trained models via

```
python evaluate.py --data-format NHWC --num-base-channels 4 --checkpoint tf-models/a_muse_1935-NHWC_nbc4_bs1_7e00_1e02_0.001 \
  --in-path examples/content/butterfly.jpg \
  --out-path examples/results/
```


## Tuning Parameters

In this practice, we offer 3 style similar level to let you experience the different style level. the are tuned by content-weight, style-weight, and learning-rate  

* --content-weight
* --style-weight
* --learning-rate


### Freeze Model

If you need get freeze model file, following the instruction that tensorflow bundled here
```
python -m tensorflow.python.tools.freeze_graph --input_graph=tf-model/starrynight-300-255-NHWC_nbc8_bs1_7e00_1e03_0.001/graph.pbtxt \
--input_checkpoint=tf-model/starrynight-300-255-NHWC_nbc8_bs1_7e00_1e03_0.001/saver \
--output_graph=tf-models/starrynight.pb --output_node_names="output"
```


## Implementation Details
The implementation is based on the [Fast Style Transfer in TensorFlow from ](https://github.com/lengstrom/fast-style-transfer) from [lengstrom](https://github.com/lengstrom/fast-style-transfer/commits?author=lengstrom)

Here are the source code for you practice on your local machine.
We also share some experience on how to fine tune the hyperparameter to gain a more suitable result of transfer target contents to as Picasso's style.<br>


Our implemetation is base on [fast-style-transfer](https://github.com/lengstrom/fast-style-transfer) 

Following are some example that training target style by parameters, content weight(cw), style weight(sw), and batch size: 1.  

<table><tr><td>content</td><td>Result</td><td>Picasso Style</td></tr>
<tr><td><img src = 'examples/content/Sidewalk.jpg' height = '180px'> <br> a sidewalk </td><td><img src='examples/results/a_muse_1935-NHWC_nbc4_bs1_7e00_1e02_0.001/Sidewalk.JPG' height='180px'><br>cw:7e0, sw:1e02</td><td><img src = 'examples/style/a_muse.jpg' height = '180px'><br>a muse 1935</td></tr>

<tr><td><img src = 'examples/content/Sidewalk.jpg' height = '180px'> <br> a sidewalk</td><td><img src = 'examples/results/bullfight_1934-NHWC_nbc4_bs1_7e00_1e02_0.001/Sidewalk.JPG' height = '180px'><br>cw:7e0, sw:1e02</td><td><img src = 'examples/style/bullfight.jpg' height = '180px'><br>bullfight 1934</td></tr>

<tr><td><img src = 'examples/content/Sidewalk.jpg' height = '180px'> <br> a sidewalk</td><td><img src = 'examples/results/crucifixion_1930-NHWC_nbc4_bs1_7e00_3e01_0.001/Sidewalk.JPG' height = '180px'><br>cw:7e0, sw:3e01</td><td><img src = 'examples/style/crucifixion.jpg' height = '180px'><br>crucifixion 1930</td></tr>

<tr><td><img src = 'examples/content/Sidewalk.jpg' height = '180px'> <br> a sidewalk</td><td><img src = 'examples/results/girl_in_front_of_mir_1932-NHWC_nbc4_bs1_7e00_5e01_0.001/Sidewalk.JPG' height = '180px'><br>cw:7e0, sw:5e01</td><td><img src = 'examples/style/girl_in_front_of_mir.jpg' height = '180px'><br>girl in front of mir 1932</td></tr>

<tr><td><img src = 'examples/content/Sidewalk.jpg' height = '180px'> <br> a sidewalk</td><td><img src = 'examples/results/boy_with_a_pipe_1905-NHWC_nbc4_bs1_7e00_1e02_0.001/Sidewalk.JPG' height = '180px'><br>cw:7e0, sw:1e02</td><td><img src = 'examples/style/boy_with_a_pipe.jpg' height = '180px'><br>boy with a pipe 1905</td></tr>

<tr><td><img src = 'examples/content/Sidewalk.jpg' height = '180px'> <br> a sidewalk</td><td><img src = 'examples/results/the_rescue_1932-NHWC_nbc4_bs1_7e00_1e02_0.001/Sidewalk.JPG' height = '180px'><br>cw:7e0, sw:1e02</td><td><img src = 'examples/style/the_rescue.jpg' height = '180px'><br>the rescue 1932</td></tr>

<tr><td><img src = 'examples/content/Sidewalk.jpg' height = '180px'> <br> a sidewalk</td><td><img src = 'examples/results/Les_Demoiselles_d_Avignon-NHWC_nbc4_bs1_7e00_1e02_0.001/Sidewalk.JPG' height = '180px'><br>cw:7e0, sw:1e02</td><td><img src = 'examples/style/Les_Demoiselles_d_Avignon.jpg' height = '180px'><br> Les Demoiselles d'Avignon</td></tr>

<tr><td><img src = 'examples/content/Sidewalk.jpg' height = '180px'> <br> a sidewalk</td><td><img src = 'examples/results/tomato_plant_1944-NHWC_nbc4_bs1_7e00_1e02_0.001/Sidewalk.JPG' height = '180px'><br>cw:7e0, sw:1e02</td><td><img src = 'examples/style/tomato_plant.jpg' height = '180px'><br>tomato plant 1944</td></tr>

<tr><td><img src = 'examples/content/Sidewalk.jpg' height = '180px'> <br> a sidewalk</td><td><img src = 'examples/results/Sleeping_Peasants_1919-NHWC_nbc4_bs1_7e00_7e01_0.001/Sidewalk.JPG' height = '180px'><br>cw:7e0, sw:7e01</td><td><img src = 'examples/style/Sleeping_Peasants.jpg' height = '180px'><br>Sleeping Peasants 1919 </td></tr>

</table>



## License

This project is licensed under the MIT, see the [LICENSE.md](LICENSE)

## Acknowledgments

Thanks all authors of following projects. 

* The source code of this practice is major borrowed from [fast-style-transfer](https://github.com/lengstrom/fast-style-transfer) Github repository.
* refer to some opinion in [Neural Style Transfer: A Review](https://arxiv.org/abs/1705.04058)


