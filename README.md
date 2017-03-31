**Work in progress**
# Image Captioning with Spatial Attention in Keras

This is a Keras & Tensorflow implementation of a captioning model. In particular, it uses the attention models described in [this](https://arxiv.org/abs/1612.01887) paper, which is depicted below:

<p align="center">
  <img src="figs/attmodel.png">
</p>

where V are the K local features from the last convolutional layer of a ConvNet (e.g. ResNet-50), x<sub>t</sub> is the input 
(composed of the embedding of the previous word and the average image feature). h<sub>t</sub> is the hidden state of the LSTM at time t,
which is used to compute the attention weights to apply to V in order to obtain the context vector c<sub>t</sub>. c<sub>t</sub> and h<sub>t</sub> are combined to predict the current word y<sub>t</sub>. 
In (b), an additional gate is incorporated into the LSTM to produce the additional s<sub>t</sub> output, which is combined with V to compute the attention weights. 
s<sub>t</sub> is used as an alternative feature to look at rather than the image features in V.

## Installation

- Clone this repository

```shell
# Make sure to clone with --recursive
git clone --recursive https://github.com/amaiasalvador/sat_keras.git
```

- Install [python 2.7](https://www.python.org/).
- Install [tensorflow 0.12](https://github.com/tensorflow/tensorflow/blob/r0.12/tensorflow/g3doc/get_started/os_setup.md).
- ```pip install -r requirements.txt```
- Set tensorflow as the keras backend in ```~/.keras/keras.json```:

```json
{
    "image_dim_ordering": "tf", 
    "epsilon": 1e-07, 
    "floatx": "float32", 
    "backend": "tensorflow"
}
```

## Data & Pretrained model

- Download [MS COCO Caption Challenge 2015 dataset](http://mscoco.org/dataset/#captions-challenge2015). Note that test images are not required for this code to work.
- After extraction, the dataset folder must have the following structure:

``` Shell
$coco/										# dataset dir
$coco/annotations/							# annotations directory
$coco/annotations/captions_train2014.json	# caption anns for training set
$coco/annotations/captions_val2014.json		# ...
$coco/images/								# image dir
$coco/images/train2014						# train image dir
$coco/images/val2014						# ...
```

- Prepare data with ```python setup.py -coco_path path/to/coco/dataset```. This involves loading and formatting images and captions to be stored in a single h5 file.
- [Coming soon] Download pretrained model [here]().

## Usage

Unless stated otherwise, run all commands from ```./imcap```:

### Demo

Run ```sample_captions.ipynb``` to test the trained network on some images and visualize attention maps.

### Training

Run ```python train.py```. Run ```python args.py --help``` for a list of the available arguments to pass.

### Testing

- Run ```python test.py``` to forward all validation images through a trained network and create json file with results.
- Navigate to ```./imcap/coco_caption/```. 
- From there run: 
  ```
  python eval_caps.py -results_file results.json -ann_file gt_file.json
  ``` 
  to get METEOR, Bleu, ROUGE_L & CIDEr scores for the previous json file with generated captions. 
  
### Note on used train/val/test splits

For the sake of comparison, data splits have been taken from the [original source code](https://github.com/kelvinxu/arctic-captions) of [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](http://arxiv.org/abs/1502.03044).

## References

- Xu et al. [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention.](http://www.jmlr.org/proceedings/papers/v37/xuc15.pdf) ICML 2015.
- Lu et al. [Knowing When to Look: Adaptive Attention via A Visual Sentinel for Image Captioning](https://arxiv.org/abs/1612.01887)
- Caption evaluation code from [this repository](https://github.com/tylin/coco-caption).

## Contact

For questions and suggestions either use the issues section or send an e-mail to amaia.salvador@upc.edu.
