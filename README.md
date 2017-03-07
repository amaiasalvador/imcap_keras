# Show, attend & tell in Keras

This is a Keras & Tensorflow implementation of the captioning model described in this paper:

Xu, Kelvin, et al. [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention.](http://www.jmlr.org/proceedings/papers/v37/xuc15.pdf) ICML 2015.

## Installation

- Install [python 3.x](https://www.python.org/)
- Install [tensorflow 0.10](https://github.com/tensorflow/tensorflow/blob/r0.10/tensorflow/g3doc/get_started/os_setup.md)
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

- Download [MS COCO Caption Challenge 2015 dataset](http://mscoco.org/dataset/#captions-challenge2015)
- Download pretrained model [here].

## Usage

### Demo

Run ```sample_captions``` to test the trained network on some validation images.

### Training

- Prepare data with ```python setup.py```. 
- Run ```python train.py```. Run ```python args.py --help``` for a list of available parameters.

### Testing

- Run ```python test.py``` to forward all validation images through a trained network and create json file with results
- Run ```python eval.py``` to get METEOR, Bleu, ROUGE_L & CIDEr scores for the previous json file with generated captions.

## References

- Xu, Kelvin, et al. [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention.](http://www.jmlr.org/proceedings/papers/v37/xuc15.pdf) ICML 2015.
- Attention LSTM implementation adapted from [this gist file](https://gist.github.com/mbollmann/ccc735366221e4dba9f89d2aab86da1e).
- Caption evaluation code from [this repository](https://github.com/tylin/coco-caption).
## Contact

For questions and suggestions either use the issues section or send an e-mail to amaia.salvador@upc.edu.
