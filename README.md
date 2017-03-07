# Show, attend & tell in Keras

This is an implementation of the captioning model described in this paper using Keras with Tensorflow backend.

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

- Install [COCO tools](https://github.com/pdollar/coco)

## Data

- Download [MS COCO Caption Challenge 2015 dataset](http://mscoco.org/dataset/#captions-challenge2015)


## Usage

### Demo

### Training

### Testing

## References


Xu, Kelvin, et al. [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention.](http://www.jmlr.org/proceedings/papers/v37/xuc15.pdf) ICML 2015.

## Contact

amaia.salvador@upc.edu