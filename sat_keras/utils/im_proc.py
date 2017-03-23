from scipy.misc import imread, imresize
import numpy as np

def read_image(impath,imsize):

    img = imread(impath,mode='RGB')
    if img.ndim == 2: # handling grayscale
        img = img[:,:,None][:,:,[0,0,0]]

    img = imresize(img, imsize)

    return img

def process_image(impath,imsize):
    img = imread(impath,mode='RGB')
    if img.ndim == 2: # handling grayscale
        img = img[:,:,None][:,:,[0,0,0]]
    H0, W0 = img.shape[0], img.shape[1]

    img = imresize(img, float(imsize) / min(H0, W0))

    return img

def center_crop(im, imsize):

    width, height,_ = im.shape   # Get dimensions

    left = int((width - imsize)/2)
    top = int((height - imsize)/2)
    #right = (width + imsize)/2
    #bottom = (height + imsize)/2

    return im[left:left+imsize,top:top+imsize,:]
