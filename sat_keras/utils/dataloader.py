from utils.im_proc import process_image, center_crop
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.np_utils import to_categorical
import numpy as np
import os
import h5py
import nltk
import pickle
import json
from tqdm import *
import random
import threading

class DataLoader(object):

    def __init__(self,args_dict):

        self.data_folder = args_dict.data_folder
        self.coco = args_dict.coco_path
        self.year = args_dict.year
        self.resize = args_dict.resize
        self.imsize = args_dict.imsize
        self.seqlen = args_dict.seqlen
        self.n_caps = args_dict.n_caps
        self.vocab_size = args_dict.vocab_size
        vocab_file = os.path.join(args_dict.data_folder,'data','vocab.pkl')
        self.vocab = pickle.load(open(vocab_file,'rb'))

    def get_anns(self,partition):

        '''
        Returns variable 'ims', which contains all image filenames, and
        'imid2caps', which gives the captions for each image.
        '''
        ann_file = os.path.join(self.coco,'annotations','captions_' + partition
                                + self.year+'.json')
        anns = json.load(open(ann_file))
        imid2caps = {}
        for item in anns['annotations']:
             if item['image_id'] in imid2caps:
                 imid2caps[item['image_id']].append(item['caption'])
             else:
                 imid2caps[item['image_id']] = [item['caption']]

        return anns['images'],imid2caps


    def word2class(self,captions):
        '''
        Formats raw captions for an image into sequence of class ids
        '''
        caption_cls_set = np.zeros((self.n_caps,self.seqlen))
        for i in range(self.n_caps):
            caption = captions[i]
            tok_caption = nltk.word_tokenize(caption.lower())
            tok_caption = tok_caption[:self.seqlen-1]
            tok_caption.append('<eos>')

            caption_cls = np.zeros((self.seqlen,))
            for j,x in enumerate(tok_caption):
                word = self.vocab.get(x)
                if word:
                    caption_cls[j] = word['id']
                else:
                    caption_cls[j] = self.vocab['UNK']['id'] # UNK class

            caption_cls_set[i] = caption_cls
        return caption_cls_set

    def write_hdf5(self):

        '''
        Reads dataset (images & captions) and stores it in a HDF5 file
        '''

        with h5py.File(os.path.join(self.data_folder,'data','dataset.h5')) as f_ds:

            for part in ['train','val']:

                ims_path = os.path.join(self.coco,'images',part+self.year)
                ims,imid2caps  = self.get_anns(part)
                nims = len(ims)

                # image data
                idata = f_ds.create_dataset('ims_%s'%(part),(nims,
                                                  self.imsize,self.imsize,3),
                                                  dtype=np.uint8)
                # caption data
                cdata = f_ds.create_dataset('caps_%s'%(part),(nims,self.n_caps,
                                                            self.seqlen))


                for i,im in tqdm(enumerate(ims)):

                    caps = imid2caps[im['id']]
                    caps_labels = self.word2class(caps)
                    filename = im['file_name']

                    img = process_image(os.path.join(ims_path,filename),
                                            self.resize)
                    img = center_crop(img, self.imsize)
                    idata[i,:,:,:] = img
                    cdata[i,:,:] = caps_labels


    def get_dataset_size(self):
        '''
        Returns number of training and validation samples
        '''
        train_ims, _ = self.get_anns('train')
        val_ims, _ = self.get_anns('val')

        return len(train_ims), len(val_ims)

    def generator(self,partition,batch_size,train_flag=True):

        imlist,_ = self.get_anns(partition)

        data = h5py.File(os.path.join(self.data_folder,'data','dataset.h5'),'r')
        ims = data['ims_%s'%(partition)]
        caps = data['caps_%s'%(partition)]

        idxs = np.array(range(len(imlist)))

        rng = int(np.ceil(len(imlist)/batch_size))

        while True:

            # shuffle samples every epoch
            if train_flag:
                random.shuffle(idxs)

            for i in range(rng):

                # adjust batch size at the end
                if i*batch_size + batch_size > len(imlist):
                    bs = len(imlist) - i*batch_size
                    sample_ids = idxs[i*batch_size:]
                else:
                    bs = batch_size
                    sample_ids = idxs[i*bs:i*bs+bs]

                batch_idxs = np.sort(sample_ids)

                # load images
                batch_ims = ims[batch_idxs,:,:,:]
                batch_ims = batch_ims.astype(np.float64)
                # handle case where batch_size = 1
                batch_ims = np.reshape(batch_ims,(bs,self.imsize,self.imsize,3))
                batch_ims = preprocess_input(batch_ims)

                # load captions
                # random caption id out of 5 possible ones
                if train_flag:
                    cap_id = random.randint(0,self.n_caps-1)
                else:
                    cap_id = 0
                batch_caps = caps[batch_idxs,cap_id,:]

                # ignore 0 samples
                sample_weight = np.zeros((bs,self.seqlen))
                sample_weight[batch_caps>0] = 1

                batch_caps = to_categorical(batch_caps,self.vocab_size + 2)
                batch_caps = np.reshape(batch_caps,(bs,self.seqlen,
                                        self.vocab_size + 2))

                if train_flag:
                    yield batch_ims,batch_caps,sample_weight
                else:
                    yield batch_ims,batch_caps,np.array(imlist)[batch_idxs]
