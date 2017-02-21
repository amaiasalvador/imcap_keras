from args import get_parser
from utils.im_proc import process_image, center_crop
import numpy as np
import os
import h5py
import nltk
import pickle
import json
from tqdm import *

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
        vocab_file = os.path.join(args_dict.data_folder,'vocab.pkl')
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

        caption_labels_set = np.zeros((self.n_caps,self.seqlen))
        for i in range(self.n_caps):
            caption = captions[i]
            caption = caption[:self.seqlen]
            tok_caption = nltk.word_tokenize(caption.lower())

            for j,x in enumerate(tok_caption):
                if x in self.vocab.keys():
                    caption_labels_set[i,j] = self.vocab[x]
                else:
                    caption_labels_set[i,j] = self.vocab_size+1 # UNK class

            return caption_labels_set

    def write_hdf5(self):

        with h5py.File(os.path.join(self.data_folder,'dataset.h5')) as f_ds:

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
