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
        self.num_val = args_dict.num_val
        self.num_test = args_dict.num_test
        self.h5file = args_dict.h5file
        self.year = args_dict.year
        self.resize = args_dict.resize
        self.imsize = args_dict.imsize
        self.seqlen = args_dict.seqlen
        self.n_caps = args_dict.n_caps
        self.vocab_size = args_dict.vocab_size
        self.vfile = args_dict.vfile
        vocab_file = os.path.join(args_dict.data_folder,'data',self.vfile)
        self.vocab = pickle.load(open(vocab_file,'rb'))

    def get_anns(self,partition):

        '''
        Returns variable 'ims', which contains all image filenames, and
        'imid2caps', which gives the captions for each image.
        '''

        # test split comes from validation partition.
        if partition == 'test':
            part = 'val'
        else:
            part = partition

        json_file = os.path.join(self.coco,'annotations','captions_' + part
                                + self.year+'.json')

        data = json.load(open(json_file))
        ims = data['images']
        anns = data['annotations']

        # val: first 5k val images, test: second 5k images
        if partition == 'test':
            ims = ims[self.num_val:self.num_val + self.num_test]
        elif partition == 'val':
            ims = ims[0:self.num_val]

        # dictionary mapping image id with the list of N captions for that image
        imid2caps = {}
        for item in anns:
             if item['image_id'] in imid2caps:
                 imid2caps[item['image_id']].append(item['caption'])
             else:
                 imid2caps[item['image_id']] = [item['caption']]

        return ims,imid2caps

    def word2class(self,captions):
        '''
        Formats raw captions for an image into sequence of class ids
        '''

        caption_cls_set = np.zeros((self.n_caps,self.seqlen))
        for i in range(self.n_caps):
            # basic sentence formatting, tokenize & lower capital letters
            caption = captions[i]
            tok_caption = nltk.word_tokenize(caption.lower())
            # seqlen - 1 because of start and end of sequence
            tok_caption = tok_caption[:self.seqlen-2]

            out_cap = ['<start>']
            out_cap.extend(tok_caption)
            out_cap.append('<eos>')

            caption_cls = np.zeros((self.seqlen,))
            for j,x in enumerate(out_cap):
                word = self.vocab.get(x)
                if word:
                    caption_cls[j] = word
                else:
                    caption_cls[j] = self.vocab['<unk>'] # UNK class

            caption_cls_set[i] = caption_cls
        return caption_cls_set

    def write_hdf5(self):

        '''
        Reads dataset (images & captions) and stores it in a HDF5 file
        '''

        with h5py.File(os.path.join(self.data_folder,'data',self.h5file)) as f_ds:

            for part in ['test','val','train']:
                # offline test set included in validation set
                if part == 'test':
                    ims_path = os.path.join(self.coco,'images','val'+self.year)
                else:
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
                    # get captions for each image
                    caps = imid2caps[im['id']]
                    caps_labels = self.word2class(caps)

                    # load and preprocess image (resize + center crop)
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
        test_ims, _ = self.get_anns('test')

        return len(train_ims), len(val_ims), len(test_ims)

    def generator(self,partition,batch_size,train_flag=True):

        '''
        Generator function to yield batches for training & testing the network
        '''

        imlist,_ = self.get_anns(partition)

        data = h5py.File(os.path.join(self.data_folder,'data',self.h5file),'r')
        ims = data['ims_%s'%(partition)]
        caps = data['caps_%s'%(partition)]

        idxs = np.array(range(len(imlist)))

        # batches need to have same number of samples (due to stateful rnn)
        # note that some samples will be left out but they will change at each
        # epoch due to shuffling

        rng = int(np.floor(len(imlist)/batch_size))

        while True:

            # shuffle samples every epoch
            if train_flag:
                random.shuffle(idxs)

            for i in range(rng):

                # adjust batch size at the end
                # note: currently we never fall into the else condition here
                if i*batch_size + batch_size > len(imlist):
                    bs = len(imlist) - i*batch_size
                    sample_ids = idxs[i*batch_size:]
                else:
                    bs = batch_size
                    sample_ids = idxs[i*bs:i*bs+bs]

                # idxs need to be in ascending order
                batch_idxs = np.sort(sample_ids)

                batch_ims = ims[batch_idxs,:,:,:]
                batch_ims = batch_ims.astype(np.float64)

                # handle case where batch_size = 1
                batch_ims = np.reshape(batch_ims,(bs,self.imsize,self.imsize,3))
                batch_ims = preprocess_input(batch_ims)

                # load captions
                # random caption id out of 5 possible ones if training
                if train_flag:
                    cap_id = random.randint(0,self.n_caps-1)
                else:
                    cap_id = 0
                batch_caps = caps[batch_idxs,cap_id,:]

                # vector of previous words is simply a shifted version of
                # current words vector
                prev_caps = np.zeros((bs,self.seqlen))
                prev_caps[1:] = batch_caps[:-1]

                # words with 0 id are padded out
                sample_weight = np.zeros((bs,self.seqlen))
                sample_weight[batch_caps>0] = 1

                # for current words (to be predicted), we need to pass one-hot vecs
                batch_caps = to_categorical(batch_caps,self.vocab_size + 4)
                batch_caps = np.reshape(batch_caps,(bs,self.seqlen,
                                        self.vocab_size + 4))

                # during testing we don't return previous words(they will be
                # generated instead), and return image ids.
                if train_flag:
                    yield [batch_ims,prev_caps],batch_caps,sample_weight
                else:
                    yield batch_ims,batch_caps,np.array(imlist)[batch_idxs]
