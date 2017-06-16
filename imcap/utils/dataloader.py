from utils.im_proc import read_image
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

class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

class DataLoader(object):

    def __init__(self,args_dict):

        self.data_folder = args_dict.data_folder
        self.h5file = args_dict.h5file
        self.json_file = args_dict.json_file
        self.imsize = args_dict.imsize

    def get_splits_and_vocab(self):
        data = json.load(open(os.path.join(self.data_folder,'data',self.json_file),'r'))

        splits = {'train':[],'val':[],'test':[]}
        idxs = {'train':[],'val':[],'test':[]}

        for i,img in enumerate(data['images']):
            splits[img['split']].append(i)
            idxs[img['split']].append(img['id'])

        vocab = data['ix_to_word']
        #vocab_int = {}
        vocab_int_inv = {}
        for idx in vocab.keys():
            #vocab_int[int(idx)] = vocab[idx]
            vocab_int_inv[vocab[idx]] = int(idx)
        return splits, idxs, vocab_int_inv

    def get_dataset_size(self):
        splits, _, vocab = self.get_splits_and_vocab()

        return len(splits['train']), len(splits['val']), len(splits['test'])

    @threadsafe_generator
    def generator(self,partition,batch_size,train_flag=True):

        '''
        Generator function to yield batches for training & testing the network
        '''
        splits, image_ids, vocab = self.get_splits_and_vocab()
        sample_list = splits[partition]
        image_ids = image_ids[partition]
        vocab_size = len(vocab.keys())

        data = h5py.File(os.path.join(self.data_folder,'data',self.h5file),'r')
        ims = data['images']
        caps = data['labels']
        starts = data['label_start_ix']
        ends = data['label_end_ix']
        labels_length = data['label_length']

        idxs = np.array(range(len(sample_list)))

        # batches need to have same number of samples (due to stateful rnn)
        # note that some samples will be left out but they will change at each
        # epoch due to shuffling

        rng = int(np.floor(len(sample_list)/batch_size))

        while True:

            # shuffle samples every epoch
            if train_flag:
                random.shuffle(idxs)

            for i in range(rng):

                sample_ids = idxs[i*batch_size:i*batch_size+batch_size]
                # idxs need to be in ascending order
                batch_idxs = np.sort(sample_ids)
                batch_idxs = batch_idxs.tolist()

                # load samples to compose batch
                ### images ###
                batch_ims = ims[batch_idxs,:,:,:]
                batch_ims = batch_ims.astype(np.float64)

                # batch_size, width,height,3
                batch_ims = np.rollaxis(batch_ims,-1,1)
                batch_ims = np.rollaxis(batch_ims,-1,1)

                # handle case where batch_size = 1
                batch_ims = np.reshape(batch_ims,(batch_size,self.imsize,self.imsize,3))
                batch_ims = preprocess_input(batch_ims)

                ### load captions ###
                label_start_ixs = starts[batch_idxs]
                label_end_ixs = ends[batch_idxs]

                # random caption id out of 5 possible ones if training
                if train_flag:
                    cap_id = np.zeros((batch_size,))
                    # select random caption out of available ones for that image
                    for ix in range(batch_size):
                        cap_id[ix] = random.randint(label_start_ixs[ix]-1,label_end_ixs[ix]-1)
                else:
                    # take first caption
                    cap_id = label_start_ixs - 1
                cap_id = cap_id.tolist()
                label_length = labels_length[cap_id]
                prev_caps = caps[cap_id,:]
                seqlen = np.shape(prev_caps)[-1]
                prev_caps = np.reshape(prev_caps,(batch_size,seqlen))

                # vector of previous words is simply a shifted version of
                # current words vector
                batch_caps = np.zeros((batch_size,seqlen))
                batch_caps[:,:-1] = prev_caps[:,1:]

                # words with 0 id are padded out
                sample_weight = np.zeros((batch_size,seqlen))
                sample_weight[batch_caps>0] = 1

                # for current words (to be predicted), we need to pass one-hot vecs
                batch_caps = to_categorical(batch_caps,vocab_size+1)
                batch_caps = np.reshape(batch_caps,(batch_size,seqlen,vocab_size+1))

                # during testing we don't return previous words(they will be
                # generated instead), and return image ids.
                if train_flag:
                    yield [batch_ims,prev_caps],batch_caps,sample_weight
                else:
                    yield [batch_ims,prev_caps],batch_caps,sample_weight, np.array(image_ids)[batch_idxs]
