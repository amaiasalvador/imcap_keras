import nltk
from args import get_parser
import pickle
from utils.lang_proc import *
from utils.dataloader import DataLoader
import os

print ('Downloading nltk packages...')
# download language resource
nltk.download('punkt')
print ("Done.")

parser = get_parser()
args_dict = parser.parse_args()
vocab_file = os.path.join(args_dict.data_folder,'vocab.pkl')

if not os.path.isfile(vocab_file):
    print ('Creating word dictionary...')
    # loads training set only
    anns = load_caps(args_dict)
    words,maxlen = topK(anns,args_dict)

    word2class = create_dict(words)
    print (len(word2class), 'most common words selected.')
    print ('maximum sentence length',maxlen)
    with open(vocab_file,'wb') as f:
        pickle.dump(word2class,f)

    print ('Done.')

dataloader = DataLoader(args_dict)
dataset_file = os.path.join(args_dict.data_folder,'dataset.h5')

if not os.path.isfile(dataset_file):
    print ('Creating dataset...')
    dataloader.write_hdf5()
    print ("Done.")

print ("Testing dataset...")

train_gen = dataloader.generator('train',200)

for ims,caps in train_gen:

    print (np.shape(ims))
    print (np.shape(caps))
