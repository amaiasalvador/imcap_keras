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

#dataset_file = os.path.join(args_dict.data_folder,'dataset.h5')
print ('Creating dataset...')
dataloader = DataLoader(args_dict)
dataloader.write_hdf5()
