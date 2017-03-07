import nltk
from args import get_parser
import pickle
from utils.lang_proc import *
from utils.dataloader import DataLoader
import os

def make_dir(dir):
    if not os.path.exists(dir):
        print ("Creating directory",dir)
        os.mkdir(dir)

print ('Downloading nltk packages...')
# download language resource
nltk.download('punkt')
print ("Done.")

parser = get_parser()
args_dict = parser.parse_args()

# creating paths
make_dir(os.path.join(args_dict.data_folder))
make_dir(os.path.join(args_dict.data_folder,'data'))
make_dir(os.path.join(args_dict.data_folder,'models'))
make_dir(os.path.join(args_dict.data_folder,'history'))
make_dir(os.path.join(args_dict.data_folder,'results'))
make_dir('../logs')

vocab_file = os.path.join(args_dict.data_folder,'data','vocab.pkl')

if not os.path.isfile(vocab_file):
    print ('Creating word dictionary...')
    # loads training set only
    anns = load_caps(args_dict)
    words,maxlen,len_corpus = topK(anns,args_dict)
    word2class = create_dict(words,len_corpus)
    print (len(word2class), 'most common words selected.')
    print(word2class['the'])
    print(word2class['UNK'])
    print ('maximum sentence length',maxlen)
    with open(vocab_file,'wb') as f:
        pickle.dump(word2class,f)

    print ('Done.')

dataloader = DataLoader(args_dict)
dataset_file = os.path.join(args_dict.data_folder,'data','dataset.h5')

if not os.path.isfile(dataset_file):
    print ('Creating dataset...')
    dataloader.write_hdf5()
    print ("Done.")

print ("Testing dataset...")

train_gen = dataloader.generator('train',200)

for ims,caps in train_gen:

    print (np.shape(ims))
    print (np.shape(caps))
    break
