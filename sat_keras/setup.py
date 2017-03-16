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
nltk.download('wordnet')
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

vocab_file = os.path.join(args_dict.data_folder,'data',args_dict.vfile)

if not os.path.isfile(vocab_file):
    print ('Creating word dictionary...')
    # loads training set only
    anns = load_caps(args_dict)
    words = topK(anns,args_dict)
    word2class = create_dict(words)
    print (len(word2class), 'most common words selected.')
    print('a',word2class['a'])
    print('kitchen',word2class['kitchen'])
    with open(vocab_file,'wb') as f:
        pickle.dump(word2class,f)

    print ('Done.')

dataloader = DataLoader(args_dict)
dataset_file = os.path.join(args_dict.data_folder,'data',args_dict.h5file)

if not os.path.isfile(dataset_file):
    print ('Creating dataset...')
    dataloader.write_hdf5()
    print ("Done.")

print ("Testing dataset...")

train_gen = dataloader.generator('train',200)

for [ims,prev_cap],caps,sw in train_gen:

    print (np.shape(ims))
    print (np.shape(caps))
    print (np.shape(prev_cap))
    print (prev_cap[1])
    break
