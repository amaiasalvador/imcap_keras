import numpy as np
import os
from args import get_parser
from utils.dataloader import DataLoader
from utils.config import get_opt
from utils.lang_proc import idx2word, sample
from model import get_model
import pickle

parser = get_parser()
args_dict = parser.parse_args()
args_dict.mode = 'test'

model = get_model(args_dict)
opt = get_opt(args_dict)

weights = args_dict.model_file
model.load_weights(weights)

vocab_file = os.path.join(args_dict.data_folder,'data',args_dict.vfile)
vocab = pickle.load(open(vocab_file,'rb'))
inv_vocab = {v:k for k,v in vocab.items()}

model.compile(optimizer=opt,loss='categorical_crossentropy')

dataloader = DataLoader(args_dict)
N = args_dict.bs
val_gen = dataloader.generator('val',batch_size=N,train_flag=False) # N samples

for ims,caps,imids in val_gen:

    prevs = np.zeros((N,1))
    word_idxs = np.zeros((N,args_dict.seqlen))

    for i in range(args_dict.seqlen):
        # get predictions
        preds = model.predict([ims,prevs]) #(N,1,vocab_size)
        preds = preds.squeeze()

        word_idxs[:,i] = np.argmax(preds,axis=-1)
        prevs = np.argmax(preds,axis=-1)

    pred_caps = idx2word(word_idxs,inv_vocab)
    true_caps = idx2word(np.argmax(caps,axis=-1),inv_vocab)

    for i in range(N):

        pred_cap = ' '.join(pred_caps[i])
        true_cap = ' '.join(true_caps[i])

        # true captions
        print ("ID:", imids[i]['file_name'])
        print ("True:", true_cap)
        print ("Gen:", pred_cap)
        print ("-"*10)

    model.reset_states()
    break
