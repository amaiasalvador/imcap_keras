import numpy as np
import os
from args import get_parser
from utils.dataloader import DataLoader
from utils.config import get_opt
from utils.lang_proc import preds2cap, sample
from model import get_model
import pickle

parser = get_parser()
args_dict = parser.parse_args()

model = get_model(args_dict)
opt = get_opt(args_dict)

weights = args_dict.model_file
model.load_weights(weights)

vocab_file = os.path.join(args_dict.data_folder,'data','vocab.pkl')
vocab = pickle.load(open(vocab_file,'rb'))
inv_vocab = {v['id']:k for k,v in vocab.items()}

model.compile(optimizer=opt,loss='categorical_crossentropy')

dataloader = DataLoader(args_dict)
N = 20
val_gen = dataloader.generator('train',batch_size=N,train_flag=False) # N samples

for ims,caps,imids in val_gen:

    # get predictions
    preds = model.predict(ims)

    for i in range(preds.shape[0]): # for each sample
        word_idxs = []
        for j in range(preds.shape[1]): # for each word
            # sample word idx from distribution
            idx = sample(preds[i,j,:],temperature = args_dict.temperature)
            word_idxs.append(idx)
        word_idxs = np.array(word_idxs)

        # predicted captions
        pred_cap = preds2cap(word_idxs,inv_vocab)
        pred_cap = ' '.join(pred_cap)
        true_cap = preds2cap(np.argmax(caps[i],axis=-1),inv_vocab)
        true_cap = ' '.join(true_cap)

        # true captions
        print ("ID:", imids[i]['file_name'])
        print ("True:", true_cap)
        print ("Gen:", pred_cap)
        print ("-"*10)
    break
