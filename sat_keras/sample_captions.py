import numpy as np
import os
from args import get_parser
from utils.dataloader import DataLoader
from utils.config import get_opt
from utils.lang_proc import preds2cap
from model import get_model
import pickle

parser = get_parser()
args_dict = parser.parse_args()

model = get_model(args_dict)
opt = get_opt(args_dict)

weights = os.path.join(args_dict.data_folder, 'models',
                          args_dict.model_name +'_weights.h5')

model.load_weights(weights)

vocab_file = os.path.join(args_dict.data_folder,'data','vocab.pkl')
vocab = pickle.load(open(vocab_file,'rb'))
inv_vocab = {v['id']:k for k,v in vocab.items()}

model.compile(optimizer=opt,loss='categorical_crossentropy')

dataloader = DataLoader(args_dict)
N = 1
val_gen = dataloader.generator('val',batch_size=N,train_flag=False) # N samples

for ims,caps,imids in val_gen:
    preds = model.predict(ims)
    preds = np.argmax(preds,axis=2)
    pred_caps = preds2cap(preds,inv_vocab)
    print (preds2cap(np.argmax(caps,axis=2),inv_vocab))
    print ('='*10)
    print (pred_caps)
    print ('='*10)
    print (imids)
    break
