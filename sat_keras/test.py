import numpy as np
import os
from args import get_parser
from utils.dataloader import DataLoader
from utils.config import get_opt
from utils.lang_proc import preds2cap, sample
from model import get_model
import pickle
import json

parser = get_parser()
args_dict = parser.parse_args()

model = get_model(args_dict)
opt = get_opt(args_dict)

weights = args.model_file

model.load_weights(weights)

vocab_file = os.path.join(args_dict.data_folder,'data','vocab.pkl')
vocab = pickle.load(open(vocab_file,'rb'))
inv_vocab = {v['id']:k for k,v in vocab.items()}

model.compile(optimizer=opt,loss='categorical_crossentropy')

dataloader = DataLoader(args_dict)
val_gen = dataloader.generator('val',batch_size=args_dict.bs,train_flag=False)

captions = []
for ims,caps,imids in val_gen:
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
        caption = ' '.join(pred_cap)

        captions.append({"image_id":imids[i]['id'],
                         "caption": caption})

results_file = os.path.join(args_dict.data_folder, 'results',
                          args_dict.model_name +'_gencaps.json')
with open(results_file, 'w') as outfile:
    json.dump(captions, outfile)
