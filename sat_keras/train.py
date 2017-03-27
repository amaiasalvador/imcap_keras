import numpy as np
import os
from model import get_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from args import get_parser
from utils.dataloader import DataLoader
from utils.config import get_opt, ResetStatesCallback
import sys
import pickle
import keras.backend as K

parser = get_parser()
args_dict = parser.parse_args()

print (args_dict)
print ("="*10)
# for reproducibility
np.random.seed(args_dict.seed)

sys.stdout = open(os.path.join('../logs/',args_dict.model_name+'_train.txt'),"w")

dataloader = DataLoader(args_dict)

N_train, N_val, N_test = dataloader.get_dataset_size()

print ('train',N_train,'val',N_val)

## DataLoaders
train_gen = dataloader.generator('train',args_dict.bs)
val_gen = dataloader.generator('val',args_dict.bs)

## Callbacks
model_name = os.path.join(args_dict.data_folder, 'models',
                          args_dict.model_name
                          +'_weights.{epoch:02d}-{val_loss:.2f}.h5')

ep = EarlyStopping(monitor='val_loss', patience=args_dict.pat,
                  verbose=0, mode='auto')

mc = ModelCheckpoint(model_name, monitor='val_loss', verbose=0,
                    save_best_only=True, mode='auto')

# reset states after each batch (bcs stateful)
rs = ResetStatesCallback()

#########################
###  Frozen Convnet   ###
#########################

# get model (frozen convnet)
model = get_model(args_dict)
opt = get_opt(args_dict)

model.compile(optimizer=opt,loss='categorical_crossentropy',
              sample_weight_mode="temporal")

#if args_dict.es_metric == 'loss':
model.fit_generator(train_gen,nb_epoch=args_dict.nepochs,
                        samples_per_epoch=N_train,
                        validation_data=val_gen,
                        nb_val_samples=N_val,
                        callbacks=[mc,ep,rs],
                        verbose = 1,
                        nb_worker = args_dict.workers,
                        pickle_safe = False)
'''
else:

    vocab_file = os.path.join(args_dict.data_folder,'data',args_dict.vfile)
    vocab = pickle.load(open(vocab_file,'rb'))

    for e in range(args_dict.nepochs):
        samples = 0
        for x,y,sw in train_gen:
            model.fit(x=x,y=y,sample_weight=sw,
                      batch_size=args_dict.bs,callbacks=[rs])
            train_samples+=args_dict.bs
            if samples >= N_train:
                break

        preds = model.predict_generator(val_gen,steps=(N_val/args_dict.bs))
        pred_idxs = np.argmax(preds,axis=-1)
        captions = idx2word(pred_idxs,vocab)
        for caption in captions:
            pred_cap = ' '.join(caption[:-1])# exclude eos
            captions.append({"image_id":imids[0]['id'],
                            "caption": pred_cap})

'''
#########################
### Fine Tune ConvNet ###
#########################

# init model again, unfreeze convnet, load weights and fine tune
print "Fine tuning convnet..."
args_dict.lr /= 100

#model = get_model(args_dict)
opt = get_opt(args_dict)

#'/work/asalvador/sat_keras/models/model_weights.10-2.57.h5'
#model.load_weights(model_name)

model_name = os.path.join(args_dict.data_folder, 'models',
                          args_dict.model_name + '_cnn_train'
                          +'_weights.{epoch:02d}-{val_loss:.2f}.h5')

for layer in model.layers[1].layers:
    layer.trainable = True

model.compile(optimizer=opt,loss='categorical_crossentropy',
              sample_weight_mode="temporal")

#if args_dict.es_metric == 'loss':
model.fit_generator(train_gen,nb_epoch=args_dict.nepochs,
                    samples_per_epoch=N_train,
                    validation_data=val_gen,
                    nb_val_samples=N_val,
                    callbacks=[mc,ep,rs],
                    verbose = 1,
                    nb_worker = args_dict.workers,
                    pickle_safe = False)
