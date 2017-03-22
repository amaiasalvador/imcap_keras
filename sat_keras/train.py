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

#########################
###  Frozen Convnet   ###
#########################

# get model (frozen convnet)
model = get_model(args_dict)
opt = get_opt(args_dict)

model.compile(optimizer=opt,loss='categorical_crossentropy',
              sample_weight_mode="temporal")

# reset states after each batch (bcs stateful)
rs = ResetStatesCallback()
model.fit_generator(train_gen,nb_epoch=args_dict.nepochs,
                    samples_per_epoch=N_train,
                    validation_data=val_gen,
                    nb_val_samples=N_val,
                    callbacks=[mc,ep,rs],
                    verbose = 1,
                    nb_worker = args_dict.workers,
                    pickle_safe = False)

#########################
### Fine Tune ConvNet ###
#########################

# init model again, unfreeze convnet, load weights and fine tune
print "Fine tuning convnet..."
args_dict.lr = 1e-5

model = get_model(args_dict)
opt = get_opt(args_dict)

#'/work/asalvador/sat_keras/models/model_weights.10-2.57.h5'
model.load_weights(model_name)

model_name = os.path.join(args_dict.data_folder, 'models',
                          args_dict.model_name + '_cnn_train'
                          +'_weights.{epoch:02d}-{val_loss:.2f}.h5')

for layer in model.layers[1].layers:
    layer.trainable = True

model.compile(optimizer=opt,loss='categorical_crossentropy',
              sample_weight_mode="temporal")

rs = ResetStatesCallback()
model.fit_generator(train_gen,nb_epoch=args_dict.nepochs,
                    samples_per_epoch=N_train,
                    validation_data=val_gen,
                    nb_val_samples=N_val,
                    callbacks=[mc,ep,rs],
                    verbose = 1,
                    nb_worker = args_dict.workers,
                    pickle_safe = False)
