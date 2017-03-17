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

model = get_model(args_dict)
opt = get_opt(args_dict)

model.compile(optimizer=opt,loss='categorical_crossentropy',
              sample_weight_mode="temporal")

dataloader = DataLoader(args_dict)

N_train, N_val, N_test = dataloader.get_dataset_size()

print ('train',N_train,'val',N_val)
train_gen = dataloader.generator('train',args_dict.bs)
val_gen = dataloader.generator('val',args_dict.bs)

# Callbacks
model_name = os.path.join(args_dict.data_folder, 'models',
                          args_dict.model_name
                          +'_weights.{epoch:02d}-{loss:.2f}.h5')

#ep = EarlyStopping(monitor='val_loss', patience=args_dict.pat,
#                  verbose=0, mode='auto')

mc = ModelCheckpoint(model_name, monitor='loss', verbose=0,
                    save_best_only=True, mode='auto')

# reset states after each batch
rs = ResetStatesCallback()
history = model.fit_generator(train_gen,nb_epoch=args_dict.nepochs,
                            samples_per_epoch=N_train,
                            validation_data=val_gen,
                            nb_val_samples=N_val,
                            callbacks=[mc,rs],
                            verbose = 1,
                            nb_worker = args_dict.workers,
                            pickle_safe = False)
