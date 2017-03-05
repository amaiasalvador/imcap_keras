import numpy as np
import os
from model import get_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from args import get_parser
from utils.dataloader import DataLoader
from utils.config import get_opt
import sys
import pickle

parser = get_parser()
args_dict = parser.parse_args()

np.random.seed(args_dict.seed)

sys.stdout = open(os.path.join('../logs/',args_dict.model_name+'_train.txt'),"w")

# Unknowk class has weight = 0
class_weight = {}
for cls in range(args_dict.vocab_size+1):
    class_weight[cls] = 1.0
class_weight[args_dict.vocab_size] = 0.1

model = get_model(args_dict)
opt = get_opt(args_dict)
model.compile(optimizer=opt,loss='categorical_crossentropy',
              class_weight = class_weight)

dataloader = DataLoader(args_dict)

N_train, N_val = dataloader.get_dataset_size()

train_gen = dataloader.generator('train',args_dict.bs)
val_gen = dataloader.generator('val',args_dict.bs)

# Callbacks
model_name = os.path.join(args_dict.data_folder, 'models',
                          args_dict.model_name +'_weights.h5')
ep = EarlyStopping(monitor='val_loss', patience=args_dict.pat,
                  verbose=0, mode='auto')
mc = ModelCheckpoint(model_name, monitor='val_loss', verbose=0,
                    save_best_only=True, mode='auto')

history = model.fit_generator(train_gen,nb_epoch=args_dict.nepochs,
                            samples_per_epoch=N_train,
                            validation_data=val_gen,
                            nb_val_samples=N_val,
                            callbacks=[ep,mc],
                            verbose = 1)

history_file = os.path.join(args_dict.data_folder, 'history',
                          args_dict.model_name +'_history.pkl')
pickle.dump(history,open(history_file,'wb'))
