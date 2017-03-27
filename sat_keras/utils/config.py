from keras.optimizers import SGD, RMSprop, Adam, Adadelta, Adagrad
from keras.callbacks import Callback
import numpy as np

def get_opt(args_dict):

    opt_name = args_dict.optim
    if opt_name is 'SGD':
        opt = SGD(lr=args_dict.lr, decay=args_dict.decay, momentum=0.9, nesterov=True,
                 clipvalue= args_dict.clip)
    elif opt_name is 'adam':
        opt = Adam(lr=args_dict.lr, decay= args_dict.decay,clipvalue = args_dict.clip)
    elif opt_name is 'adadelta':
        opt = Adadelta(lr=args_dict.lr, decay=args_dict.decay, clipvalue = args_dict.clip)
    elif opt_name is 'adagrad':
        opt = Adagrad(lr=args_dict.lr, decay=args_dict.decay, clipvalue = args_dict.clip)
    elif opt_name is 'rmsprop':
        opt = RMSprop(lr=args_dict.lr, decay=args_dict.decay, clipvalue = args_dict.clip)
    else:
        print ("Unknown optimizer! Using Adam by default...")
        opt = Adam(lr=args_dict.lr, decay=args_dict.decay, clipvalue = args_dict.clip)

    return opt

class ResetStatesCallback(Callback):
    def on_batch_end(self, batch, logs={}):
        self.model.reset_states()

'''

class EarlyStoppingCap(Callback):

    def __init__(self,args_dict):

        self.verbose = False
        self.save_best_only = True

        self.best_val_loss = np.inf
        self.val_loss = []

        self.val_metric = []
        self.best_val_metric = np.NINF


    def on_epoch_end(self,epoch,logs={}):


    def early_stop_decision(self,epoch,val_metric,val_loss):

        if val_loss < self.best_val_loss:
            self.wait = 0
        elif val_metric > self.best_val_metric:
            self.wait = 0
        else:
            self.wait +=1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
'''
