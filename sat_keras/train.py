import numpy as np
from model import get_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from args import get_args
from utils.dataloader import DataLoader
from keras.optimizers import SGD, RMSprop, Adam, Adadelta, Adagrad
np.random.seed(1337)

def get_opt(args_dict):

    opt_name = args_dict.optim
    if opt_name is 'sgd':
        opt = SGD(lr=args_dict.lr, decay=1e-6, momentum=0.9, nesterov=True)
    elif opt_name is 'adam':
        opt = Adam(lr=args_dict.lr)
    elif opt_name is 'adadelta':
        opt = Adadelta(lr=args_dict.lr)
    elif opt_name is 'adagrad':
        opt = Adagrad(lr=args_dict.lr)
    elif opt_name is 'rmsprop':
        opt = RMSprop(lr=args_dict.lr)
    else:
        print "Unknown optimizer! Using RMSprop by default..."
        opt = RMSprop(lr=args_dict.lr)

    return opt

parser = get_parser()
args_dict = parser.parse_args()

# Unknowk class has weight = 0
class_weight = {}
for cls in range(args.n_classes+1):
    class_weight[cls] = 1.0
class_weight[args.n_classes] = 0.0

model = get_model(args_dict)
opt = get_opt(args_dict)
model.compile(optimizer=opt,loss='categorical_crossentropy',
              class_weight = class_weight)

dataloader = DataLoader(args_dict)

N_train, N_val = dataloader.get_dataset_size()

train_gen = dataloader.generator('train',args.bs)
val_gen = dataloader.generator('val',args.bs)

# Callbacks
model_name = os.path.join(args_dict.data_path, 'weights.h5')
ep = EarlyStopping(monitor='val_loss', patience=args_dict.pat,
                  verbose=0, mode='auto')
mc = ModelCheckpoint(args.data_path, monitor='val_loss', verbose=0,
                    save_best_only=True, mode='auto')

model.fit_generator(train_gen,nb_epoch=args_dict.nepochs,
                    samples_per_epoch=N_train,
                    validation_data=val_gen,
                    nb_val_samples=N_val,
                    callbacks=[ep,mc])
