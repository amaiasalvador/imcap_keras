from keras.optimizers import SGD, RMSprop, Adam, Adadelta, Adagrad


def get_opt(args_dict):

    opt_name = args_dict.optim
    if opt_name is 'SGD':
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
        print ("Unknown optimizer! Using Adam by default...")
        opt = Adam(lr=args_dict.lr)

    return opt
