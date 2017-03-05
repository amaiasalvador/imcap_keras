import pickle
import matplotlib.pylab as plt
from args import get_parser
import os
import numpy as np

def read_lines(txtfile):

    with open(txtfile,'r') as f:
        lines = f.readlines()
    return lines

def plot_curves_history(history):

    """
    Plots accuracy and loss curves given model history and number of epochs
    """

    nb_epoch = len(history)
    t = np.arange(0, nb_epoch, 1)

    plt.plot(t, history.history['loss'], 'r-*')
    plt.plot(t, history.history['val_loss'], 'b-*')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss','val_loss'], loc='upper right')

    plt.savefig(os.path.join('../logs/',args_dict.model_name+'_curves.png'))

def plot_curves_parser(lines):

    train_loss = []
    val_loss = []
    for line in lines:
        if 'val_loss' in line:
            sp = line.split(' - ')
            tr = sp[2].split('loss: ')[1]
            tr = float(tr.rstrip())

            va = sp[3].split('val_loss: ')[1]
            va = float(va.rstrip())

            train_loss.append(tr)
            val_loss.append(va)

    nb_epoch = len(train_loss)
    t = np.arange(0, nb_epoch, 1)

    plt.plot(t, train_loss, 'r-*')
    plt.plot(t, val_loss, 'b-*')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss','val_loss'], loc='upper right')

    plt.savefig(os.path.join('../logs/',args_dict.model_name+'_curves.png'))


if __name__ == "__main__":

    parser = get_parser()
    args_dict = parser.parse_args()
    #history_file = os.path.join(args_dict.data_folder, 'history',
    #                          args_dict.model_name +'_history.pkl')


    txtfile = os.path.join('../logs/',args_dict.model_name+'_train.txt')
    lines = read_lines(txtfile)
    plot_curves_parser(lines)
