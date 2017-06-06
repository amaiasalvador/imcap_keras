import pickle
import matplotlib.pylab as plt
from args import get_parser
import os
import numpy as np

def read_lines(txtfile):

    with open(txtfile,'r') as f:
        lines = f.readlines()
    return lines

def plot_curves_parser(args,lines):

    train_loss = []
    val_loss = []
    metrics = []
    for line in lines:
        if 'val_loss' in line:
            sp = line.split(' - ')

            tr = sp[2].split('loss: ')[1]
            tr = float(tr.rstrip())

            va = sp[3].split('val_loss: ')[1]
            va = float(va.rstrip())

            if len(sp) > 4:
                me = sp[4].split(args.es_metric+ ': ')[1]
                metrics.append(me)
            train_loss.append(tr)
            val_loss.append(va)

    nb_epoch = len(train_loss)
    t = np.arange(0, nb_epoch, 1)
    fig, ax1 = plt.subplots()

    ax1.plot(t, train_loss, 'r-*')
    ax1.plot(t, val_loss, 'r--*')
    ax1.set_ylabel('loss',color='r')
    ax1.set_xlabel('epoch')
    ax1.tick_params('y', colors='r')
    ax1.legend(['train_loss','val_loss'], loc='upper center')

    if len(metrics) > 0:
        ax2 = ax1.twinx()
        ax2.plot(t,metrics,'b-*')
        ax2.set_ylabel('val_'+args.es_metric, color='b')
        ax2.tick_params('y', colors='b')

    plt.savefig(os.path.join('../logs/',args.model_name+'_curves.png'))


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    txtfile = os.path.join('../logs/',args.model_name+'_train.log')
    lines = read_lines(txtfile)
    plot_curves_parser(args,lines)
    pngfile = os.path.join('../logs/',args.model_name+'_curves.png')
    print ("Figure saved in %s"%(pngfile))
