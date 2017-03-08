import argparse

def get_parser():

    parser = argparse.ArgumentParser(description='SAT_keras',epilog='The end.')
    parser.add_argument('-model_name', dest='model_name',
                        default = 'model',help='Base name to save model')
    parser.add_argument('-model_file', dest='model_file',
                        default = 'model_file',help='path to model file to load.')
    parser.add_argument('-cnn', dest='cnn',
                        default = 'vgg16', choices=['vgg16','vgg19','resnet'],
                        help='Pretrained CNN to use')
    parser.add_argument('-coco_path', dest='coco_path',
                        default = '/seq/segmentation/COCO/tools',
                        help='COCO database')
    parser.add_argument('-year', dest='year',
                        default = '2014',help='COCO year')
    parser.add_argument('-resize', dest='resize',
                        default = 256, help='Image resize',type=int)
    parser.add_argument('-imsize', dest='imsize',
                        default = 224, help='Image crop size',type=int)
    parser.add_argument('-vocab_size', dest='vocab_size',
                        default = 10000, help='Vocabulary size' ,type=int)
    parser.add_argument('-n_caps', dest='n_caps',
                        default = 5, help='Number of captions for training',
                        type=int)
    parser.add_argument('-data_folder', dest='data_folder',
                        default = '/work/asalvador/sat_keras/',
                        help='save folder')
    # Model parameters
    parser.add_argument('-seqlen',dest='seqlen', default = 20,
                        help='Maximum sentence length',type=int)
    parser.add_argument('-lstm_dim',dest='lstm_dim', default = 256,
                        help='Number of LSTM units',type=int)
    parser.add_argument('-fc_dim',dest='fc_dim', default = 256,
                        help='Number of FC layer units',type=int)
    parser.add_argument('-dr_ratio',dest='dr_ratio', default = 0.5,
                        help='Dropout ratio',type=int)

    # Training params
    parser.add_argument('-seed', dest='seed',
                        default = 4242, help='Random seed',type=int)
    parser.add_argument('-bs',dest='bs', default = 32,
                            help='Batch Size',type=int)
    parser.add_argument('-pat',dest='pat', default = 5,
                            help='Patience',type=int)
    parser.add_argument('-lr',dest='lr', default = 0.001,
                                help='Learning rate')
    parser.add_argument('-optim',dest='optim', default ='adam',
                                choices=['adam','SGD','adadelta','adagrad',
                                'rmsprop'], help='Optimizer')
    parser.add_argument('-nepochs',dest='nepochs', default = 40,
                        help='Number of train epochs',type=int)
    # bools
    parser.add_argument('--cnnfreeze', dest='cnn_train', action='store_false')
    parser.add_argument('--cnntrain', dest='cnn_train', action='store_true')
    parser.set_defaults(cnn_train=False)

    return parser

if __name__ =="__main__":

    parser = get_parser()
    args_dict = parser.parse_args()
