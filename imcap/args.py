import argparse

def get_parser():

    parser = argparse.ArgumentParser(description='Image Captioning in Keras',epilog='Good luck!')

    # naming & savefiles
    parser.add_argument('-model_name', dest='model_name',
                        default = 'model',help='Base name to save model')
    parser.add_argument('-h5file', dest='h5file',
                        default = 'cocotalk_challenge.h5',
                        help='name of hdf5 file (with extension)')
    parser.add_argument('-json_file', dest='json_file',
                        default = 'cocotalk_challenge.json',
                        help='name of json file (with extension)')
    parser.add_argument('-model_file', dest='model_file',
                        default = None,help='path to model file to load (either for testing or snapshooting).')
    # data-related
    parser.add_argument('-coco_path', dest='coco_path',
                        default = '/seq/segmentation/COCO/tools',
                        help='COCO database')
    parser.add_argument('-data_folder', dest='data_folder',
                        default = '/work/asalvador/sat_keras/',
                        help='save folder')

    # model parts, inputs
    parser.add_argument('-cnn', dest='cnn',
                        default = 'resnet', choices=['vgg16','vgg19','resnet'],
                        help='Pretrained CNN to use')
    parser.add_argument('-imsize', dest='imsize',
                        default = 256, help='Image size',type=int)
    parser.add_argument('-vocab_size', dest='vocab_size',
                        default = 9570, help='Vocabulary size' ,type=int)
    parser.add_argument('-n_caps', dest='n_caps',
                        default = 5, help='Number of captions for training',
                        type=int)

    # Model parameters
    parser.add_argument('-seqlen',dest='seqlen', default = 18,
                        help='Maximum sentence length',type=int)
    parser.add_argument('-lstm_dim',dest='lstm_dim', default = 512,
                        help='Number of LSTM units',type=int)
    parser.add_argument('-emb_dim',dest='emb_dim', default = 512,
                        help='Embedding dim to input lstm',type=int)
    parser.add_argument('-z_dim',dest='z_dim', default = 512,
                        help='Dimensionality of z space (att operations)',type=int)
    parser.add_argument('-dr_ratio',dest='dr_ratio', default = 0.5,
                        help='Dropout ratio',type=float)
    parser.add_argument('-finetune_start_layer',dest='finetune_start_layer',
                        default = 6,help='CNN layer to start fine tuning',type=int)
    parser.add_argument('-nfilters',dest='nfilters', default = 2048,
                        help='Number of channels for conv layer',type=int)
    parser.add_argument('-convsize',dest='convsize', default = 7,
                        help='Spatial dimension of conv layer',type=int)
    # Training params
    parser.add_argument('-seed', dest='seed',
                        default = 123, help='Random seed',type=int)
    parser.add_argument('-bs',dest='bs', default = 32,
                            help='Batch Size',type=int)
    parser.add_argument('-optim',dest='optim', default ='adam',
                                choices=['adam','SGD','adadelta','adagrad',
                                'rmsprop'], help='Optimizer')
    parser.add_argument('-alpha',dest='alpha', default = 0.9,
                                help='Adams alpha',type=float)
    parser.add_argument('-beta',dest='beta', default = 0.999,
                                help='Adams beta',type=float)
    parser.add_argument('-lr',dest='lr', default = 5e-4,
                                help='Learning rate',type=float)
    parser.add_argument('-lrmult_conv',dest='lrmult_conv', default = 0.001,
                                help='Learning rate multiplier for convnet',type=float)
    parser.add_argument('-ftlr',dest='ftlr', default = 1e-5,
                                help='Learning rate when fine tuning',type=float)
    parser.add_argument('-decay',dest='decay', default = 0.0,
                                help='LR decay',type=float)
    parser.add_argument('-clip',dest='clip', default = 5,
                        help='Gradient clipping threshold (value)',type=float)
    parser.add_argument('-nepochs',dest='nepochs', default = 20,
                        help='Number of train epochs (frozen cnn)',type=int)
    parser.add_argument('-ftnepochs',dest='ftnepochs', default = 30,
                        help='Number of train epochs (ft cnn)',type=int)
    parser.add_argument('-pat',dest='pat', default = 5,
                            help='Patience',type=int)
    parser.add_argument('-workers',dest='workers', default = 2,
                        help='Number of data loading threads',type=int)
    parser.add_argument('-es_prev_words',dest='es_prev_words', default = 'gen',
                        choices=['gt','gen'])
    parser.add_argument('-es_metric',dest='es_metric', default = 'CIDEr',
                        help='Early stopping metric',
                        choices=['loss','CIDEr','Bleu_4','Bleu_3','Bleu_2',
                                 'Bleu_1','ROUGE_L','METEOR'])

    parser.add_argument('-bsize',dest='bsize', default = 1,
                        help='Beam size',type=int)
    parser.add_argument('-temperature',dest='temperature', default = -1,
                        help='Sampling temperature',type=float)
    # flags & bools
    parser.add_argument('-mode', dest='mode',
                        default = 'train',choices=['train','test'],
                        help='Model mode')
    parser.add_argument('--cnntrain', dest='cnn_train', action='store_true')
    parser.set_defaults(cnn_train=False)

    parser.add_argument('--lstm', dest='attlstm', action='store_false')
    parser.set_defaults(attlstm=True)

    parser.add_argument('--lrmults', dest='lrmults', action='store_true')
    parser.set_defaults(lrmults=False)


    parser.add_argument('--sgate', dest='sgate', action='store_true')
    parser.set_defaults(sgate=False)

    parser.add_argument('--log_term', dest='log_term', action='store_true')
    parser.set_defaults(log_term=False)

    parser.add_argument('--dr',dest='dr', action ='store_true')
    parser.add_argument('--bn',dest='bn', action ='store_true')
    parser.set_defaults(dr=False)
    parser.set_defaults(bn=False)
    return parser

if __name__ =="__main__":

    parser = get_parser()
    args_dict = parser.parse_args()
