from keras.models import Model
from keras.layers import Input, BatchNormalization
from keras.layers.core import Dense, Activation, Permute, Merge
from keras.layers.core import RepeatVector, Dropout, Reshape
from keras.layers.pooling import GlobalAveragePooling2D,GlobalAveragePooling1D
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import TimeDistributed
from keras import backend as K
from keras.regularizers import l2

from args import get_parser

def get_base_model(args_dict):
    '''
    Loads specified pretrained convnet
    '''

    dim_ordering = K.image_dim_ordering()
    assert dim_ordering in {'tf','th'}

    if dim_ordering == 'th':
        input_shape = (3,args_dict.imsize,args_dict.imsize)
    else:
        input_shape = (args_dict.imsize,args_dict.imsize,3)

    assert args_dict.cnn in {'vgg16','vgg19','resnet'}

    if args_dict.cnn == 'vgg16':
        from keras.applications.vgg16 import VGG16 as cnn
    elif args_dict.cnn == 'vgg19':
        from keras.applications.vgg19 import VGG19 as cnn
    elif args_dict.cnn == 'resnet':
        from keras.applications.resnet50 import ResNet50 as cnn

    base_model = cnn(weights='imagenet', include_top=False,
                           input_shape = input_shape)

    return base_model

def get_model(args_dict):

    '''
    Build the captioning model
    '''

    # for testing stage where caption is predicted word by word
    if args_dict.mode == 'train':
        seqlen = args_dict.seqlen
    else:
        seqlen = 1

    # get pretrained convnet
    base_model = get_base_model(args_dict)

    num_classes = args_dict.vocab_size + 4
    wh = base_model.output_shape[1] # size of conv5
    dim = base_model.output_shape[3] # number of channels

    # specific for vgg16/vgg19. todo: adapt layer names to resnet
    for layer in base_model.layers:
        if not args_dict.cnn_train:
            layer.trainable = False
        else:
            if not 'block5' in layer.name and not 'block4' in layer.name:
                layer.trainable = False

    im = Input(batch_shape=(args_dict.bs,args_dict.imsize,args_dict.imsize,3),name='image')
    prev_words = Input(batch_shape=(args_dict.bs,seqlen),name='prev_words')

    imfeats = base_model(im)

    # imfeats need to be "flattened" eg 15x15x512 --> 225x512
    V = Reshape((wh*wh,dim))(imfeats) # 225x512
    V = BatchNormalization(name='V')(V)

    # input is the average of conv feats
    Vg = GlobalAveragePooling1D(name='Vg')(V)
    Vg = Dense(args_dict.z_dim,activation='relu',W_regularizer=l2(args_dict.l2reg),name='Vg_')(Vg)

    # we keep spatial image feats to compute context vector later
    Vi = TimeDistributed(Dense(args_dict.z_dim,activation='relu',
                               W_regularizer=l2(args_dict.l2reg)),name='Vi')(V)

    # repeat as many times as seqlen to infer output size
    x = RepeatVector(seqlen)(Vg) # seqlen,512

    # embedding for previous words
    wemb = Embedding(num_classes,args_dict.emb_dim,input_length = seqlen,name='wemb')
    emb = wemb(prev_words)

    # input is the concatenation of avg imfeats and previous words
    x = Merge(mode='concat',name='lstm_in')([x,emb])

    in_lstm = (args_dict.bs,seqlen,args_dict.emb_dim + dim)
    lstm_ = LSTM(args_dict.lstm_dim,return_sequences=True,stateful=True,
                 dropout_W=args_dict.dr_ratio,dropout_U=args_dict.dr_ratio,
                 W_regularizer = l2(args_dict.l2reg),
                 U_regularizer=l2(args_dict.l2reg), name='lstm')

    h = lstm_(x) # seqlen,lstm_dim
    #h = LSTM(args_dict.lstm_dim,return_sequences=True)(x) # seqlen,lstm_dim
    if args_dict.attlstm:

        # repeat all vectors as many times as timesteps (seqlen)
        z_v = TimeDistributed(RepeatVector(seqlen),name='Vi_rep')(Vi) # 225,seqlen,z_dim
        z_v = Permute((2,1,3),name='Vi_rep_p')(z_v) # seqlen,225,z_dim

        # map h vectors (of all timesteps) to z space
        z_h = TimeDistributed(Dense(args_dict.z_dim,activation='relu',
                                    W_regularizer=l2(args_dict.l2reg)),name='h_')(h) # seqlen,z_dim
        z_h = BatchNormalization()(z_h)

        # repeat all h vectors as many times as v features
        z_h = TimeDistributed(RepeatVector(wh*wh),name='h_rep')(z_h) # seqlen,225,z_dim

        # sum outputs from z_v and z_h
        z = Merge(mode='concat',name='merge_v_h')([z_h,z_v]) # seqlen,225,z_dim

        # compute attention values
        att = TimeDistributed(TimeDistributed(Dense(1,W_regularizer=l2(args_dict.l2reg))),name='att')(z) # seqlen,225,1
        att = Reshape((seqlen,wh*wh),name='att_res')(att)
        # softmax activation
        att = TimeDistributed(Activation('softmax'),name='att_scores')(att) # seqlen,225
        att = TimeDistributed(RepeatVector(args_dict.z_dim),name='att_rep')(att) #seqlen,512,225
        att = Permute((1,3,2),name='att_rep_p')(att) # seqlen,225,512

        # get image vectors (repeated seqlen times)
        Vi_r = TimeDistributed(RepeatVector(seqlen),name='vi_rep')(Vi) # 225,seqlen,512
        Vi_r = Permute((2,1,3),name='vi_rep_p')(Vi_r) # seqlen,225,512

        # get context vector as weighted sum of image features using att
        w_Vi = Merge(mode='mul',name='vi_weighted')([att,Vi_r]) # seqlen,225,512
        c_vec = TimeDistributed(GlobalAveragePooling1D(),name='c_vec')(w_Vi) # seqlen,512

        h = Merge(mode='concat',name='mlp_in')([h,c_vec])
        h = Activation('tanh')(h)
        h = Dropout(args_dict.dr_ratio)(h)

    predictions = TimeDistributed(Dense(num_classes,activation='softmax',W_regularizer=l2(args_dict.l2reg)),name='out')(h)

    model = Model(input=[im,prev_words], output=predictions)

    return model

if __name__ == "__main__":

    parser = get_parser()
    args_dict = parser.parse_args()

    model = get_model(args_dict)

    model.summary()
