from keras.models import Model
from keras.layers import Input, BatchNormalization, Lambda
from keras.layers.core import Dense, Activation, Permute, Merge
from keras.layers.core import RepeatVector, Dropout, Reshape
from keras.layers.pooling import GlobalAveragePooling2D,GlobalAveragePooling1D
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import TimeDistributed
from keras import backend as K
from keras.regularizers import l2
from layers.lstm_sent import LSTM_sent

from args import get_parser

def slice_0(x):
    return x[0]
def slice_1(x):
    return x[1]
def slice_output_shape(input_shape):
    return input_shape[0]

def get_base_model(args_dict):
    '''
    Loads specified pretrained convnet
    '''

    dim_ordering = K.image_dim_ordering()
    assert dim_ordering in {'tf','th'}

    if dim_ordering == 'th':
        input_shape = (3,args_dict.imsize,args_dict.imsize)
        input_tensor = Input(batch_shape=(args_dict.bs,3,args_dict.imsize,args_dict.imsize))
    else:
        input_shape = (args_dict.imsize,args_dict.imsize,3)
        input_tensor = Input(batch_shape=(args_dict.bs,args_dict.imsize,args_dict.imsize,3))

    assert args_dict.cnn in {'vgg16','vgg19','resnet'}

    if args_dict.cnn == 'vgg16':
        from keras.applications.vgg16 import VGG16 as cnn
    elif args_dict.cnn == 'vgg19':
        from keras.applications.vgg19 import VGG19 as cnn
    elif args_dict.cnn == 'resnet':
        from keras.applications.resnet50 import ResNet50 as cnn

    base_model = cnn(weights='imagenet', include_top=False,
                     input_tensor = input_tensor, input_shape = input_shape)
    if args_dict.cnn == 'resnet':
        return Model(input=base_model.input,output=[base_model.layers[-2].output])
    else:
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

    if not args_dict.cnn_train:
        for layer in base_model.layers:
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

    if args_dict.sgate:
        lstm_ = LSTM_sent(args_dict.lstm_dim,return_sequences=True,stateful=True,
                     dropout_W=args_dict.dr_ratio,dropout_U=args_dict.dr_ratio,
                     W_regularizer = l2(args_dict.l2reg),
                     U_regularizer=l2(args_dict.l2reg), name='hs')
        hs = lstm_(x)

        h = Lambda(slice_0,output_shape=slice_output_shape,name='h')(hs)
        h = Reshape((seqlen,args_dict.lstm_dim))(h)
        s = Lambda(slice_1,output_shape=slice_output_shape,name='s')(hs)
        s = Reshape((seqlen,args_dict.lstm_dim))(s)

    else:
        lstm_ = LSTM(args_dict.lstm_dim,return_sequences=True,stateful=True,
                     dropout_W=args_dict.dr_ratio,dropout_U=args_dict.dr_ratio,
                     W_regularizer = l2(args_dict.l2reg),
                     U_regularizer=l2(args_dict.l2reg), name='h')
        h = lstm_(x)

    num_vfeats = wh*wh
    if args_dict.sgate:
        num_vfeats = num_vfeats + 1

    if args_dict.attlstm:

        # repeat all vectors as many times as timesteps (seqlen)
        z_v = TimeDistributed(RepeatVector(seqlen),name='zv_rep')(Vi)
        z_v = Permute((2,1,3),name='zv_rep_p')(z_v)
        z_v = BatchNormalization(name='zv_rep_p_bn')(z_v)

        # map h vectors (of all timesteps) to z space
        h = TimeDistributed(Dense(args_dict.z_dim,activation='relu',
                                    W_regularizer=l2(args_dict.l2reg)),name='zh')(h)
        h = BatchNormalization(name='zh_bn')(h)
        # repeat all h vectors as many times as v features
        z_h = TimeDistributed(RepeatVector(num_vfeats),name='zh_rep_bn')(h)

        if args_dict.sgate:

            # map s vectors to z space
            s = TimeDistributed(Dense(args_dict.z_dim,activation='relu',
                                W_regularizer=l2(args_dict.l2reg)),name='zs')(s)
            s = BatchNormalization(name='zs_bn')(s) # (seqlen,zdim)
            # reshape for merging with visual feats
            s = Reshape((seqlen,1,args_dict.z_dim))(s)

            z_v = Merge(mode='concat',concat_axis=-2)([z_v,s])

            z_h = Reshape((seqlen,num_vfeats,args_dict.z_dim))(z_h)

        # sum outputs from z_v and z_h
        z = Merge(mode='sum',name='merge_v_h')([z_h,z_v])
        z = Activation('tanh',name='merge_v_h_tanh')(z)

        # compute attention values
        att = TimeDistributed(TimeDistributed(Dense(1,W_regularizer=l2(args_dict.l2reg))),name='att')(z)

        att = Reshape((seqlen,num_vfeats),name='att_res')(att)
        # softmax activation
        att = TimeDistributed(Activation('softmax'),name='att_scores')(att)
        att = TimeDistributed(RepeatVector(args_dict.z_dim),name='att_rep')(att)
        att = Permute((1,3,2),name='att_rep_p')(att)

        # get context vector as weighted sum of image features using att
        w_Vi = Merge(mode='mul',name='vi_weighted')([att,z_v])
        c_vec = TimeDistributed(GlobalAveragePooling1D(),name='c_vec')(w_Vi)
        c_vec = BatchNormalization(name='c_vec_bn')(c_vec)

        h = Merge(mode='sum',name='mlp_in')([h,c_vec])
        h = Activation('tanh',name='mlp_in_tanh')(h)
        h = Dropout(args_dict.dr_ratio,name='mlp_in_tanh_dp')(h)

    predictions = TimeDistributed(Dense(num_classes,activation='softmax',W_regularizer=l2(args_dict.l2reg)),name='out')(h)

    model = Model(input=[im,prev_words], output=predictions)

    return model

if __name__ == "__main__":

    parser = get_parser()
    args_dict = parser.parse_args()

    model = get_model(args_dict)

    model.summary()
