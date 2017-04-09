from keras.models import Model
from keras.layers import Input, BatchNormalization, Lambda
from keras.layers.core import Dense, Activation, Permute, Merge
from keras.layers.core import RepeatVector, Dropout, Reshape
from keras.layers.pooling import GlobalAveragePooling2D,GlobalAveragePooling1D
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import TimeDistributed
from keras import backend as K
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

    # convnet feats (NxMxdim)
    imfeats = base_model(im)

    # imfeats need to be "flattened" eg 15x15x512 --> 225x512
    V = Reshape((wh*wh,dim))(imfeats) # 225x512
    V = BatchNormalization()(V)

    # input is the average of conv feats
    Vg = GlobalAveragePooling1D(name='Vg')(V)
    # embed average imfeats
    Vg = Dense(args_dict.emb_dim,activation='relu',name='Vg_')(Vg)
    Vg = Dropout(args_dict.dr_ratio)(Vg)

    # we keep spatial image feats to compute context vector later
    # project to z_space
    Vi = TimeDistributed(Dense(args_dict.z_dim,activation='relu'),name='Vi')(V)
    Vi = Dropout(args_dict.dr_ratio)(Vi)

    # embed
    Vi_emb = TimeDistributed(Dense(args_dict.emb_dim),name='Vi_emb')(Vi)

    # repeat average feat as many times as seqlen to infer output size
    x = RepeatVector(seqlen)(Vg) # seqlen,512

    # embedding for previous words
    wemb = Embedding(num_classes,args_dict.z_dim,input_length = seqlen,name='wemb')
    emb = wemb(prev_words)
    emb = BatchNormalization()(emb)

    emb = Activation('relu')(emb)
    emb = Dropout(args_dict.dr_ratio)(emb)

    # input is the concatenation of avg imfeats and previous words
    x = Merge(mode='sum',name='lstm_in')([x,emb])

    if args_dict.sgate:
        # lstm with two outputs
        lstm_ = LSTM_sent(args_dict.lstm_dim,return_sequences=True,stateful=True,
                     dropout_W=args_dict.dr_ratio,dropout_U=args_dict.dr_ratio,
                     name='hs')
        h, s = lstm_(x)

    else:
        # regular lstm
        lstm_ = LSTM(args_dict.lstm_dim,return_sequences=True,stateful=True,
                     dropout_W=args_dict.dr_ratio,dropout_U=args_dict.dr_ratio,
                     name='h')
        h = lstm_(x)

    num_vfeats = wh*wh
    if args_dict.sgate:
        num_vfeats = num_vfeats + 1

    if args_dict.attlstm:

        # repeat all image vectors as many times as timesteps (seqlen)
        # linear feats are used to apply attention, embedded feats are used to compute attention
        z_v_linear = TimeDistributed(RepeatVector(seqlen),name='z_v_linear')(Vi)
        z_v_embed = TimeDistributed(RepeatVector(seqlen),name='z_v_embed')(Vi_emb)

        z_v_linear = Permute((2,1,3))(z_v_linear)
        z_v_embed = Permute((2,1,3))(z_v_embed)

        # embed ht vectors.
        # linear used as input to final classifier, embedded ones are used to compute attention
        h = BatchNormalization()(h)
        h_out_linear = TimeDistributed(Dense(args_dict.z_dim,activation='tanh'),name='zh_linear')(h)
        h_out_linear = Dropout(args_dict.dr_ratio)(h_out_linear)
        h_out_embed = TimeDistributed(Dense(args_dict.emb_dim),name='zh_embed')(h_out_linear)

        # repeat all h vectors as many times as local feats in v
        z_h_embed = TimeDistributed(RepeatVector(num_vfeats))(h_out_embed)

        if args_dict.sgate:

            # embed sentinel vec
            s = BatchNormalization()(s)
            # linear used as additional feat to apply attention, embedded used as add. feat to compute attention
            fake_feat = TimeDistributed(Dense(args_dict.z_dim,activation='relu'),name='zs_linear')(s)
            fake_feat = Dropout(args_dict.dr_ratio)(fake_feat)

            fake_feat_embed = TimeDistributed(Dense(args_dict.emb_dim),name='zs_embed')(fake_feat)
            # reshape for merging with visual feats
            z_s_linear = Reshape((seqlen,1,args_dict.z_dim))(fake_feat)
            z_s_embed = Reshape((seqlen,1,args_dict.emb_dim))(fake_feat_embed)

            # concat fake feature to the rest of image features
            z_v_linear = Merge(mode='concat',concat_axis=-2)([z_v_linear,z_s_linear])
            z_v_embed = Merge(mode='concat',concat_axis=-2)([z_v_embed,z_s_embed])

        # sum outputs from z_v and z_h
        z = Merge(mode='sum',name='merge_v_h')([z_h_embed,z_v_embed])
        z = Activation('tanh',name='merge_v_h_tanh')(z)
        z = Dropout(args_dict.dr_ratio)(z)
        # compute attention values
        att = TimeDistributed(TimeDistributed(Dense(1)),name='att')(z)

        att = Reshape((seqlen,num_vfeats),name='att_res')(att)
        # softmax activation
        att = TimeDistributed(Activation('softmax'),name='att_scores')(att)
        att = TimeDistributed(RepeatVector(args_dict.z_dim),name='att_rep')(att)
        att = Permute((1,3,2),name='att_rep_p')(att)

        # get context vector as weighted sum of image features using att
        w_Vi = Merge(mode='mul',name='vi_weighted')([att,z_v_linear])
        c_vec = TimeDistributed(GlobalAveragePooling1D(),name='c_vec')(w_Vi)
        c_vec = BatchNormalization()(c_vec)
        h = Merge(mode='sum',name='mlp_in')([h_out_linear,c_vec])
        h = TimeDistributed(Dense(args_dict.emb_dim,activation='tanh'))(h)
        h = Dropout(args_dict.dr_ratio,name='mlp_in_tanh_dp')(h)

    predictions = TimeDistributed(Dense(num_classes,activation='softmax'),name='out')(h)

    model = Model(input=[im,prev_words], output=predictions)

    return model

if __name__ == "__main__":

    parser = get_parser()
    args_dict = parser.parse_args()

    model = get_model(args_dict)
    model.summary()
