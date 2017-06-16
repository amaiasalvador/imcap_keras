from keras.models import Model, Sequential
from keras.layers import Input, BatchNormalization
from keras.layers.core import Dense, Activation, Permute, Merge, Lambda
from keras.layers.core import RepeatVector, Dropout, Reshape
from keras.layers.convolutional import Convolution1D
from keras.layers.pooling import GlobalAveragePooling2D,GlobalAveragePooling1D
from keras.layers.recurrent import LSTM
from layers.lstm_sent import LSTM_sent
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import TimeDistributed
from keras import backend as K
from utils.config import get_opt
import json
from args import get_parser


def image_model(args_dict,input_tensor):
    '''
    Loads specified pretrained convnet
    '''
    dim_ordering = K.image_dim_ordering()
    assert dim_ordering in {'tf','th'}

    input_shape = (args_dict.imsize,args_dict.imsize,3)

    assert args_dict.cnn in {'vgg16','vgg19','resnet'}

    if args_dict.cnn == 'vgg16':
        from keras.applications.vgg16 import VGG16 as cnn
    elif args_dict.cnn == 'vgg19':
        from keras.applications.vgg19 import VGG19 as cnn
    elif args_dict.cnn == 'resnet':
        from keras.applications.resnet50 import ResNet50 as cnn

    base_model = cnn(weights='imagenet', include_top=False,
                     input_tensor = input_tensor, input_shape = input_shape)

    if args_dict.lrmults:
        for layer in base_model.layers:
            layer.W_learning_rate_multiplier = args_dict.lrmult_conv
            layer.b_learning_rate_multiplier = args_dict.lrmult_conv

    if args_dict.cnn == 'resnet':
        model = Model(input=base_model.input,output=[base_model.layers[-2].output])
    else:
        model = base_model

    return model

def language_model(args_dict,wh,dim,convfeats,prev_words):

    # for testing stage where caption is predicted word by word
    if args_dict.mode == 'train':
        seqlen = args_dict.seqlen
    else:
        seqlen = 1
    num_classes = args_dict.vocab_size

    # imfeats need to be "flattened" eg 15x15x512 --> 225x512
    V = Reshape((wh*wh,dim),name='conv_feats')(convfeats) # 225x512

    # input is the average of conv feats
    Vg = GlobalAveragePooling1D(name='Vg')(V)
    # embed average imfeats
    Vg = Dense(args_dict.emb_dim,activation='relu',name='Vg_')(Vg)
    if args_dict.dr:
        Vg = Dropout(args_dict.dr_ratio)(Vg)

    # we keep spatial image feats to compute context vector later
    # project to z_space
    Vi = Convolution1D(args_dict.z_dim,1,border_mode='same',
                       activation='relu',name='Vi')(V)

    if args_dict.dr:
        Vi = Dropout(args_dict.dr_ratio)(Vi)
    # embed
    Vi_emb = Convolution1D(args_dict.emb_dim,1,border_mode='same',
                       activation='relu',name='Vi_emb')(Vi)

    # repeat average feat as many times as seqlen to infer output size
    x = RepeatVector(seqlen)(Vg) # seqlen,512

    # embedding for previous words
    wemb = Embedding(num_classes,args_dict.emb_dim,input_length = seqlen)
    emb = wemb(prev_words)
    emb = Activation('relu')(emb)
    if args_dict.dr:
        emb = Dropout(args_dict.dr_ratio)(emb)

    x = Merge(mode='concat',name='lstm_in')([x,emb])
    if args_dict.dr:
        x = Dropout(args_dict.dr_ratio)(x)
    if args_dict.sgate:
        # lstm with two outputs
        lstm_ = LSTM_sent(output_dim = args_dict.lstm_dim,
                          return_sequences=True,stateful=True,
                          dropout_W = args_dict.dr_ratio,
                          dropout_U = args_dict.dr_ratio,
                          sentinel=True,name='hs')
        h, s = lstm_(x)

    else:
        # regular lstm
        lstm_ = LSTM_sent(args_dict.lstm_dim,
                          return_sequences=True,stateful=True,
                          dropout_W = args_dict.dr_ratio,
                          dropout_U = args_dict.dr_ratio,
                          sentinel=False,name='h')
        h = lstm_(x)

    num_vfeats = wh*wh
    if args_dict.sgate:
        num_vfeats = num_vfeats + 1

    if args_dict.attlstm:

        # embed ht vectors.
        # linear used as input to final classifier, embedded ones are used to compute attention
        h_out_linear = Convolution1D(args_dict.z_dim,1,activation='tanh',name='zh_linear',border_mode='same')(h)
        if args_dict.dr:
            h_out_linear = Dropout(args_dict.dr_ratio)(h_out_linear)
        h_out_embed = Convolution1D(args_dict.emb_dim,1,name='zh_embed',border_mode='same')(h_out_linear)
        # repeat all h vectors as many times as local feats in v
        z_h_embed = TimeDistributed(RepeatVector(num_vfeats))(h_out_embed)

        # repeat all image vectors as many times as timesteps (seqlen)
        # linear feats are used to apply attention, embedded feats are used to compute attention
        z_v_linear = TimeDistributed(RepeatVector(seqlen),name='z_v_linear')(Vi)
        z_v_embed = TimeDistributed(RepeatVector(seqlen),name='z_v_embed')(Vi_emb)

        z_v_linear = Permute((2,1,3))(z_v_linear)
        z_v_embed = Permute((2,1,3))(z_v_embed)

        if args_dict.sgate:

            # embed sentinel vec
            # linear used as additional feat to apply attention, embedded used as add. feat to compute attention
            fake_feat = Convolution1D(args_dict.z_dim,1,activation='relu',name='zs_linear',border_mode='same')(s)
            if args_dict.dr:
                fake_feat = Dropout(args_dict.dr_ratio)(fake_feat)

            fake_feat_embed = Convolution1D(args_dict.emb_dim,1,name='zs_embed',border_mode='same')(fake_feat)
            # reshape for merging with visual feats
            z_s_linear = Reshape((seqlen,1,args_dict.z_dim))(fake_feat)
            z_s_embed = Reshape((seqlen,1,args_dict.emb_dim))(fake_feat_embed)

            # concat fake feature to the rest of image features
            z_v_linear = Merge(mode='concat',concat_axis=-2)([z_v_linear,z_s_linear])
            z_v_embed = Merge(mode='concat',concat_axis=-2)([z_v_embed,z_s_embed])

        # sum outputs from z_v and z_h
        z = Merge(mode='sum',name='merge_v_h')([z_h_embed,z_v_embed])
        if args_dict.dr:
            z = Dropout(args_dict.dr_ratio)(z)
        z = TimeDistributed(Activation('tanh',name='merge_v_h_tanh'))(z)
        # compute attention values
        att = TimeDistributed(Convolution1D(1,1,border_mode='same'),name='att')(z)

        att = Reshape((seqlen,num_vfeats),name='att_res')(att)
        # softmax activation
        att = TimeDistributed(Activation('softmax'),name='att_scores')(att)
        att = TimeDistributed(RepeatVector(args_dict.z_dim),name='att_rep')(att)
        att = Permute((1,3,2),name='att_rep_p')(att)

        # get context vector as weighted sum of image features using att
        w_Vi = Merge(mode='mul',name='vi_weighted')([att,z_v_linear])
        sumpool = Lambda(lambda x: K.sum(x, axis=-2),
                       output_shape=(args_dict.z_dim,))
        c_vec = TimeDistributed(sumpool,name='c_vec')(w_Vi)
        atten_out = Merge(mode='sum',name='mlp_in')([h_out_linear,c_vec])
        h = TimeDistributed(Dense(args_dict.emb_dim,activation='tanh'))(atten_out)
        if args_dict.dr:
            h = Dropout(args_dict.dr_ratio,name='mlp_in_tanh_dp')(h)

    predictions = TimeDistributed(Dense(num_classes,activation='softmax'),name='out')(h)

    model = Model(input=[convfeats,prev_words], output=predictions)
    opt = get_opt(args_dict)

    return model

def get_model(args_dict):

    # for testing stage where caption is predicted word by word
    if args_dict.mode == 'train':
        seqlen = args_dict.seqlen
    else:
        seqlen = 1

    # get pretrained convnet
    in_im = Input(batch_shape=(args_dict.bs,args_dict.imsize,args_dict.imsize,3),name='image')
    convnet = image_model(args_dict,in_im)

    wh = convnet.output_shape[1] # size of conv5
    dim = convnet.output_shape[3] # number of channels

    if not args_dict.cnn_train:
        for i,layer in enumerate(convnet.layers):
            if i > args_dict.finetune_start_layer:
                layer.trainable = False

    imfeats = convnet(in_im)
    convfeats = Input(batch_shape=(args_dict.bs,wh,wh,dim))
    prev_words = Input(batch_shape=(args_dict.bs,seqlen),name='prev_words')
    lang_model = language_model(args_dict,wh,dim,convfeats,prev_words)

    out = lang_model([imfeats,prev_words])

    model = Model(input=[in_im,prev_words], output=out)

    return model


if __name__ == "__main__":

    parser = get_parser()
    args_dict = parser.parse_args()
    model = get_model(args_dict)
    print (model.summary())
