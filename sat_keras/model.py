from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dense, Activation, Permute, Merge
from keras.layers.core import RepeatVector, Dropout, Reshape
from keras.layers.pooling import GlobalAveragePooling2D,GlobalAveragePooling1D
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import TimeDistributed
from keras import backend as K

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

    num_classes = args_dict.vocab_size + 2
    wh = base_model.output_shape[1] # size of conv5
    dim = base_model.output_shape[3] # number of channels

    for layer in base_model.layers:
        if not args_dict.cnn_train:
            layer.trainable = False
        else:
            if not 'block5' in layer.name and not 'block4' in layer.name:
                layer.trainable = False

    im = Input(batch_shape=(args_dict.bs,args_dict.imsize,args_dict.imsize,3))
    prev_words = Input(batch_shape=(args_dict.bs,seqlen))

    imfeats = base_model(im)
    # input is the average of conv feats
    avg_feats = GlobalAveragePooling2D()(imfeats)

    # repeat as many times as seqlen to infer output size
    x = RepeatVector(seqlen)(avg_feats) # seqlen,512

    wemb = Embedding(num_classes,args_dict.emb_dim,input_length = seqlen)
    emb = wemb(prev_words)
    x = Merge(mode='concat')([x,emb])

    in_lstm = (args_dict.bs,seqlen,args_dict.emb_dim + dim)

    h = LSTM(args_dict.lstm_dim,return_sequences=True,stateful=True)(x) # seqlen,lstm_dim
    #h = LSTM(args_dict.lstm_dim,return_sequences=True)(x) # seqlen,lstm_dim
    if args_dict.attlstm:


        # imfeats need to be "flattened" eg 15x15x512 --> 225x512
        V = Reshape((wh*wh,dim))(imfeats) # 225x512

        # map all V vectors to z space
        z_v = TimeDistributed(Dense(args_dict.z_dim,activation='tanh'))(V) # 225,z_dim

        # repeat all vectors as many times as timesteps (seqlen)
        z_v = TimeDistributed(RepeatVector(seqlen))(z_v) # 225,seqlen,z_dim
        z_v = Permute((2,1,3))(z_v) # seqlen,225,z_dim

        # map h vectors (of all timesteps) to z space
        z_h = TimeDistributed(Dense(args_dict.z_dim))(h) # seqlen,z_dim

        # repeat all h vectors as many times as v features
        z_h = TimeDistributed(RepeatVector(wh*wh))(z_h) # seqlen,225,z_dim


        # sum outputs from z_v and z_h
        z = Merge(mode='sum')([z_h,z_v]) # seqlen,225,z_dim
        z = TimeDistributed(Activation('tanh'))(z)

        # compute attention values
        att = TimeDistributed(TimeDistributed(Dense(1)))(z) # seqlen,225,1
        att = Reshape((seqlen,wh*wh))(att)
        # softmax activation
        att = TimeDistributed(Activation('softmax'))(att) # seqlen,225
        att = TimeDistributed(RepeatVector(dim))(att) #seqlen,512,225
        att = Permute((1,3,2))(att) # seqlen,225,512

        # get image vectors (repeated seqlen times)
        V_r = TimeDistributed(RepeatVector(seqlen))(V) # 225,seqlen,512
        V_r = Permute((2,1,3))(V_r) # seqlen,225,512

        # get context vector as weighted sum of image features using att
        w_imfeats = Merge(mode='mul')([att,V_r]) # seqlen,225,512
        c_vec = TimeDistributed(GlobalAveragePooling1D())(w_imfeats) # seqlen,512

        h = Merge(mode='sum')([h,c_vec])

    predictions = TimeDistributed(Dense(num_classes,activation='softmax'))(h)

    model = Model(input=[im,prev_words], output=predictions)

    return model

if __name__ == "__main__":

    parser = get_parser()
    args_dict = parser.parse_args()

    model = get_model(args_dict)

    model.summary()
