from keras.models import Model
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.core import RepeatVector, Dropout, Reshape
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.wrappers import TimeDistributed
from layers.attention import AttentionLSTM
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

    # get pretrained convnet
    base_model = get_base_model(args_dict)


    for layer in base_model.layers:
        if not args_dict.cnn_train:
            layer.trainable = False
        else:
            if not 'block5' in layer.name and not 'block4' in layer.name:
                layer.trainable = False

    # input to captioning model will be last conv layer
    imfeats = base_model.output
    # context vector is initialized as the spatial average of the conv layer
    avg_feats = GlobalAveragePooling2D()(imfeats)
    # repeat as many times as seqlen to infer output size
    avg_feats = RepeatVector(args_dict.seqlen)(avg_feats)

    wh = base_model.output_shape[1] # size of conv5
    dim = base_model.output_shape[3] # number of channels
    # imfeats need to be "flattened" eg 15x15x512 --> 225x512
    imfeats = Reshape((wh*wh,dim))(imfeats)

    att_lstm = AttentionLSTM(args_dict.lstm_dim,
                              return_sequences=True)
    hdims = att_lstm([avg_feats,imfeats])
    # + 2 because <UNK> and padded values (class 0 with 0 weight)
    predictions = TimeDistributed(Dense(args_dict.vocab_size + 2,activation='softmax'))(hdims)

    model = Model(input=base_model.input, output=predictions)

    return model

if __name__ == "__main__":

    parser = get_parser()
    args_dict = parser.parse_args()

    model = get_model(args_dict)

    model.summary()
