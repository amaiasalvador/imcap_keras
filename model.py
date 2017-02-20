from keras.models import Model
from keras.layers.core import Dense, Activation, RepeatVector, Dropout
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.wrappers import TimeDistributed
from layers.attention import AttentionLSTM

from args import get_parser

def get_base_model(args_dict):
    '''
    Loads specified pretrained convnet
    '''

    input_shape = (args_dict.imsize,args_dict.imsize,3)

    if args_dict.cnn == 'vgg16':
        from keras.applications.vgg16 import VGG16 as cnn
    elif args_dict.cnn == 'vgg19':
        from keras.applications.vgg19 import VGG19 as cnn
    elif args_dict.cnn == 'resnet50':
        from keras.applications.resnet50 import ResNet50 as cnn

    base_model = cnn(weights='imagenet', include_top=False,
                           input_shape = input_shape)

    return base_model

def get_model(args_dict):

    base_model = get_base_model(args_dict)

    if not args_dict.cnn_train:
        for layer in base_model.layers:
            layer.trainable = False

    imfeats = base_model.output
    avg_feats = GlobalAveragePooling2D()(imfeats)
    avg_feats = RepeatVector(args_dict.seqlen)(avg_feats)

    att_layer = AttentionLSTM(args_dict.lstm_dim,return_sequences=True)
    hdims = att_layer([avg_feats,imfeats])

    d1 = TimeDistributed(Dense(args_dict.fc_dim))(hdims)
    d1 = Dropout(args_dict.dr_ratio)(d1)
    d1 = Activation('relu')(d1)

    d2 = TimeDistributed(Dense(args_dict.n_classes))(d1)
    predictions = TimeDistributed(Activation('softmax'))(d2)

    model = Model(input=base_model.input, output=predictions)

    return model

if __name__ == "__main__":

    parser = get_parser()
    args_dict = parser.parse_args()

    model = get_model(args_dict)

    model.summary()
