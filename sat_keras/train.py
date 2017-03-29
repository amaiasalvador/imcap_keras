import numpy as np
import os
from model import get_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.generic_utils import Progbar
from args import get_parser
from utils.dataloader import DataLoader
from utils.lang_proc import idx2word
from utils.config import get_opt, ResetStatesCallback
from cococaption.pycocotools.coco import COCO
from cococaption.pycocoevalcap.eval import COCOEvalCap
import sys
import pickle
import json

def gencaps(args_dict,model,gen,vocab,N_val):
    '''
    Extract and save validation captions for early stopping based on metrics
    '''
    captions = []
    samples = args_dict.bs
    for x,y,imids in gen:
        preds = model.predict_on_batch(x)
        model.reset_states()
        pred_idxs = np.argmax(preds,axis=-1)
        caps = idx2word(pred_idxs,vocab)

        for i,caption in enumerate(caps):

            pred_cap = ' '.join(caption)# exclude eos
            captions.append({"image_id":imids[i]['id'],
                            "caption": pred_cap.split('<eos>')[0]})
        samples+=args_dict.bs
        if samples >= N_val:
            break
    results_file = os.path.join(args_dict.data_folder, 'results',
                              args_dict.model_name +'_gencaps_val.json')
    with open(results_file, 'w') as outfile:
        json.dump(captions, outfile)

    return results_file

def get_metric(args_dict,results_file,ann_file):
    coco = COCO(ann_file)
    cocoRes = coco.loadRes(results_file)

    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.evaluate()

    return cocoEval.eval[args_dict.es_metric]

def trainloop(args_dict,model,suff_name=''):

    ## DataLoaders
    dataloader = DataLoader(args_dict)
    N_train, N_val, N_test = dataloader.get_dataset_size()
    train_gen = dataloader.generator('train',args_dict.bs)
    val_gen = dataloader.generator('val',args_dict.bs)

    if args_dict.es_metric == 'loss':

        model_name = os.path.join(args_dict.data_folder, 'models',
                                  args_dict.model_name
                                  +'_weights.{epoch:02d}-{val_loss:.2f}.h5')

        ep = EarlyStopping(monitor='val_loss', patience=args_dict.pat,
                          verbose=0, mode='auto')

        mc = ModelCheckpoint(model_name, monitor='val_loss', verbose=0,
                            save_best_only=True, mode='auto')

        # reset states after each batch (bcs stateful)
        rs = ResetStatesCallback()

        model.fit_generator(train_gen,nb_epoch=args_dict.nepochs,
                            samples_per_epoch=N_train,
                            validation_data=val_gen,
                            nb_val_samples=N_val,
                            callbacks=[mc,ep,rs],
                            verbose = 1,
                            nb_worker = args_dict.workers,
                            pickle_safe = False)

    else: # models saved based on other metrics - manual train loop

        # validation generator in test mode to output image names
        val_gen_test = dataloader.generator('val',args_dict.bs,train_flag=False)

        # load vocab to convert captions to words and compute cider
        vocab_file = os.path.join(args_dict.data_folder,'data',args_dict.vfile)
        vocab = pickle.load(open(vocab_file,'rb'))
        inv_vocab = {v:k for k,v in vocab.items()}
        # init waiting param and best metric values
        wait = 0
        best_metric = -np.inf

        for e in range(args_dict.nepochs):
            print "Epoch %d/%d"%(e+1,args_dict.nepochs)
            prog = Progbar(target = N_train)

            samples = 0
            for x,y,sw in train_gen: # do one epoch
                loss = model.train_on_batch(x=x,y=y,sample_weight=sw)
                model.reset_states()

                samples+=args_dict.bs
                if samples >= N_train:
                    break
                prog.update(current= samples ,values = [('loss',loss)])

            samples = 0
            val_losses = []
            for x,y,sw in val_gen:
                val_losses.append(model.test_on_batch(x,y,sw))
                model.reset_states()
                samples+=args_dict.bs
                if samples > N_val:
                    break

            # get val metrics
            # this will save a file with generated captions
            results_file = gencaps(args_dict,model,val_gen_test,inv_vocab,N_val)
            # the ground truth file
            ann_file = os.path.join(args_dict.coco_path,
                                    'annotations/captions_val2014.json')
            # score captions and return requested metric
            metric = get_metric(args_dict,results_file,ann_file)
            prog.update(current= N_train,
                        values = [('loss',loss),('val_loss',np.mean(val_losses)),
                                  (args_dict.es_metric,metric)])
            if metric > best_metric:
                best_metric = metric
                wait = 0
                model_name = os.path.join(args_dict.data_folder, 'models',
                                          args_dict.model_name + '_'+ suff_name
                                          +'_weights_'+ '.e' + str(e)+ '_'
                                          + args_dict.es_metric +
                                          "%0.2f"%metric+'.h5')
                model.save_weights(model_name)
            else:
                wait+=1

            if wait > args_dict.pat:
                break

    return model

if __name__ == "__main__":

    parser = get_parser()
    args_dict = parser.parse_args()

    print (args_dict)
    print ("="*10)
    # for reproducibility
    np.random.seed(args_dict.seed)
    sys.stdout = open(os.path.join('../logs/',args_dict.model_name+'_train.txt'),"w")

    ###  Frozen Convnet ###
    model = get_model(args_dict)
    opt = get_opt(args_dict)

    model.compile(optimizer=opt,loss='categorical_crossentropy',
                  sample_weight_mode="temporal")
    model = trainloop(args_dict,model)

    ### Fine Tune ConvNet ###
    args_dict.lr = args_dict.ftlr
    args_dict.nepochs = args_dict.ftnepochs
    opt = get_opt(args_dict)

    for layer in model.layers[1].layers:
        layer.trainable = True

    model.compile(optimizer=opt,loss='categorical_crossentropy',
                  sample_weight_mode="temporal")

    model = trainloop(args_dict,model,suff_name='cnn_train')
