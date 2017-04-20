import numpy as np
import os
from model import get_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.generic_utils import Progbar
from keras import backend as K
from args import get_parser
from utils.dataloader import DataLoader
from utils.lang_proc import idx2word, sample
from utils.config import get_opt, ResetStatesCallback
from coco_caption.pycocotools.coco import COCO
from coco_caption.pycocoevalcap.eval import COCOEvalCap
import sys
import pickle
import json

def gencaps(args_dict,model,gen,vocab,N_val):
    '''
    Generate and save validation captions for early stopping based on metrics
    '''
    captions = []
    samples = 0
    samples += args_dict.bs
    for [ims,prevs],_,_,imids in gen:

        # we can either use generated words as feedback, or gt ones
        if args_dict.es_prev_words == 'gt':
            preds = model.predict_on_batch([ims,prevs])
            word_idxs = np.argmax(preds,axis=-1)
        else:
            prevs = np.zeros((args_dict.bs,1))
            word_idxs = np.zeros((args_dict.bs,args_dict.seqlen))
            for i in range(args_dict.seqlen):
                # get predictions
                preds = model.predict_on_batch([ims,prevs])
                preds = preds.squeeze()
                word_idxs[:,i] = np.argmax(preds,axis=-1)
                prevs = np.argmax(preds,axis=-1)
                prevs = np.reshape(prevs,(args_dict.bs,1))
        model.reset_states()

        caps = idx2word(word_idxs,vocab)
        for i,caption in enumerate(caps):

            pred_cap = ' '.join(caption)# exclude eos
            captions.append({"image_id":imids[i]['id'],
                            "caption": pred_cap.split('<eos>')[0]})
        samples+=args_dict.bs
        if samples >= N_val:
            break

    # json file with captions to evaluate
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

def trainloop(args_dict,model,suff_name='',model_aux=None):

    ## DataLoaders
    dataloader = DataLoader(args_dict)
    N_train, N_val, _ = dataloader.get_dataset_size()
    train_gen = dataloader.generator('train',args_dict.bs)
    val_gen = dataloader.generator('val',args_dict.bs)

    if args_dict.es_metric == 'loss':

        model_name = os.path.join(args_dict.data_folder, 'models',
                                  args_dict.model_name + suff_name
                                  +'_weights.{epoch:02d}-{val_loss:.2f}.h5')

        ep = EarlyStopping(monitor='val_loss', patience=args_dict.pat,
                          verbose=0, mode='auto')

        mc = ModelCheckpoint(model_name, monitor='val_loss', verbose=0,
                            save_best_only=True, save_weights_only=True,
                            mode='auto')

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

            # forward val images to get loss
            samples = 0
            val_losses = []
            for x,y,sw in val_gen:
                val_losses.append(model.test_on_batch(x,y,sw))
                model.reset_states()
                samples+=args_dict.bs
                if samples > N_val:
                    break

            # forward val images to get captions and compute metric
            # this can either be done with true prev words or gen prev words:
            # args_dict.es_prev_words to 'gt' or 'gen'
            if args_dict.es_prev_words =='gt':
                results_file = gencaps(args_dict,model,val_gen_test,inv_vocab,N_val)
            else:
                aux_model = os.path.join(args_dict.data_folder, 'tmp',
                                         args_dict.model_name +'_aux.h5')
                model.save_weights(aux_model,overwrite=True)
                model_aux.load_weights(aux_model)
                results_file = gencaps(args_dict,model_aux,val_gen_test,
                                       inv_vocab,N_val)

            # get ground truth file to eval caps
            ann_file = os.path.join(args_dict.coco_path,
                                    'annotations/captions_val2014.json')
            # score captions and return requested metric
            metric = get_metric(args_dict,results_file,ann_file)
            prog.update(current= N_train,
                        values = [('loss',loss),('val_loss',np.mean(val_losses)),
                                  (args_dict.es_metric,metric)])

            # decide if we save checkpoint and/or stop training
            if metric > best_metric:
                best_metric = metric
                wait = 0
                model_name = os.path.join(args_dict.data_folder, 'models',
                                          args_dict.model_name + suff_name
                                          +'_weights_e' + str(e)+ '_'
                                          + args_dict.es_metric +
                                          "%0.2f"%metric+'.h5')
                model.save_weights(model_name)
            else:
                wait+=1

            if wait > args_dict.pat:
                break

    args_dict.mode = 'train'

    return model


def init_aux_model(args_dict,opt):
    if not args_dict.es_metric =='loss':
        args_dict.mode = 'test' # to have seqlen = 1 and use preds as prev words
        model_aux = get_model(args_dict)
        model_aux.compile(optimizer=opt,loss='categorical_crossentropy',
                          sample_weight_mode="temporal")
    else:
        model_aux = None
    return model_aux

def init_models(args_dict):
    model = get_model(args_dict)
    opt = get_opt(args_dict)

    model.compile(optimizer=opt,loss='categorical_crossentropy',
                  sample_weight_mode="temporal")
    model_aux = init_aux_model(args_dict,opt)

    return model, model_aux

if __name__ == "__main__":

    parser = get_parser()
    args_dict = parser.parse_args()

    print (args_dict)
    print ("="*10)
    # for reproducibility
    np.random.seed(args_dict.seed)

    if not args_dict.log_term:
        sys.stdout = open(os.path.join('../logs/',
                                       args_dict.model_name + '_train.log'), "w")

    if not args_dict.model_file: # run all training

        model, model_aux = init_models(args_dict)
        print (model.summary())
        model = trainloop(args_dict,model,model_aux=model_aux)

        # save the last state of the model to be loaded for fine tuning
        aux_model = os.path.join(args_dict.data_folder, 'tmp',
                                 args_dict.model_name +'_aux.h5')
        model.save_weights(aux_model)

        del model
        K.clear_session()

    ### Fine Tune ConvNet ###
    # get fine tuning params in place
    args_dict.lr = args_dict.ftlr
    args_dict.nepochs = args_dict.ftnepochs
    model = get_model(args_dict)
    opt = get_opt(args_dict)

    if args_dict.model_file:
        print "Loading snapshot: ", args_dict.model_file
        model.load_weights(args_dict.model_file)
    else:
        model.load_weights(aux_model)

    for layer in model.layers[1].layers:
        layer.trainable = True

    args_dict.cnn_train = True
    model_aux = init_aux_model(args_dict,opt)
    model.compile(optimizer=opt,loss='categorical_crossentropy',
                  sample_weight_mode="temporal")

    model = trainloop(args_dict,model,suff_name='_cnn_train',model_aux=model_aux)
