import nltk
import numpy as np
import os
import json

'''
Utilities to create word dictionary & generate captions
'''

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def idx2word(idxs,vocab):

    captions = []
    for i in range(idxs.shape[0]): # for all images
        caption = []
        for j in range(idxs.shape[1]): # for all elements in sequence
            word = vocab.get(idxs[i,j])
            if word:
                caption.append(word)
                if word == '<eos>':
                    break
            else:
                caption.append('<unk>')
        captions.append(caption)

    return captions

def load_caps(args_dict):

    ann_file = os.path.join(args_dict.coco_path,'annotations',
                            'captions_train' + args_dict.year+'.json')
    anns = json.load(open(ann_file))

    return anns['annotations']

def topK(anns,args_dict):

    all_words = []
    maxlen = 0
    avglen = 0

    for i,ann in enumerate(anns):
        caption =ann['caption'].lower()
        tok_caption = nltk.word_tokenize(caption)

        if len(tok_caption) > maxlen:
            maxlen = len(tok_caption)
        avglen+=len(tok_caption)
        all_words.extend(tok_caption)

    print('Average length:',avglen/i)
    print('Max length:',maxlen)
    fdist = nltk.FreqDist(all_words)
    topk_words = fdist.most_common(args_dict.vocab_size)
    return topk_words

def create_dict(topk_words):

    word2class = {'<start>':1,'<eos>':2,'<unk>':3}
    p = 4
    for word in topk_words:
        word2class[word[0]] = p
        p+=1

    return word2class

def lemmatize_sentence(sentence):
    lem = []
    for word in sentence:
        lem.append(nltk.stem.WordNetLemmatizer().lemmatize(word))
    return lem

def beamsearch(model,image,start=1,eos=2,maxsample=15,k=3,max_keep=200):

    prevs = np.ones((1,1))
    live_samples  = [[start]]
    live_scores = [0]
    dead_k = 0
    dead_samples = []
    dead_scores = []
    live_k = 1 # samples that did not yet reached eos

    while live_k and dead_k < k:

        probas = []
        # for each of the live samples
        for i in range(live_k):
            for j in range(np.shape(live_samples)[1]): #and for all elements in seq
                prev = np.expand_dims(live_samples[i][j],axis=0)
                # feed all samples to model until the last one
                probs = model.predict([image,prev]).squeeze()
            # the probabilities obtained for the last sample are the kept ones
            probas.append(probs)
            model.reset_states() # reset between different sequences
        probas = np.array(probas)
        probas = np.reshape(probas,(live_k,probas.shape[-1]))

        # top K with highest probability
        idxs = np.argsort(probas,axis=-1)[::-1]
        idxs = idxs[:,:(k-dead_k)].flatten()
        aux_samples = []
        aux_scores = []

        voc_size = probas.shape[-1]*probas.shape[0]
        for i,idx in enumerate(idxs):
            aux_samples.append(live_samples[i//k] + [idx])
            aux_scores.append(live_scores[i//k] + probas[i//k,idx])
        live_samples = aux_samples
        live_scores = aux_scores

         # live samples that should be dead are...
        zombie = [s[-1] == eos or len(s) >= maxsample for s in live_samples]

        # add zombies to the dead
        dead_samples += [s for s,z in zip(live_samples,zombie) if z]  # remove first label == empty
        dead_scores += [s for s,z in zip(live_scores,zombie) if z]
        dead_k = len(dead_samples)
        # remove zombies from the living
        live_samples = [s for s,z in zip(live_samples,zombie) if not z]
        live_scores = [s for s,z in zip(live_scores,zombie) if not z]

        if len(live_samples) > max_keep:
            top_samples = np.argsort(np.array(live_scores))[::-1][:max_keep]
            live_samples = np.array(live_samples)[top_samples].tolist()
            live_scores = np.array(live_scores)[top_samples].tolist()
        live_k = len(live_samples)
        if live_k > 0:
            prevs = np.array(live_samples)[:,-1]

    return dead_samples + live_samples, dead_scores + live_scores
