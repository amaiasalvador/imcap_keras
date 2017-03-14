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
        for j in range(idxs.shape[1]):
            caption = []
            word = vocab.get(preds[i])
            if word:
                caption.append(word)
                if word == '<eos>':
                    break
            else:
                caption.append('UNK')
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

        tok_caption.append('<eos>')

        if len(tok_caption) > maxlen:
            maxlen = len(tok_caption)
        avglen+=len(tok_caption)
        all_words.extend(tok_caption)

    print('Average length:',avglen/i)
    print('Max length:',maxlen)
    fdist = nltk.FreqDist(all_words)
    topk_words = fdist.most_common(args_dict.vocab_size)
    return topk_words,len(all_words)

def create_dict(topk_words,len_corpus):

    word2class = {}

    n_samples = 0
    for word in topk_words:
        n_samples+=word[1]
    unk_samples = len_corpus - n_samples

    p = 1
    for word in topk_words:
        weight = float(len_corpus)/(word[1]*(len(topk_words)+2))

        word2class[word[0]] = {'id':p,'w':weight}
        p+=1
    word2class['UNK'] = {'id':p,'w':len_corpus/(unk_samples*(len(topk_words)+2))}

    return word2class
