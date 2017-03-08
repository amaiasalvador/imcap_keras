import copy
import numpy as np
import math

def get_probs(vgroups):
    # get lengths for each group
    lens = {k:len(v) for k,v in vgroups.items()}

    # get number of groups based on max id
    n_groups = np.max(list(lens.keys()))+1
    probs = np.zeros(n_groups)
    for k,v in lens.items():
        probs[k] = v
    # array of probabilities based on number of samples remaining in each group
    probs = np.array(probs)/np.sum(probs)

    return probs, n_groups

groups = {1:[1,2,3],2:[4,5,7,8,10],3:[6,9]}
batch_size = 10
nan = float('nan')

vgroups = copy.deepcopy(groups)
probs, n_groups = get_probs(vgroups)
while not any(map(math.isnan,probs)):

    # sample group id based on probabilities
    group_id = np.random.choice(n_groups,1,p=probs)[0]
    # choose list of ids of that group
    ids = vgroups[group_id]

    batch_ids = ids[:batch_size]
    bs = len(batch_ids)
    print (bs)

    for i in batch_ids:
        vgroups[group_id].remove(i)

    lens = [len(v) for k,v in vgroups.items()]
    if np.sum(lens) == 0:
        break
    probs,n_groups = get_probs(vgroups)
