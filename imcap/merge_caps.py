import json
import os

COCO_ROOT = '/seq/segmentation/COCO/tools'
val = json.load(open(os.path.join(COCO_ROOT,'annotations/captions_val2014.json'), 'r'))
train = json.load(open(os.path.join(COCO_ROOT,'annotations/captions_train2014.json'), 'r'))

print len(val)
print len(train),type(train)

for k in val.keys():
    try:
        val[k].extend(train[k])
    except:
        print "Not a list."

json.dump(val, open('./utils/captions_merged.json', 'w'))
