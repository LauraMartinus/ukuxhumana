
#open enzu en
#open enzu zu

# append <zu> en to all.en
# appen zu to all.all

import sys
import os
import glob
import codecs

if len(sys.argv) < 4:
    print("usage: python compress_datasets_for_t2t.py <root_folder> <language1> <language2>")
root_folder = sys.argv[1]
L1 = sys.argv[2]
L2 = sys.argv[3]

root_folder = os.path.join(root_folder, "%s_%s" % (L1, L2)) 
train_en_input = '%s%s_parallel.train.%s' % (L1,L2,L1)
train_t_input = '%s%s_parallel.train.%s' % (L1,L2,L2)
dev_en_input = '%s%s_parallel.dev.%s' % (L1,L2,L1)
dev_t_input = '%s%s_parallel.dev.%s' % (L1,L2,L2)

train_en_path = os.path.join(root_folder, train_en_input)
train_t_path = os.path.join(root_folder, train_t_input)
dev_en_path = os.path.join(root_folder, dev_en_input)
dev_t_path = os.path.join(root_folder, dev_t_input)

eng_train = codecs.open('all_parallel.train.'+L1,'a','utf-8')
t_train = codecs.open('all_parallel.train.all','a','utf-8')
eng_dev = codecs.open('all_parallel.dev.'+L1,'a','utf-8')
t_dev = codecs.open('all_parallel.dev.all','a','utf-8')

en_transcriptions = [line.rstrip('\n') for line in codecs.open(train_en_path, "r", "utf-8")]
t_transcriptions = [line.rstrip('\n') for line in codecs.open(train_t_path, "r", "utf-8")]
for counter in range(len(en_transcriptions)):
    eng_train.write(u''.join("<"+L2+"> "+(en_transcriptions[counter])))
    t_train.write(u''.join((t_transcriptions[counter])))

en_transcriptions = [line.rstrip('\n') for line in codecs.open(dev_en_path, "r", "utf-8")]
t_transcriptions = [line.rstrip('\n') for line in codecs.open(dev_t_path, "r", "utf-8")]
for counter in range(len(en_transcriptions)):
    eng_dev.write(u''.join("<"+L2+"> "+(en_transcriptions[counter])))
    t_dev.write(u''.join((t_transcriptions[counter])))