from os import path
import operator

data_dir = path.abspath('/home/shamil/Workspace/lm/corelm_Mar2016/_data')
data_path = path.join(data_dir,'pos')
input_vocab_file = path.join(data_path,'input.vocab')
new_vocab_file = path.abspath('../../data/words.lst')
train_file = path.join(data_path,'train/train.conll.txt')
_PADDING = '<s>'
def readDataSet(winsz,file):

    data_set = list()
    data_cap = list()

    with open(file) as f:
        for line in f:
            # padded_line = [re.sub(r'(-?\d+)','NUMBER',a) for a in line.lower().strip().split()]
            
            padded_line = line.strip().split()
            # padded_line = [i.split('_')[0] for i in temp]
            # data_cap_line = [i.split('_')[1] for i in temp]

            for i in range(winsz/2):
                padded_line.insert(0,_PADDING)
                # data_cap_line.insert(0,_PADDING)
                padded_line.append(_PADDING)
                # data_cap_line.append(_PADDING)

            # data_cap.append(data_cap_line)
            data_set.append(padded_line)

    return data_set


def readVocab(file,isInput):

    vocab = list()
    vocab_dict = dict()
    i = 0
    with open(file) as f:  
        for line in f:
            if isInput:
                line = line.strip()
            else:
                line = line.strip().split()[0]

            vocab.append(line)
            vocab_dict[line] = i
            i+=1

    return sorted(vocab), vocab_dict

input_vocab, input_vocab_dict = readVocab(input_vocab_file,True)
new_vocab, new_vocab_dict = readVocab(new_vocab_file,True)

print input_vocab[:10]
print new_vocab[:10]
print len(input_vocab)
print len(new_vocab)

flag = True
l1 = dict()
for i in input_vocab:
    if i not in new_vocab_dict:
        l1[i] = input_vocab.index(i)
        flag = False

print flag
print len(l1)
print sorted(l1.items(), key=operator.itemgetter(1))

flag = True
l2 = dict()
for i in new_vocab:
    if i not in input_vocab_dict:
        l2[i] = new_vocab.index(i)
        flag = False

print flag
print len(l2)
print sorted(l2.items(), key=operator.itemgetter(1))

train_set = readDataSet(5,train_file)
tset = [i for sent in train_set for i in sent]

print tset[:10]

c = dict()
for i in l2:
    print i
    c[i] = tset.count(i)

print c

