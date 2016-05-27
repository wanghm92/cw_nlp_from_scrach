import numpy as np
from os import path
filename = path.abspath('../../data/embeddings.txt')
outfile = path.abspath('../../data/lookuptable')
infile = path.abspath('../../data/lookuptable.npz')
infile2 = path.abspath('../model/model_20epochs.npz')

tmp = list()
def readEmbeddings(file):
    with open(file) as f:
        for line in f:
            tmp.append(line.strip().split())

    print len(tmp)
    print len(tmp[0])
    return np.array(tmp,dtype=np.float32)

embeddings = readEmbeddings(filename)
print embeddings.shape
print embeddings.dtype

# np.savez(outfile, embeddings=embeddings)

# inembedding = np.load(infile)['embeddings']
lt = np.load(infile2)['lt']

print lt[0:1].shape
print lt[0:1].dtype

newem = np.concatenate((lt[0:2], embeddings), axis=0)

print newem.shape
print newem.dtype

print newem[-4:-1]
print embeddings[-4:-1]
print newem[1:4]
print embeddings[1:4]

print newem.dtype

np.savez(outfile, embeddings=newem)