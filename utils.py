import numpy as np
import theano as theano
import theano.tensor as T
import os, re
from os import path
import operator
import xlwt

###################
# Section 0 Setup #
###################

data_dir = path.abspath('/home/shamil/Workspace/lm/corelm_Mar2016/_data')

data_path = path.join(data_dir,'pos')
model_path = path.abspath('../model')
output_path = path.abspath('../output')

train_file = path.join(data_path,'train/train.conll.txt')
train_feat_file = path.join(data_path,'train.dat')
train_label_file = path.join(data_path,'train.label.dat')

test_file = path.join(data_path,'test/test.conll.txt')
test_feat_file = path.join(data_path,'test.dat')
test_label_file = path.join(data_path,'test.label.dat')

dev_file = path.join(data_path,'dev/dev.conll.txt')
dev_feat_file = path.join(data_path,'dev.dat')
dev_label_file = path.join(data_path,'dev.label.dat')

input_vocab_file = path.join(data_path,'input.vocab')
cap_vocab_file = path.join(data_path,'input.features.vocab')
output_vocab_file = path.join(data_path,'output.vocab')

model_para_output_file = path.join(model_path,'model')
# model_para_input_file = path.join(model_path, 'model.npz')
excel_filename = path.join(output_path,'stats.xls')
test_output_filename = path.join(output_path,'test_output.txt')
infile = path.abspath('../../data/lookuptable.npz')

_PADDING = '<s>'
_UNKNOWN = '<unk>'

def excelSetup (ws):

    ws.write(0,0,"Dev Acc")
    ws.write(0,1,"Train Acc")
    ws.write(0,2,"Test Acc")

#######################
# Section 1 Read Data #
#######################

def readData(winsz):

    def readDataSet(winsz,file):

        data_set = list()
        data_cap = list()

        with open(file) as f:
            for line in f:
                # padded_line = [re.sub(r'(-?\d+)','NUMBER',a) for a in line.lower().strip().split()]
                
                temp = line.strip().split()
                padded_line = [i.split('_')[0] for i in temp]
                data_cap_line = [i.split('_')[1] for i in temp]

                for i in range(winsz/2):
                    padded_line.insert(0,_PADDING)
                    data_cap_line.insert(0,_PADDING)
                    padded_line.append(_PADDING)
                    data_cap_line.append(_PADDING)

                data_cap.append(data_cap_line)
                data_set.append(padded_line)

        return data_set, data_cap

    def readLabel(file):
        
        label = list()
        
        with open(file) as f:
            for line in f:
                label.append(line.strip().split())

        return label

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

        return vocab, vocab_dict

    input_vocab, input_vocab_dict = readVocab(input_vocab_file,True)
    output_vocab, output_vocab_dict = readVocab(output_vocab_file,False)
    cap_vocab, cap_vocab_dict = readVocab(cap_vocab_file,False)

    input_vocab_index = range(1,len(input_vocab)+1)
    output_vocab_index = range(1,len(output_vocab)+1)
    cap_vocab_index = range(1,len(cap_vocab)+1)

    # Training Set
    train_set, train_cap = readDataSet(winsz,train_feat_file)
    train_index = [[input_vocab_dict[w] if w in input_vocab_dict else input_vocab_dict[_UNKNOWN] for w in t] for t in train_set]
    train_cap_index = [[cap_vocab.index(x) for x in l] for l in train_cap]
    train_label = readLabel(train_label_file)
    train_label_index = [[output_vocab.index(x) for x in l] for l in train_label]

    # Test Set
    test_set, test_cap = readDataSet(winsz,test_feat_file)
    test_index = [[input_vocab_dict[w] if w in input_vocab_dict else input_vocab_dict[_UNKNOWN] for w in t] for t in test_set]
    test_cap_index = [[cap_vocab.index(x) for x in l] for l in test_cap]
    test_label = readLabel(test_label_file)
    test_label_index = [[output_vocab.index(x) for x in l] for l in test_label]

    # Dev Set
    dev_set, dev_cap = readDataSet(winsz,dev_feat_file)
    dev_index = [[input_vocab_dict[w] if w in input_vocab_dict else input_vocab_dict[_UNKNOWN] for w in t] for t in dev_set]
    dev_cap_index = [[cap_vocab.index(x) for x in l] for l in dev_cap]
    dev_label = readLabel(dev_label_file)
    dev_label_index = [[output_vocab.index(x) for x in l] for l in dev_label]

    return train_index,train_label_index,train_cap_index,test_index,test_label_index,test_cap_index,dev_index,dev_label_index,dev_cap_index,input_vocab_index,output_vocab_index,cap_vocab_index,input_vocab, output_vocab

########################
# Section 2 Save Model #
########################

def save_model_parameters_theano(outfile, model):

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    window_size, word_dim, feat_dim, embedding_dim, hidden_dim, input_dim, label_dim, lt, cap_lt, w1, b1, w2, b2 = model.window_size, model.word_dim, model.feat_dim, model.embedding_dim, model.hidden_dim, model.input_dim, model.label_dim, model.lt.get_value(), model.cap_lt.get_value(), model.w1.get_value(), model.b1.get_value(), model.w2.get_value(), model.b2.get_value()
    np.savez(outfile, window_size=window_size, word_dim=word_dim, feat_dim=feat_dim, embedding_dim=embedding_dim, hidden_dim=hidden_dim, input_dim=input_dim, label_dim=label_dim, lt=lt, cap_lt=cap_lt, w1=w1, b1=b1, w2=w2, b2=b2)
    print "Saved model parameters to %s." % outfile

########################
# Section 3 Load Model #
########################

def load_model_parameters_theano(path, model):
    npzfile = np.load(path)

    window_size = npzfile["window_size"]
    word_dim = npzfile["word_dim"]
    feat_dim = npzfile["feat_dim"]
    embedding_dim = npzfile["embedding_dim"]
    hidden_dim = npzfile["hidden_dim"]
    input_dim = npzfile["input_dim"]
    label_dim = npzfile["label_dim"]
    lt = npzfile["lt"]
    cap_lt = npzfile["cap_lt"]
    w1 = npzfile["w1"]
    b1 = npzfile["b1"]
    w2 = npzfile["w2"]
    b2 = npzfile["b2"]

    model.window_size = window_size
    model.word_dim = word_dim
    model.feat_dim = feat_dim
    model.embedding_dim = embedding_dim
    model.hidden_dim = hidden_dim
    model.input_dim = input_dim
    model.label_dim = label_dim
    model.lt.set_value(lt)
    model.cap_lt.set_value(cap_lt)
    model.w1.set_value(w1)
    model.b1.set_value(b1)
    model.w2.set_value(w2)
    model.b2.set_value(b2)

    print "Loaded model parameters from\n \"%s\"\nwindow_size = %d, word_dim (vocab size) = %d, feat_dim = %d, embedding_dim = %d,  hidden_dim = %d, input_dim = %d, label_dim = %d\n" % (path, window_size, word_dim, feat_dim, embedding_dim, hidden_dim, input_dim, label_dim), "Dimensions of layers: shape of lt = %s, cap_lt = %s, w1 = %s, b1 = %s, w2 = %s, b2 = %s" % (lt.shape, cap_lt.shape, w1.shape, b1.shape, w2.shape, b2.shape,)

############################
# Section 4 Gradient Check #
############################
def gradient_check_theano(model, model_param, x, y, h=0.0001, error_threshold=0.01):
    # Overwrite the bptt attribute. We need to backpropagate all the way to get the correct gradient
    # model.bptt_truncate = 1000
    # Calculate the gradients using backprop
    print 'x: ', x
    print 'y: ', y

    gradients = model.calculate_gradients(x, y)
    # List of all parameters we want to chec.
    # model_parameters = ['U', 'V', 'W']
    # Gradient check for each parameter
    for pidx, pname in enumerate(model_param):
        # Get the actual parameter value from the mode, e.g. model.W
        parameter_T = operator.attrgetter(pname)(model)
        parameter = parameter_T.get_value()
        print "Performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape))
        # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
        it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            ix = it.multi_index
            # Save the original value so we can reset it later
            original_value = parameter[ix]
            # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
            parameter[ix] = original_value + h
            parameter_T.set_value(parameter)
            gradplus = model.calculate_loss(x,y)
            parameter[ix] = original_value - h
            parameter_T.set_value(parameter)
            gradminus = model.calculate_loss(x,y)
            estimated_gradient = (gradplus - gradminus)/(2*h)
            parameter[ix] = original_value
            parameter_T.set_value(parameter)
            # The gradient for this parameter calculated using backpropagation
            backprop_gradient = gradients[pidx][ix]
            # calculate The relative error: (|x - y|/(|x| + |y|))
            relative_error = np.abs(backprop_gradient - estimated_gradient)/(np.abs(backprop_gradient) + np.abs(estimated_gradient))
            # If the error is to large fail the gradient check
            if relative_error > error_threshold:
                print "Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix)
                print "+h Loss: %f" % gradplus
                print "-h Loss: %f" % gradminus
                print "Estimated_gradient: %f" % estimated_gradient
                print "Backpropagation gradient: ", backprop_gradient
                print "Relative Error: %f" % relative_error
                return 
            it.iternext()
        print "Gradient check for parameter %s passed." % (pname)
