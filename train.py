import os
from os import path
import sys
import nltk
import time
import argparse
import itertools
import xlwt
from xlrd import open_workbook
from xlutils.copy import copy
import numpy as np
import theano as theano
import theano.tensor as T
from theano import printing

from utils import *
from ffn import *

_EMBEDDING_DIM = 50
_FEAT_EMBEDDING_DIM = 5
_HIDDEN_DIM = 360
_LABEL_DIM = 45

def contextWin(data_set, winsz):
    assert (winsz % 2) == 1
    assert winsz >= 1
    ngrams = list()

    for line in data_set:
        ngrams_line = [line[i:(i+winsz)] for i in range(len(line)-winsz+1)]
        assert len(ngrams_line) == len(line)-winsz+1
        ngrams+=ngrams_line

    return ngrams

def test_with_trained_model(model, data, feat, label, epoch, print_progress=False, load_model=False, write_to_file=False):

    if load_model:
        load_model_parameters_theano(model_para_output_file+"_%depochs.npz"%epoch, model)
    
    counter = 0
    total = len(data)
    assert len(data) == len(feat) == len(label)
    tic = time.time()
    
    result = list()
    tuples = list()

    for d, f in zip(data,feat):
        counter += 1

        result.append(model.predict([d],[f])[0])

        if print_progress:
            print '[testing] example %i >> %2.2f%%'%(counter,(counter+1)*100./total),'completed in %.2f (sec) <<\r'%(time.time()-tic),
            sys.stdout.flush()

    correct = [i for i, j in zip(result, label) if i == j]

    if write_to_file:
        tuples = [[d[2],r,l] for d,r,l in zip (data, label, result)]

    acc = len(correct)*100./total

    return acc, tuples

def main():

    ###################
    # Section 0 Setup #
    ###################

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01, action='store')
    parser.add_argument('--winsz', type=int, default=5, action='store')
    parser.add_argument('--epoch', type=int, default=40, action='store')
    parser.add_argument('--batchsize', type=int, default=50, action='store')
    parser.add_argument('--test_epoch', type=int, action='store')
    parser.add_argument('--cont_epoch', type=int, action='store')
    parser.add_argument('--print_loss', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--dev', action='store_true')
    parser.add_argument('--iscontinue', action='store_true')
    parser.add_argument('--print_progress', action='store_true')
    parser.add_argument('--write_to_existing_excel', action='store_true')
    parser.add_argument('--uselm', action='store_true')

    args = parser.parse_args()

    lr = np.float32(args.lr).astype(theano.config.floatX)
    winsz = args.winsz
    batch_size = args.batchsize
    test_epoch = args.test_epoch
    cont_epoch = args.cont_epoch
    train = args.train
    test = args.test
    dev = args.dev
    iscontinue = args.iscontinue
    print_progress = args.print_progress
    write_to_existing_excel = args.write_to_existing_excel
    uselm = args.uselm

    if write_to_existing_excel:
        rb = open_workbook(excel_filename)
        wb = copy(rb)
        ws = wb.get_sheet(0)
    else:
        wb = xlwt.Workbook()
        ws = wb.add_sheet('sheet')
        excelSetup(ws)

    #######################
    # Section 1 Read Data #
    #######################

    train_set,train_label,train_cap,test_set,test_label,test_cap,dev_set,dev_label,dev_cap,input_vocab,output_vocab,cap_vocab, input_vocab_words, output_vocab_words = readData(winsz)

    vocab_size = len(input_vocab)
    output_vocab_size = len(output_vocab)
    cap_vocab_size = len(cap_vocab)

    train_ngrams = contextWin(train_set, winsz)
    test_ngrams = contextWin(test_set,winsz)
    dev_ngrams = contextWin(dev_set,winsz)

    train_cap_ngrams = contextWin(train_cap, winsz)
    test_cap_ngrams = contextWin(test_cap, winsz)
    dev_cap_ngrams = contextWin(dev_cap, winsz)
    
    train_label = [i for sub in train_label for i in sub]
    test_label = [i for sub in test_label for i in sub]
    dev_label = [i for sub in dev_label for i in sub]

    assert len(train_ngrams) == len(train_cap_ngrams) == len(train_label)
    assert len(test_ngrams) == len(test_cap_ngrams) ==len(test_label)
    assert len(dev_ngrams) == len(dev_cap_ngrams) ==len(dev_label)

    print 'train_set: ', train_set[:10]
    print 'train_label: ', train_label[:10]
    print 'train_cap: ', train_cap[:10]
    print 'train_ngrams: ', train_ngrams[:10]
    print 'train_cap_ngrams: ', train_cap_ngrams[:10]
    print 'test_set: ', test_set[:10]
    print 'test_label: ', test_label[:10]
    print 'test_cap: ', test_cap[:10]
    print 'test_ngrams: ', test_ngrams[:10]
    print 'test_cap_ngrams: ', test_cap_ngrams[:10]
    print 'dev_set: ', dev_set[:10]
    print 'dev_label: ', dev_label[:10]
    print 'dev_cap: ', dev_cap[:10]
    print 'dev_ngrams: ', dev_ngrams[:10]
    print 'dev_cap_ngrams: ', dev_cap_ngrams[:10]
    print 'vocab size = %d' % vocab_size

    ####################################
    # Section 2 Initialize Model Graph #
    ####################################

    num_iter = len(train_ngrams) // batch_size + 1
    print "num_iter = %d" % num_iter
    
    model = FFNTheano(winsz, vocab_size, cap_vocab_size)

    if uselm:
        lookTb = np.load(infile)['embeddings']
        model.lt.set_value(lookTb)

    #########################
    # Section 3 Train Model #
    #########################

    if iscontinue:

        load_model_parameters_theano(model_para_output_file+"_%depochs.npz"%cont_epoch, model)

    if train:

        start_epoch = 1 if not iscontinue else cont_epoch+1

        for i in range(start_epoch, args.epoch+1):

            print "Training Epoch %d ......" %i
            total_loss = 0
            tic = time.time()

            for j in range(num_iter):

                if j == num_iter-1:
                    t_ngram = np.asarray(train_ngrams[j*batch_size:])
                    c_ngram = np.asarray(train_cap_ngrams[j*batch_size:])
                    label = np.eye(_LABEL_DIM)[(np.array(train_label[j*batch_size:])).tolist()]
                else:
                    t_ngram = np.asarray(train_ngrams[j*batch_size:(j+1)*batch_size])
                    c_ngram = np.asarray(train_cap_ngrams[j*batch_size:(j+1)*batch_size])
                    label = np.eye(_LABEL_DIM)[(np.array(train_label[j*batch_size:(j+1)*batch_size])).tolist()]

                total_loss += model.gradient_step(t_ngram, c_ngram, label, lr)

                if print_progress:
                    print '[learning] epoch %i >> %2.2f%%'%(i,(j+1)*100./num_iter),'completed in %.2f (sec) <<\r'%(time.time()-tic),
                    sys.stderr.flush()

            print "Loss after epoch %i: %f" %(i, total_loss)

            if dev:
                print "Testing on dev set for Epoch %d ......" %i
                accuracy = test_with_trained_model(model, dev_ngrams, dev_cap_ngrams, dev_label, i, print_progress, False, False)[0]
                print "Dev accuracy after %d epochs of training : %2.2f%%" %(i, accuracy)
                ws.write(i,0,accuracy)

            if i%5 == 0:
                print "Testing on training set for Epoch %d ......" %i
                accuracy = test_with_trained_model(model, train_ngrams, train_cap_ngrams, train_label, i, print_progress,False, False)[0]
                print "Training accuracy after %d epochs of training : %2.2f%%" %(i, accuracy)
                ws.write(i,1,accuracy)

            save_model_parameters_theano(model_para_output_file+"_%depochs"%i, model)
            print "Model after %d epochs is saved to %s" %(i, model_para_output_file+"_%depochs.npz"%i)

            print "Training completed for %d epochs in %.2f seconds" %(i, time.time() -tic)

            wb.save(excel_filename)

    ########################
    # Section 4 Test Model #
    ########################

    if test:

        accuracy, tuples = test_with_trained_model(model, test_ngrams, test_cap_ngrams, test_label, test_epoch, print_progress,True, True)
        print tuples[:10]
        print np.array(tuples).shape

        f = open(test_output_filename,'w')

        for i in tuples:
            w = input_vocab_words[i[0]]
            c = output_vocab_words[i[1]]
            g = output_vocab_words[i[2]]
            f.write("%s %s %s\n"%(w,c,g))

        f.close()

        ws.write(test_epoch,2,accuracy)
        ws.write(0,3,"Test Epoch")
        ws.write(test_epoch,3,test_epoch)

        print "Accuracy = %2.2f%%" %accuracy

        wb.save(excel_filename)

    wb.save(excel_filename)    

if __name__ == '__main__':
    main()
