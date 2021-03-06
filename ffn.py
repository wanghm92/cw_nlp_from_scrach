import numpy as np
import theano as theano
import theano.tensor as T
from collections import OrderedDict

_EMBEDDING_DIM = 50
_FEAT_EMBEDDING_DIM = 5
_HIDDEN_DIM = 360
_LABEL_DIM = 45

class FFNTheano:
    
    def __init__(self, window_size, word_dim, feat_dim, embedding_dim=_EMBEDDING_DIM, feat_embedding_dim=_FEAT_EMBEDDING_DIM, hidden_dim=_HIDDEN_DIM, label_dim=_LABEL_DIM):

        np.random.seed(0)

        input_dim = window_size * (embedding_dim+feat_embedding_dim)

        self.window_size = window_size
        self.word_dim = word_dim # vocab size
        self.feat_dim = feat_dim # feature vocab size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.label_dim = label_dim
        
        # lt = np.random.randn(word_dim, embedding_dim)
        # lt = np.random.uniform(-np.sqrt(12./np.sqrt(word_dim)), np.sqrt(12./np.sqrt(word_dim)),(word_dim, embedding_dim))
        lt = np.random.uniform(-np.sqrt(1./np.sqrt(word_dim)), np.sqrt(1./np.sqrt(word_dim)),(word_dim, embedding_dim))

        cap_lt = np.random.uniform(-np.sqrt(1./np.sqrt(feat_dim)), np.sqrt(1./np.sqrt(feat_dim)),(feat_dim, feat_embedding_dim))

        # w1 = np.random.uniform(-np.sqrt(1./input_dim), np.sqrt(1./input_dim), (input_dim, hidden_dim))
        w1 = np.random.uniform(-np.sqrt(12./np.sqrt(input_dim)), np.sqrt(12./np.sqrt(input_dim)), (input_dim, hidden_dim))
        b1 = np.zeros(hidden_dim)
        # w2 = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, label_dim))
        w2 = np.random.uniform(-np.sqrt(12./np.sqrt(hidden_dim)), np.sqrt(12./np.sqrt(hidden_dim)), (hidden_dim, label_dim))
        b2 = np.zeros(label_dim)
        
        self.lt = theano.shared(name='lt', value=lt.astype(theano.config.floatX))
        self.cap_lt = theano.shared(name='cap_lt', value=cap_lt.astype(theano.config.floatX))
        self.w1 = theano.shared(name='w1', value=w1.astype(theano.config.floatX))      
        self.b1 = theano.shared(name='b1', value=b1.astype(theano.config.floatX))
        self.w2 = theano.shared(name='w2', value=w2.astype(theano.config.floatX))      
        self.b2 = theano.shared(name='b2', value=b2.astype(theano.config.floatX))

        self.dims = np.array([1, 1, self.input_dim, self.input_dim, self.hidden_dim, self.hidden_dim], dtype=np.float32)
        self.params = [self.lt, self.cap_lt, self.w1, self.b1, self.w2, self.b2]

        self.__theano_build__()

    def __theano_build__(self):

        train_ngram = T.imatrix('train_ngram')
        cap_ngram = T.imatrix('cap_ngram')
        train_label = T.imatrix('train_label')
        lr = T.fscalar('lr')
        
        # word_vectors = self.lt[train_ngram].reshape((train_ngram.shape[0], self.input_dim))
        word_vectors = self.lt[train_ngram]
        cap_vectors = self.cap_lt[cap_ngram]

        input_vectors = T.concatenate([word_vectors,cap_vectors], axis=-1).reshape((train_ngram.shape[0], self.input_dim))
        
        # z1 = T.dot(word_vectors, self.w1) + self.b1
        z1 = T.dot(input_vectors, self.w1) + self.b1
        a1 = T.tanh(z1)
        z2 = T.dot(a1, self.w2) + self.b2
        y_hat = T.nnet.softmax(z2)

        prediction = T.argmax(y_hat, axis=1)

        loss = T.nnet.categorical_crossentropy(y_hat,train_label).mean()
        # loss = -T.log(y_hat)[train_label]

        gradients = T.grad(loss, self.params)

        # Variable Learning Rate for each layer
        # updates = OrderedDict(( p, p-(lr/dim)*g ) for p, g, dim in zip(self.params , gradients, self.dims))
        # Fixed Learning Rate for all layers
        updates = OrderedDict(( p, p-lr*g ) for p, g in zip(self.params , gradients))

        self.forward_prop = theano.function([train_ngram, cap_ngram],y_hat)
        self.predict = theano.function([train_ngram, cap_ngram],prediction)
        self.calculate_loss = theano.function([train_ngram, cap_ngram, train_label],loss)
        self.calculate_gradients = theano.function([train_ngram, cap_ngram, train_label],gradients)

        # GPU NOTE: Removed the input values to avoid copying data to the GPU.
        self.gradient_step = theano.function(
            [train_ngram, cap_ngram, train_label, lr], outputs = loss, allow_input_downcast=True, updates = updates)
        # , train_label, lr