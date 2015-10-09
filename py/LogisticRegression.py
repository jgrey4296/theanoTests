import theano
import theano.tensor as T
from theano import function
import numpy as np


class LogisticRegression(object):
    def __init__(self,input,n_in,n_out):
        self.W = theano.shared(
            value=np.zeros(
                (n_in,n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )

        self.b = theano.shared(
            value=np.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        #computing the class membership,
        #ie: all probabilities of classes
        #produces a matrix of [inputExample,[probabilities for class]]
        #input === x, so a matrix of examples
        #returns matrix of class probabilities
        self.p_y_given_x = T.nnet.softmax(T.dot(input,self.W)+self.b)

        #gets the maximum predicted value index for each
        #axis 1=rows
        self.y_pred = T.argmax(self.p_y_given_x,axis=1)

        self.params = [self.W, self.b]
        self.input = input

    def negLogLikelihood(self,y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]),y])

    def errors(self,y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                "y should have the same shape as self.y_pred",
                ('y',y.type,'y_pred',self.y_pred.type))

        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred,y))
        else:
            raise NotImplementedError()
