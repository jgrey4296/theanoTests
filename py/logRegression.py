import cPickle
import gzip
import os
import sys
import timeit
import numpy
import theano
import theano.tensor as T

##Log Regression
# A Classifier, of:
# x = input vector, i and j as  elements of Y (potential classes),
# w = weights, b = bias
#  probability(i| x,w,b) = softmax(Wx+b)
#                        = e^Wi*x+b / sum_of_j(e^Wjx+bj)


class LogisticRegression(objecT):

    #input: type description of the input
    #n_in: dimension of input vector,
    #n_out: dimension of output class possibilities
    def __init__(self,input,n_in,n_out):
        #initialise weights of W as matrix of shape (n_in, n_out):
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in,n_out),
                dtype=theano.config.floatX
            ),
            name="W",
            borrow=True
        )

        #initialise bias' similarly
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name="b",
            borrow=True
        )
        
        #computing the probability for a class:
        self.p_g_given_x = T.nnet.softmax(T.dot(input,self.W) + self.b)

        #softmax could be written explicitly as?
        
        
        #compute all class probabilities, select the max:
        self.y_pred = T.argmax(self.py_y_given_x, axis=1)

        #the parameters of the model
        self.params = [self.W,self.b]
        #the model input:
        self.input = input

    #at this point, we have the predictor, but only has default weights and biases

    #first, define a loss function:
    #y is a vector of correct labels
    def negative_log_likelihood(self,y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]),y])


    #calculate number of errors in the minibatch / total examples
    def errors(self,y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError("incorrect shape")

        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred,y))
        else:
            raise NotImplementedError()
    




