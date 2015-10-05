import theano
import theano.tensor as T
from theano import function
import numpy as np

#have a string of letters

#convert to a list of indices

#convert to an embedding of [0 for x in len(indices)]
#embedding[i] = 1
#this is a numpy array

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
            )
            name='b',
            borrow=True
        )

        #computing the class membership:
        self.p_y_given_x = T.nnet.softmax(T.dot(input,self.W)+self.b)

        #get the maximum value from axis 1 of the output of pygx
        self.y_pred = T.argmax(self.p_y_given_x,axis=1)

        self.params = [self.W, self.b]
        self.input = input

    def negLogLikelihood(self,y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]),y])

    def errors(self,y):
        if y.ndim != self.y_pred_ndim:
            raise TypeError(
                "y should have the same shape as self.y_pred",
                ('y',y.type,'y_pred',self.y_pred.type))

        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred,y))
        else:
            raise NotImplementedError()
        
#end of log regression class

class HiddenLayer(object):
    def __init__(self,rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):
        self.input = input

        if W is None:
            #create random weights
            #for weight matrix
            w_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                    ),
                dtype=theano.config.floatX
                )
            if activation == theano.tensor.nnet.sigmoid:
                #if sigmoid, scale up?
                w_values *= 4

            #actual weight matrix:
            W = theano.shared(value=W_values, name='W',borrow=True)

            if b is None:
                #if no bias vector create a zero vector
                b_values = numpy_zeros((n_out,),dtype=theano.config.floatX)
                b = theano.shared(value=b_values,name='b',borrow=True)

            #actually store the W and b
            self.W = W
            self.b = b

            #Setup the activation function:
            #first the non-linear, pre-activation function
            lin_ouput = T.dot(input, self.W) + self.b
            self.output = (
                lin_output if activation is None
                else activation(lin_output)
                )

            self.params = [self.W,self.b]
#Hidden Layer finished


class MLP(object):
    def __init__(self,rng,input,n_in,n_hidden,n_out):
        #first create the hidden layer:
        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh
        )


        #output layer:
        self.logRegressionLayer = LogisticRegression(
            input = self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
        )

        #setup regularization
        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.logRegressionLayer.W).sum()
            )

        self.L2_sqr = (
            (self.hiddenLayer.@ ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )

        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
            )

        self.errors = self.logRegressionLayer.errors
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params
        self.input = input

        
