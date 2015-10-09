import theano
import theano.tensor as T
from theano import function
import numpy as np

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
                #if sigmoid, scale up? why?
                w_values *= 4

            #actual weight matrix:
            #creates the shared weight matrix, with the same
            #type as w_values 
            W = theano.shared(value=w_values, name='W',borrow=True)

            if b is None:
                #if no bias vector create a zero vector
                b_values = np.zeros((n_out,),dtype=theano.config.floatX)
                b = theano.shared(value=b_values,name='b',borrow=True)

            #actually store the W and b
            self.W = W
            self.b = b

            #Setup the activation function:
            #first the non-linear, pre-activation function
            lin_output = T.dot(input, self.W) + self.b
            self.output = (
                lin_output if activation is None
                else activation(lin_output)
                )

            self.params = [self.W,self.b]
#Hidden Layer finished
