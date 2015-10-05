import theano
import theano.tensor as T
from theano import function
import numpy as np


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
