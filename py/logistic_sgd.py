import cPickle
import gzip
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T


#Logistic Regression classifier:

#general values:
n_in = 20
n_out = 20

#the input to use:
input = T.matrix('x')

#####################
#weights:
W = theano.shared(
    value=numpy.zeros( #value of W is an array of zeros
        (n_in,n_out), #dimensions
        dtype=theano.config.floatX #numpy array type
    ),
    name='W',
    borrow=True #borrow aliases, but wont update on the GPU
)


b = theano.shared(
    value=numpy.zeros(
        (n_out,),
        dtype= theano.config.floatX
    ),
    name='b',
    borrow=True
)
############

#activation function:
#mult and sum the input with W
p_y_given_x = T.nnet.softmax(T.dot(input,W) + b)

#prediction function:
#get the maximum across the second axis?
y_pred = T.argmax(self.p_y_given_x,axis=1)

#note: a[t.arange(y.shape[0]),y] -> forall rows of a, get column y
likelihood = T.mean(T.log(self_p_y_given_x)[T.arange(y.shape[0]),y])
loss = -likelihood

#to check the number of errors of the model:
#given y, get the number of y that don't match the prediction:
errors = T.mean(T.neq(y_pred,y))

###################################
#dataset notes:
#a dataset is a tuple(input,target)
#input = matrix[examples,features]
#target = vector() of indices of classes for each row of input
#target.length === input.shape[0]
data_xy = (np.array([[1,2,3],[4,5,6]]), [1,2])

data_x, data_y = data_xy
#store everything as a float for GPU usage:
shared_x = theano.shared(numpy.asarray(data_x,
                                       dtype=theano.config.floatX),
                         borrow=borrow)
shared_y = theano.shared(numpy.asarray(data_y,
                                       dtype=theano.config.floatX),
                         borrow=borrow)
#case y to int32 for class comparisons, float doesnt make any sense
usable_y = T.cast(shared_y, 'int32')



#Stochastic Gradient Descent:

#parameters:
learning_rate = 0.13
n_epochs = 1000
batch_size=600


#don't forget to divide up into training, testing, and validation sets


# compute number of minibatches for training, validation and testing
n_train_batches = shared_x.get_value(borrow=True).shape[0] / batch_size
#n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
#n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size



######################
# BUILD ACTUAL MODEL #
######################

# allocate symbolic variables for the data
index = T.fscalar()  # index to a [mini]batch

# generate symbolic variables for input (x and y represent a
# minibatch)
x = T.matrix('x')  # data, presented as rasterized images
y = T.ivector('y')  # labels, presented as 1D vector of [int] labels

# compute the gradient of cost with respect to theta = (W,b)
g_W = T.grad(cost=cost, wrt=classifier.W)
g_b = T.grad(cost=cost, wrt=classifier.b)

# specify how to update the parameters of the model as a list of
# (variable, update expression) pairs.
updates = [(W,W - learning_rate * g_W),
           (b, b - learning_rate * g_b)]

#a function to train a model
#outputs the loss amount, but also updates the model,
#and subs in actual values for x and y
train_model = theano.function(
    inputs=[index],
    outputs=loss,
    updates=updates,
    givens={
        x: train_set_x[index * batch_size: (index + 1) * batch_size],
        y: train_set_y[index * batch_size: (index + 1) * batch_size]
    }
)

#use the model, but don't update it:
# predict_using_model = theano.function(
#     inputs=[index],
#     outputs=loss,
#     givens={
#         x: 
#     })





###############
# TRAIN MODEL #
###############
patience = 5000  # look as this many examples regardless
patience_increase = 2  # wait this much longer when a new best is
improvement_threshold = 0.995  # a relative improvement of this much is considered significant
validation_frequency = min(n_train_batches, patience / 2) #how many batches before validation

best_validation_loss = numpy.inf
test_score = 0.
start_time = timeit.default_timer()

done_looping = False
epoch = 0

#while there are more epochs:
while (epoch < n_epochs) and (not done_looping):
    epoch = epoch + 1
    #for each minibatch:
    for minibatch_index in xrange(n_train_batches):
        #train the model for the minibatch
        minibatch_avg_cost = train_model(minibatch_index)
        # iteration number
        iter = (epoch - 1) * n_train_batches + minibatch_index

        #validate the model:
        if (iter + 1) % validation_frequency == 0:
            # compute zero-one loss on validation set
            validation_losses = [validate_model(i)
                                 for i in xrange(n_valid_batches)]
            this_validation_loss = numpy.mean(validation_losses)

            #print information about the training:
            print(
                'epoch %i, minibatch %i/%i, validation error %f %%' %
                (
                    epoch,
                    minibatch_index + 1,
                    n_train_batches,
                    this_validation_loss * 100.
                )
            )

            # if we got the best validation score until now, try on the TEST SET
            if this_validation_loss < best_validation_loss:
                #improve patience if loss improvement is good enough
                if this_validation_loss < best_validation_loss *  \
                   improvement_threshold:
                    patience = max(patience, iter * patience_increase)
                    
                best_validation_loss = this_validation_loss
                # test it on the test set
                test_losses = [test_model(i)
                               for i in xrange(n_test_batches)]
                test_score = numpy.mean(test_losses)

                print(
                    (
                        '     epoch %i, minibatch %i/%i, test error of'
                        ' best model %f %%'
                    ) %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        test_score * 100.
                    )
                )

                #to save the model, get the weights and bias's HERE

                if patience <= iter:
                    done_looping = True
                break

end_time = timeit.default_timer()
#print information about the overall training:
print(
    (
        'Optimization complete with best validation score of %f %%,'
        'with test performance %f %%'
    )
    % (best_validation_loss * 100., test_score * 100.)
)
print 'The code run for %d epochs, with %f epochs/sec' % (
    epoch, 1. * epoch / (end_time - start_time))
print >> sys.stderr, ('The code for file ' +
                      os.path.split(__file__)[1] +
                      ' ran for %.1fs' % ((end_time - start_time)))


