import os
import sys
import timeit
import cPickle
import gzip


import theano
import theano.tensor as T
from theano import function
import numpy as np
from LogisticRegression import LogisticRegression
from HiddenLayer import HiddenLayer
from MLP import MLP

#----------------------------------------
#          DATA LOADING:
#----------------------------------------

#load data and split it into input/output for
#training, validation, and testing
def load_data(dataset):
    returnSet = []
    returnSet[0] = [(,)]
    returnSet[1] = [(,)]
    returnSet[2] = [(,)]

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    #download it if it doesnt exist
    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)
        
    print '... loading data'

    ########
    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    ########


    
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #whose row's correspond to an example.
    #target is a 1 dimensional np array / vector with length of
    #the number of rows in the input.
    #It should give the target to the example with the same
    #index in the input.    
    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables
        
        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')
    
    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)
    
    returnSet[0] = [train_set_x,train_set_y]
    returnSet[1] = [valid_set_x,valid_set_y]
    returnSet[2] = [test_set_x,test_set_y]

    
    return returnSet



#----------------------------------------
#          MODEL BUILDING:
#----------------------------------------


#build  training, testing AND validation models
def buildModel(n_hidden=500,batch_size=20,trainData,testData,validData):

    #initial (symbolic) variables shared everywhere
    index = T.lscalar() #index to minibatch
    x = T.matrix('x')   #input matrix
    y = T.ivector('y')  #output label vector
    rng = numpy.random.RandomState(1244)

    #the network:
    classifier = MLP(
        rng=rng,
        input=x,
        n_in=28*28, #TODO: change this to dimensions of x
        n_hidden=n_hidden,
        n_out=10    #TODO: change this to dimensions of y
        )

    #the cost function:
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    #create testing and validation models
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: testData[0][index*batch_size:(index+1)*batch_size],
            y: testData[1][index*batch_size:(index+1)*batch_size]
        }
    )
    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x:validData[0][index*batch_size:(index+1)*batch_size],
            y:validData[1][index*batch_size:(index+1)*batch_size]
        }
    )


    #cost gradient:
    gparams = [T.grad(cost,param) for param in classifier.params]
    #how to update parameters based on gradients:
    updates = [
        (param, param - leraning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]


    #the training model that will update itself
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: trainData[0][index * batch_size: (index + 1) * batch_size],
            y: trainData[1][index * batch_size: (index + 1) * batch_size]
        }
    )

    return [train_model,test_model,valid_model]

#actually train the model:
def trainModel(model,validationModel,testModel,n_train_batches,n_valid_batches,n_test_batches):
    patience = 10000
    patience_increase = 2
    improvement_threshold = 0.995
    #how often to validate:
    validation_frequency = min(n_train_batches, patience / 2)

    best_validation_loss = np.inf
    best_iter = 0
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = false


    #now actually loop:
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            #call the training model function, giving index to use for data
            #happens every loop
            minibatch_avg_cost = model(minibatch_index)

            
            iter = (epoch - 1) * n_train_batches + minibatch_index
            #if this is a validation epoch:
            if (iter+1) % validation_frequency == 0:
                validation_losses = [validationModel(i) for i in xrange(n_valid_batches)]
                thisValidationLoss = np.mean(validation_losses)

                print (
                    'epoch %i, minibatch %i/%i, validation error: %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        thisValidationLoss * 100.
                    )
                )

                #if the validation is the current best:
                if (thisValidationLoss < best_validation_loss):
                    if (thisValidationLoss < best_validation_loss * improvement_threshold):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    #test on test set:
                    test_losses = [testModel(i) for i in xrange(n_test_batches)]
                    test_score = np.mean(test_losses)

                    print(('		epoch %i, minibatch %i/%i, test error of best model: %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches, test_score * 100.))
                    
        if patience <= iter:
            done_looping = True
            break
            
    end_time = timeit.default_timer()
    print(('Optimization complete. Best Validation score of %f %%'
           'obtained at iteration %i, with test performance %f %%'
           'and ran in %.2fm') %
          (best_validation_loss * 100., best_iter+1, test_score * 100.,
           (end_time - start_time) / 60.))

    

def test_mlp(dataset,learning_rate=0.01, L1_reg=0.00,L2_reg=0.0001,
             n_epochs=1000,batch_size=20,n_hidden=500):
    
    #TODO: load and setup the datasets:
    loaded_dataset = load_data(dataset)
    #pairs of input + output data:
    trainData = dataset[0]
    validData = dataset[1]
    testData  = datasets[2]
    
    # compute number of minibatches for training, validation and testing
    n_train_batches = trainData[0].get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = validData[0].get_value(borrow=True).shape[0] / batch_size
    n_test_batches  = testData[0].get_value(borrow=True).shape[0] / batch_size

    trainingModel, testModel, validModel = buildModel(n_hidden,batch_size,trainData,testData,validData)
                                                   
    trainModel(trainingModel,validModel,testModel)

                                                   
if __name__ == '__main__':
    test_mlp("mnist.pkl.gz")
