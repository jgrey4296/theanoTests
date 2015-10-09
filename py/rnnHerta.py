#from http://www.christianherta.de/lehre/dataScience/machineLearning/neuralNetworks/recurrentNeuralNetworks.php
from __future__ import print_function
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

#input data:
import reberGrammar

dtype=theano.config.floatX
srng = RandomStreams(seed=235)

#setup values:
n_in = 7
n_hid = 10
n_out = 7

#the input:
#Matrix: a vector of timesteps of a vector of input values
v = T.matrix(dtype=dtype) 
#the expected output
target = T.matrix(dtype=dtype)

#Hyperparameters:
#learning rate:
lr = np.cast[dtype](0.2) #set to right type and initialise
learning_rate = theano.shared(lr) #turn it into a shared variable
#reminder: theano.shared copies the type of the value passed in

#training epochs:
nb_epochs = 100

#number of steps to sample:
sampleStepLength = 100


#setup weights appropriately:
def rescale_weights(values,factor=1.0):
    print("Rescaling weights:",values,factor)
    factor = np.cast[dtype](factor)
    _,svs,_ = np.linalg.svd(values)
    values = values / svs[0]
    return values

#initialise weights
def sample_weights(sizeX,sizeY):
    print("Sampling Weights:",sizeX,sizeY)
    values = np.ndarray([sizeX,sizeY],dtype=dtype)
    for dx in range(sizeX):
        vals = np.random.uniform(low=-1., high=1., size=(sizeY,))
        #if normalising?:
        #vals_norm = np.sqrt((vals ** 2 ).sum())
        #vals = vals / vals_norm
        values[dx,:] = vals
    _,svs,_ = np.linalg.svd(values)
    values = values / svs[0]
    return values

#parameter creation:
def get_parameters(n_in,n_out,n_hid):
    print("Creating Parameters")
    output_bias = theano.shared(np.zeros(n_out,dtype=dtype))
    hidden_bias = theano.shared(np.zeros(n_hid,dtype=dtype))
    hidden_state = theano.shared(np.zeros(n_hid,dtype=dtype))
    
    inputGate_weights = theano.shared(sample_weights(n_in,n_hid))
    hiddenGate_weights = theano.shared(sample_weights(n_hid,n_hid))
    outputGate_weights = theano.shared(sample_weights(n_hid,n_out))
    
    return inputGate_weights,hiddenGate_weights,hidden_bias,outputGate_weights,output_bias,hidden_state

w_ih, w_hh, b_h, w_ho, b_o, h0 = get_parameters(n_in,n_out,n_hid)
params = [w_ih,w_hh,b_h,w_ho,b_o,h0]

#squash to between 0 and 1
def logistic_function(x):
    return 1./(1+ T.exp(-x))


#a step of the network:
#input sequence: x_t,
#prior results : h_tm1
#constants:
#	input,hidden, and output weights: w_ih,w_hh,w_ho,
#	biases : b_h,b_o
print("Defining one_step")
def one_step(x_t,h_tm1,w_ih,w_hh,w_ho,b_h,b_o):
    hidden_state = T.tanh(theano.dot(x_t,w_ih) + theano.dot(h_tm1,w_hh) + b_h)
    output_state = theano.dot(hidden_state,w_ho) + b_o
    squashed_output = logistic_function(output_state)
    return [hidden_state,squashed_output]

#show how to process a sequence:
#updates = none
#the dict constucted is really as simple as it looks,
#specify what to use (input) and which to use from the input (taps)
[hidden_states, output_states], nonSpecifiedUpdates = theano.scan(fn=one_step,
                                                                  sequences = dict(input=v, taps=[0]),
                                                                  #o_i initialises first inptu, for h_tm1
                                                                  outputs_info = [h0, None], #return type of one_step
                                                                  non_sequences = [w_ih,w_hh,w_ho,b_h,b_o])

#define the cost function
#cross entropy loss
cost = -T.mean(target * T.log(output_states) + (1. - target) * T.log(1. - output_states))

#create the training function, setting up parameter adjustments:
def get_train_function(cost, v, target):
    print("Creating Training Function")
    #gradients:
    gparams = []
    for param in params:
        gparam = T.grad(cost, param)
        gparams.append(gparam)

    updates = []
    for param,gparam in zip(params,gparams):
        updates.append((param,param - gparam * learning_rate))

    trainFunction = theano.function(inputs=[v,target],
                                    outputs = cost,
                                    updates = updates)
    print("Training Function Created")
    return trainFunction

#actually create the training function
trainFunction = get_train_function(cost,v,target)

#------------------------------
# Training Data:
#------------------------------

#TODO:load data:
train_data = []
#todo: inspect the form of this, adapt loaded data to it
#should be a 3d matrix of examples.

#a list of examples
#each example is a pairing: actualSequence,sequenceOfPossibilities
#so: actualSequence: 'B'. poss: T/P
#or: AS: 'BT' poss: S/X
#but the actual sequence is more than one, so:
#AS  : B,   BT,  BTS, BTSX,BTSXS,BTSXSE
#POSS: T/P, S/X, S/X, X/S, E    ,0
train_data = reberGrammar.get_n_examples(1000)

#todo: load this first, figure out vocab size,
#and set layer sizes appropriately

#load file

#convert to list of sequences

#convert sequences to num vectors

#convert list to matrix


#create the training routine:
def train_routine(train_data, nb_epochs=50):
    print("Starting Training routine")
    train_errors = np.ndarray(nb_epochs)
    for x in range(nb_epochs):
        error = 0.
        for j in range(len(train_data)):
            index = np.random.randint(0,len(train_data))
            dataInput, trueOutput = train_data[index]
            train_cost = trainFunction(dataInput,trueOutput)
            print("Epoch: ",x, " Trained: ",j," cost: ",train_cost)
            error = train_cost
        train_errors[x] = error
    return train_errors


train_errors = train_routine(train_data, nb_epochs)

#------------------------------
# PREDICTION
#------------------------------
print("predicting")
#to predict:
predictionFunction = theano.function(inputs=[v],outputs = output_states)

#TODO:
inp, outp = reberGrammar.get_one_example(10)
prediction = predictionFunction(inp)

for p,o in zip(prediction,outp):
    print( "Predicted:",p)
    print( "Output:",o)
    print()


#------------------------------
# SAMPLING
#------------------------------
print("sampling")

#sampling:
def get_sample(probs):
    return srng.multinomial(n=1,pvals=probs,ndim=(1,7))


#no sequences,
#prior results of: h_tm1, y_tm1
#non-sequences of: w_ih,w_hh,w_ho,b_h,b_o 

def sampling_step(h_tm1,y_tm1,w_ih,w_hh,w_ho,b_h,b_o):
    hiddenState = T.tanh(theano.dot(y_tm1,w_ih) + theano.dot(h_tm1,w_hh) + b_h)
    currentState = theano.dot(hiddenState,w_ho) + b_o
    squashedState = logistic_function(currentState)
    normalisedState = squashedState / T.sum(squashedState)
    probabilityDist = get_sample(normalisedState)
    castForOutput = T.cast(probabilityDist,dtype)
    #keep going until a terminal symbol in original dataset is reached:
    return [hiddenState,castForOutput], theano.scan_module.until(T.eq(T.argmax(castForOutput),6))


#create the start symbol:
startValues = np.array([1.,0.,0.,0.,0.,0.,0.],dtype=dtype)
startState = theano.shared(startValues)

#create the sampling scan:
[hiddenStates,outputState], updates = theano.scan(fn=sampling_step,
                                                  sequences=[],
                                                  n_steps = sampleStepLength,
                                                  outputs_info=[h0,startState],
                                                  non_sequences=[w_ih,w_hh,w_ho,b_h,b_o])

#create the main sampling function
sampleFunction = theano.function(inputs=[],outputs=outputState, updates=updates)

#now print out the results of sampling:
for i in range(20):
    sampled = sampleFunction()
    print(sampled.shape)
    sample_sequence = np.concatenate((np.array([[1.,0.,0.,0.,0.,0.,0.]],dtype=dtype), sampled), axis=0)
    word = reberGrammar.sequenceToWord(sample_sequence)
    print("Generated Word:",word,"   -> ",reberGrammar.in_grammar(word))
