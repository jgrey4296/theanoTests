#from http://www.christianherta.de/lehre/dataScience/machineLearning/neuralNetworks/recurrentNeuralNetworks.php
from __future__ import print_function
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import codecs
from textblob import TextBlob
import logging

logging.basicConfig(filename="../data/last_rnnOutput.log",level=logging.DEBUG)

dtype=theano.config.floatX
srng = RandomStreams(seed=235)


#------------------------------
# Training Data:
# Loaded first, to setup the network sizes
#------------------------------

#load data:
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
with codecs.open("../data/Horus Rising - Dan Abnett.txt","r","utf-8") as f:
    blob = TextBlob(f.read())

#takes a blob, returns a new blob
def preprocess(blob):
    return TextBlob("".join(filter(lambda x: x!=u'\n',blob)).lower())

lowerCaseBlob = preprocess(blob)

sentences = lowerCaseBlob.sentences
vocabChars = list(set(lowerCaseBlob)) #set to filter duplicates
#figure out the index of a full stop:
sentenceStop = 0
for index,item in enumerate(vocabChars):
    if item == ".":
        sentenceStop = index


print("Vocab Size:",len(vocabChars))

trainingData = []

for sent in sentences:
    sequence = []
    nextOutput = []
    for char in sent:
        sequence.append(char)
    for char in sent[1:]:
        nextOutput.append(char)
    nextOutput.append(sent[-1])#to make them the same length
    trainingData.append((sequence,nextOutput))
#at this point we have an array of examples
#where examples are tuples of (inputDataSequence,outputSequence)

#convert sequences to num vectors
def charToOneHot(char):
    oneHot = np.zeros(len(vocabChars),dtype=dtype)
    for index,item in enumerate(vocabChars):
        if(item == char):
            oneHot[index] = 1.0
            return oneHot
    raise LookupError("Char not found in vocab",char)

#and the inverse:
def oneHotToChar(oneHot):
    maxIndex = np.argmax(oneHot)
    return vocabChars[maxIndex]

def sequenceToText(sequence):
    print("Entering sequence to text")
    string = ""
    for x in sequence:
        string += oneHotToChar(x)
    return string

#now convert the data
numericTrainingData = []
for sequence,output in trainingData:
    numericInput = []
    numericOutput = []
    for char in sequence:
        numericInput.append(charToOneHot(char))
    for char in output:
        numericOutput.append(charToOneHot(char))
    numericTrainingData.append((numericInput,numericOutput))

#at this point, all the training data should be loaded into numericTrainingData
    
#convert to matrix? reber isnt?

#--------------------
# NETWORK SETUP:
#--------------------

#network sizes
n_in = len(vocabChars)
n_hid = int(len(vocabChars) * 0.8)
n_out = len(vocabChars)

#the input:
#Matrix: a vector of timesteps of a vector of input values
v = T.matrix(dtype=dtype) 
#the expected output
target = T.matrix(dtype=dtype)

#----------------------------------------
#Hyperparameters:
#--------------------
#learning rate:
lr = np.cast[dtype](0.4) #set to right type and initialise
learning_rate = theano.shared(lr) #turn it into a shared variable
#reminder: theano.shared copies the type of the value passed in

#training epochs:
nb_epochs = 500

#number of steps to sample:
sampleStepLength = 50

#squash to between 0 and 1
#def logistic_function(x):
sigma = lambda x: 1./(1+ T.exp(-x))
actF = T.tanh
#cross entropy loss:
def getCost(inV,expected):
    return -T.mean(expected * T.log(inV)+(1.-expected)*T.log(1.-inV))


#---------- END OF HYPERPARAMETERS

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
n_i = n_c = n_o = n_f = n_hid
print("Creating Parameters")
#weights:
#input:xt1,htm1,ctm1
input_xt1 = theano.shared(sample_weights(n_in,n_i))
input_htm1 = theano.shared(sample_weights(n_hid,n_i))
input_ctm1 = theano.shared(sample_weights(n_c,n_i))
#forget:xt,htm1,ctm1
forget_xt = theano.shared(sample_weights(n_in,n_f))
forget_htm1 = theano.shared(sample_weights(n_hid,n_f))
forget_ctm1 = theano.shared(sample_weights(n_c,n_f))
#cell:xt,htm1
cell_xt = theano.shared(sample_weights(n_in,n_c))
cell_htm1 = theano.shared(sample_weights(n_hid,n_c))
#output:xt,htm1,ct
out_xt = theano.shared(sample_weights(n_in,n_o))
out_htm1 = theano.shared(sample_weights(n_hid,n_o))
out_ct = theano.shared(sample_weights(n_c,n_o))
#squash:ht
hidden_ht = theano.shared(sample_weights(n_hid,n_out))

#biases:
#input,forget,cell,output,hiden
b_i = theano.shared(np.cast[dtype](np.random.uniform(-0.5,.5,size = n_i)))
b_f = theano.shared(np.cast[dtype](np.random.uniform(0, 1.,size = n_f)))
b_c = theano.shared(np.zeros(n_c, dtype=dtype))
b_o = theano.shared(np.cast[dtype](np.random.uniform(-0.5,.5,size = n_o)))
b_out = theano.shared(np.zeros(n_out, dtype=dtype))

#sequenceState:
c0 = theano.shared(np.zeros(n_hid,dtype=dtype))
h0 = T.tanh(c0)

params = [
    input_xt1,input_htm1,input_ctm1,forget_xt,forget_htm1,forget_ctm1,cell_xt,cell_htm1,out_xt,out_htm1,out_ct,hidden_ht,b_i,b_f,b_c,b_o,b_out
]

#MODIFIED FOR LSTM:

#a step of the network:
#input sequence: x_t,
#prior results : h_tm1, c_tm1,
#constants: 
#       W_[input,forget]gate[xt,htm1,ctm1], B_[input,forget]
#       W_[cell][xt,htm1], B_cell
#       W_output[xt,htm1,ct]gate, B_output
#       W_squash_ht, b_ht

print("Defining one_step")
def one_step(x_t,h_tm1,c_tm1,
             w_ixt,w_ihtm1,w_ictm1,b_i,
             w_fxt,w_fhtm1,w_fctm1,b_f,
             w_cxt,w_chtm1,b_c,
             w_oxt,w_ohtm1,w_oct,b_o,
             w_sht,b_ht):

    inputGate_t = sigma(theano.dot(x_t,w_ixt) + theano.dot(h_tm1,w_ihtm1) + theano.dot(c_tm1,w_ictm1) + b_i)
    forgetGate_t = sigma(theano.dot(x_t,w_fxt) + theano.dot(h_tm1,w_fhtm1) + theano.dot(c_tm1,w_fctm1) + b_f)
    cellState_t = forgetGate_t * c_tm1 + inputGate_t * actF(theano.dot(x_t,w_cxt) + theano.dot(h_tm1,w_chtm1) + b_c)
    outGate_t = sigma(theano.dot(x_t,w_oxt) + theano.dot(h_tm1,w_ohtm1) + theano.dot(cellState_t,w_oct) + b_o)
    hidden_t = outGate_t * actF(cellState_t)
    outState_t = sigma(theano.dot(hidden_t,w_sht) + b_ht)
    return [hidden_t,cellState_t,outState_t]
    
#show how to process a sequence:
#updates = none
#the dict constucted is really as simple as it looks,
#specify what to use (input) and which to use from the input (taps)
[hidden_states, _, output_states], _ = theano.scan(fn=one_step,\
                                                   sequences = dict(input=v, taps=[0]),
                                                   #o_i initialises first inptu, for h_tm1
                                                   outputs_info = [h0, c0, None], #return type of one_step
                                                   non_sequences = [
                                                       input_xt1,input_htm1,input_ctm1,b_i,
                                                       forget_xt,forget_htm1,forget_ctm1,b_f,
                                                       cell_xt,cell_htm1,b_c,
                                                       out_xt,out_htm1,out_ct,b_o,
                                                       hidden_ht,b_out]
)

#define the cost function
#cross entropy loss
#cost = -T.mean(target * T.log(output_states) + (1. - target) *# T.log(1. - output_states))

#create the training function, setting up parameter adjustments:
def get_train_function(v, target):
    print("Creating Training Function")
    cost = getCost(output_states,target)
    #gradients:
    gparams = []
    for param in params:
        #print(param)
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
trainFunction = get_train_function(v,target)



#------------------------------
#PREDICTION
#------------------------------
print("predicting")
#to predict, just get the output instead of the cost,
#and don't update
predictionFunction = theano.function(inputs=[v],outputs = output_states)

def testPrediction():
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
#PAY ATTENTION TO THE SHIFTED ORDER OF THE FIRST LINE:
def sampling_step(h_tm1,c_tm1,x_t,
                  w_ixt,w_ihtm1,w_ictm1,b_i,
                  w_fxt,w_fhtm1,w_fctm1,b_f,
                  w_cxt,w_chtm1,b_c,
                  w_oxt,w_ohtm1,w_oct,b_o,
                  w_sht,b_ht):
    inputGate_t = sigma(theano.dot(x_t,w_ixt) + theano.dot(h_tm1,w_ihtm1) + theano.dot(c_tm1,w_ictm1) + b_i)
    forgetGate_t = sigma(theano.dot(x_t,w_fxt) + theano.dot(h_tm1,w_fhtm1) + theano.dot(c_tm1,w_fctm1) + b_f)
    cellState_t = forgetGate_t * c_tm1 + inputGate_t * actF(theano.dot(x_t,w_cxt) + theano.dot(h_tm1,w_chtm1) + b_c)
    outGate_t = sigma(theano.dot(x_t,w_oxt) + theano.dot(h_tm1,w_ohtm1) + theano.dot(cellState_t,w_oct) + b_o)
    hidden_t = outGate_t * actF(cellState_t)
    outState_t = sigma(theano.dot(hidden_t,w_sht) + b_ht)

    #then normalise and sample:
    normalisedState = outState_t / T.sum(outState_t)
    probabilityDist = get_sample(normalisedState)
    castForOutput = T.cast(probabilityDist,dtype)
    
    return [hidden_t,cellState_t,castForOutput],theano.scan_module.until(T.eq(T.argmax(castForOutput),sentenceStop))


#create the start symbol:
#startValues = np.array([1.,0.,0.,0.,0.,0.,0.],dtype=dtype)
startValues = charToOneHot('t')
startState = theano.shared(startValues)

#create the sampling scan:
[sampleHiddenStates,sampleCellStates,sampleOutputStates], updates = theano.scan(fn=sampling_step,
                                                  sequences=[],
                                                  n_steps = sampleStepLength,
                                                  outputs_info=[h0,c0,startState],
                                                  non_sequences = [
                                                       input_xt1,input_htm1,input_ctm1,b_i,
                                                       forget_xt,forget_htm1,forget_ctm1,b_f,
                                                       cell_xt,cell_htm1,b_c,
                                                       out_xt,out_htm1,out_ct,b_o,
                                                       hidden_ht,b_out]
)



#create the main sampling function
sampleFunction = theano.function(inputs=[],outputs=sampleOutputStates, updates=updates)

def createSample():
    sampled = sampleFunction()
    sample_sequence = np.concatenate(([startValues], sampled), axis=0)
    words = sequenceToText(sample_sequence)
    logging.info("Generated String: %s" % (words))
    print("Generated: ",words)
    
#create the training routine:
def train_routine(train_data, nb_epochs=50):
    print("Starting Training routine")
    train_errors = np.ndarray(nb_epochs)
    for x in range(nb_epochs):
        error = 0.
        print('Epoch: ',x)
        logging.info("Epoch: %s" % (x))
        for j in range(len(train_data)):
            index = np.random.randint(0,len(train_data))
            dataInput, trueOutput = train_data[index]
            train_cost = trainFunction(dataInput,trueOutput)
            if((x-1) % 100 == 0 and j % 100 == 0):
                print("Epoch: ",x, " Trained: ",j," cost: ",train_cost)
                
            error = train_cost
        createSample()
        train_errors[x] = error
    return train_errors

train_errors = train_routine(numericTrainingData, nb_epochs)

#now print out the results of sampling:
for i in range(20):
    print("\n\n Generating:")
    sampled = sampleFunction()
    print(sampled.shape)
    sample_sequence = np.concatenate(([startValues], sampled), axis=0)
    words = sequenceToText(sample_sequence)
    print("Generated Words:",words)
