#!/usr/bin/python

import numpy as np

chars='BTSXPVE'

#graph represented both numerically and symbolically
graph = [[(1,5),('T','P')],
         [(1,2),('S','X')],
         [(3,5),('S','X')],
         [(6,),('E')],#terminal
         [(3,2),('V','P')],
         [(4,5),('V','T')]]


def in_grammar(word):
    if word[0] != 'B':
        return False
    node = 0
    for c in word[1:]:
        transitions = graph[node]
        try:
            #lookup the connection between
            node = transitions[0][transitions[1].index(c)]
        except ValueError:
            return False
    return True

#assuming sequence is a list of one hot vectors:
def sequenceToWord(sequence):
    reberString = ''
    for v in sequence:
        index = np.argmax(v)
        reberString += chars[index]
    return reberString

#generate a sequence of characters,
#and the sequence of options for each stage of that sequence
#so B -> T/P. Then BT -> S/X, then BTS -> S/X....
def generateSequences(minLength):
    while True:
        inchars = ['B']
        node = 0
        outchars = []
        while node != 6:
            transitions = graph[node]
            i = np.random.randint(0, len(transitions[0]))
            inchars.append(transitions[1][i])
            outchars.append(transitions[1])
            node = transitions[0][i]
        if len(inchars) > minLength:
            return inchars, outchars


#generate sequences, and then convert them to numeric arrays
def get_one_example(minLength):
    inchars, outchars = generateSequences(minLength)
    inseq = []
    outseq= []
    for i,o in zip(inchars, outchars):
        inpt = np.zeros(7)
        inpt[chars.find(i)] = 1.
        outpt = np.zeros(7)
        for oo in o:
            outpt[chars.find(oo)] = 1.
        inseq.append(inpt)
        outseq.append(outpt)
    return inseq, outseq


#convert a character to its numeric array representation
def get_char_one_hot(char):
    char_oh = np.zeros(7)
    for c in char:
        char_oh[chars.find(c)] = 1.
    return [char_oh]

#get a list of examples. each example a tuple of
#string at a point in sequence, options at that point in the sequence
def get_n_examples(n, minLength=10):
    examples = []
    for i in xrange(n):
        examples.append(get_one_example(minLength))
    return examples

emb_chars = "TP"

#create a long range dependency sequence,
#where the second character in the sequence is the same as the second
#to last character in sequence
def get_one_embedded_example(minLength=10):
    i, o = get_one_example(minLength)
    emb_char = emb_chars[np.random.randint(0, len(emb_chars))]
    new_in = get_char_one_hot(('B',))
    new_in += get_char_one_hot((emb_char,))
    new_out= get_char_one_hot(emb_chars)
    new_out+= get_char_one_hot('B',)
    new_in += i
    new_out += o
    new_in += get_char_one_hot(('E',))
    new_in += get_char_one_hot((emb_char,))
    new_out += get_char_one_hot((emb_char, ))
    new_out += get_char_one_hot(('E',))
    return new_in, new_out

#list of (actualSequence,possibilitiesAtEachStageofSequence)
def get_n_embedded_examples(n, minLength=10):
    examples = []
    for i in xrange(n):
        examples.append(get_one_embedded_example(minLength))
    return examples
    
