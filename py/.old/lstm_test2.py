import theano, theano.tensor as T
import numpy as np
import theano_lstm
import random

#Take a probability table of the form: {"a":0.5,"b":0.5}
#get a sample using sampler()

class Sampler:
    def __init__(self, prob_table):
        total_prob = 0.0
        #if passed in a dictionary of probabilities
        if type(prob_table) is dict:
            for key, value in prob_table.items():
                total_prob += value
        #if passed in a list, convert it
        elif type(prob_table) is list:
            prob_table_gen = {}
            for key in prob_table:
                prob_table_gen[key] = 1.0 / (float(len(prob_table)))
                total_prob = 1.0
                prob_table = prob_table_gen
        #else complain:
        else:
            raise ArgumentError("__init__ takes either a dict or a list as its first argument")
        if total_prob <= 0.0:
            raise ValueError("Probability is not strictly positive.")
        
        #additional setup:
        self._keys = []
        self._probs = []
        for key in prob_table:
            self._keys.append(key)
            self._probs.append(prob_table[key] / total_prob)

    #when called, return a random word
    def __call__(self):
        sample = random.random()
        seen_prob = 0.0
        for key, prob in zip(self._keys, self._probs):
            if (seen_prob + prob) >= sample:
                return key
            else:
                seen_prob += prob
        return key
