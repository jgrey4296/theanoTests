import theano, theano.tensor as T
import numpy as np
import theano_lstm
import random

class Sampler:
    def __init__(self, prob_table):
        total_prob = 0.0
        if type(prob_table) is dict:
            for key, value in prob_table.items():
                total_prob += value
        elif type(prob_table) is list:
            prob_table_gen = {}
            for key in prob_table:
                prob_table_gen[key] = 1.0 / (float(len(prob_table)))
                total_prob = 1.0
                prob_table = prob_table_gen
        else:
            raise ArgumentError("__init__ takes either a dict or a list as its first argument")
        if total_prob <= 0.0:
            raise ValueError("Probability is not strictly positive.")
        self._keys = []
        self._probs = []
        for key in prob_table:
            self._keys.append(key)
            self._probs.append(prob_table[key] / total_prob)
            
    def __call__(self):
        sample = random.random()
        seen_prob = 0.0
        for key, prob in zip(self._keys, self._probs):
            if (seen_prob + prob) >= sample:
                return key
            else:
                seen_prob += prob
        return key
