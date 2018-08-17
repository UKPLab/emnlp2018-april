import numpy as np
import random
import math

def normaliseList(ll,max_value=10.):
    minv = min(ll)
    maxv = max(ll)

    new_ll = [(x-minv)*max_value/maxv for x in ll]

    return new_ll

def sigmoid(x,temp=1.):
    return 1.0/(1.+math.exp(-x/temp))


def softmaxSample(value_list,strict,softmax_list=[]):
    if len(softmax_list) == 0:
        softmax_list = getSoftmaxList(value_list,strict)

    pointer = random.random()*sum(softmax_list)
    tier = 0
    idx = 0

    for value in softmax_list:
        if pointer >= tier and pointer < tier+value:
            return idx
        else:
            tier += value
            idx += 1
    return -1


def getSoftmaxList(value_list, strict):
    softmax_list = []
    for value in value_list:
        softmax_list.append(np.exp(value/strict))
    return softmax_list



