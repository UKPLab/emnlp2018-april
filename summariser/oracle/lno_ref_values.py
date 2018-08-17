import math
import random

from summariser.utils.misc import sigmoid

class SimulatedUser:
    def __init__(self,ref_values,m=2.5):
        self.ref_values = ref_values
        self.temperature = m

    def getPref(self,idx1,idx2):
        prob = sigmoid(self.ref_values[idx1]-self.ref_values[idx2], self.temperature)

        if random.random() <= prob:
            return 0 ## summary1 is preferred
        else:
            return 1 ## summary2 is preferred

