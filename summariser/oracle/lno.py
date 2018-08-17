from summariser.rouge.rouge import Rouge
from resources import BASE_DIR, ROUGE_DIR
import random
import math

class SimulatedUser:
    def __init__(self,model,len,temp=2.5):
        self.rouge_score = Rouge(ROUGE_DIR,BASE_DIR,False)
        self.temperature = temp
        self.model = model
        self.summary_len = len

    def getPref(self,sum1,sum2):
        R11, R12, R1SU = self.rouge_score(sum1, self.model, self.summary_len)
        R21, R22, R2SU = self.rouge_score(sum2, self.model, self.summary_len)
        self.rouge_score.clean()
        score1 = R11 / 0.47 + R12 / 0.212 + R1SU / 0.185
        score2 = R21 / 0.47 + R22 / 0.212 + R2SU / 0.185

        if random.random() > 1./(1.+math.exp((score2-score1)/self.temperature)):
            # prefers summary 1
            return 0
        else:
            # prefers summary 2
            return 1



