import random
from summariser.querier.logistic_reward_learner import *
from summariser.utils.misc import normaliseList
from summariser.utils.misc import sigmoid

class UncQuerier:

    def __init__(self,summary_vectors,heuristic_values,learnt_weight=0.5):
        self.summary_vectors = summary_vectors
        self.reward_learner = LogisticRewardLearner()
        self.learnt_values = [0.]*len(summary_vectors)
        self.heuristics = heuristic_values
        self.learnt_weight = learnt_weight

    def getUncScores(self,scores):
        unc_scores = []

        for vv in scores:
            prob = sigmoid((vv-5)*.6)
            if prob > 0.5:
                unc_scores.append(2*(1-prob))
            else:
                unc_scores.append(2*prob)

        return unc_scores

    def inLog(self,sum1,sum2,log):
        for entry in log:
            if [sum1,sum2] in entry:
                return True
            elif [sum2,sum1] in entry:
                return True

        return False

    def getMostUncertainPair(self,num,unc_scores,sorted_unc,log):
        max_value = -999
        pair = [-1,-1]
        if num > len(unc_scores):
            num = len(unc_scores)

        for i in range(num-1):
            for j in range(i+1,num):
                if sorted_unc[i]+sorted_unc[j] > max_value:
                    idx_i = unc_scores.index(sorted_unc[i])
                    idx_j = unc_scores.index(sorted_unc[j])
                    if not self.inLog(idx_i,idx_j,log):
                        pair = [idx_i,idx_j]
                        max_value = sorted_unc[i]+sorted_unc[j]

        return pair


    def getQuery(self,log):
        mix_values = self.getMixReward()
        unc_scores = self.getUncScores(mix_values)
        sorted_unc = sorted(unc_scores,reverse=True)
        num = 10

        pair = self.getMostUncertainPair(num,unc_scores,sorted_unc,log)
        while pair == [-1,-1]:
            pair = self.getMostUncertainPair(2*num,unc_scores,sorted_unc)


        if random.random() > 0.5:
            return pair[0], pair[1]
        else:
            return pair[1], pair[0]

    def updateRanker(self,pref_log):
        self.reward_learner.train(pref_log,self.summary_vectors)
        self.learnt_values = self.getReward()

    def getReward(self):
        values = [np.dot(self.reward_learner.weights,vv) for vv in self.summary_vectors]
        return normaliseList(values)

    def getMixReward(self,learnt_weight=-1):
        if learnt_weight == -1:
            learnt_weight = self.learnt_weight

        mix_values = np.array(self.learnt_values)*learnt_weight+np.array(self.heuristics)*(1-learnt_weight)

        return normaliseList(mix_values)



