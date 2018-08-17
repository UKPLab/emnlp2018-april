from sklearn.metrics import mean_squared_error
import scipy.stats as stats
from summariser.utils.rank_metrics import ndcg_at_k
from summariser.utils.misc import *
from summariser.rouge.rouge import Rouge
from resources import *

def evaluateReward(learnt_values, ref_values):
    metrics_dic = {}

    ### compute the absolute errors
    mse = mean_squared_error(ref_values,learnt_values)
    metrics_dic['mse'] = mse

    ### compute KL divergence
    prob_optimal = getSoftmaxList(ref_values, 1.0)
    prob_learnt = getSoftmaxList(learnt_values, 1.0)
    kld = stats.entropy(prob_optimal, prob_learnt)
    metrics_dic['kld'] = kld

    ### compute Kendall's tau, Spearman's rho and Pearson correlation coefficient
    sorted_list = sorted(learnt_values)
    new_reward_ranking = [sorted_list.index(i) for i in learnt_values]
    sorted_list = sorted(ref_values)
    true_reward_ranking = [sorted_list.index(i) for i in ref_values]
    tau, _ = stats.kendalltau(new_reward_ranking, true_reward_ranking)
    rho, _ = stats.pearsonr(new_reward_ranking, true_reward_ranking)
    pcc, _ = stats.pearsonr(learnt_values, ref_values)
    metrics_dic['tau'] = tau
    metrics_dic['rho'] = rho
    metrics_dic['pcc'] = pcc

    ### compute nDCG
    sorted_list = sorted(learnt_values,reverse=True)
    ll = [ref_values[learnt_values.index(ele)] for ele in sorted_list]

    ndcg = ndcg_at_k(ll,int(0.01*len(ll)))
    metrics_dic['ndcg_at_1%'] = ndcg
    ndcg = ndcg_at_k(ll,int(0.05*len(ll)))
    metrics_dic['ndcg_at_5%'] = ndcg
    ndcg = ndcg_at_k(ll,int(0.1*len(ll)))
    metrics_dic['ndcg_at_10%'] = ndcg
    ndcg = ndcg_at_k(ll,int(0.2*len(ll)))
    metrics_dic['ndcg_at_20%'] = ndcg
    ndcg = ndcg_at_k(ll,int(0.5*len(ll)))
    metrics_dic['ndcg_at_50%'] = ndcg
    ndcg = ndcg_at_k(ll,len(ll))
    metrics_dic['ndcg_at_all'] = ndcg

    return metrics_dic


def evaluateSummary(cand,model,len=100):
    rouge_scorer = Rouge(ROUGE_DIR,BASE_DIR,True)
    r1, r2, rl, rsu4 = rouge_scorer(cand,[model],len)
    rouge_scorer.clean()
    dic = {}
    dic['ROUGE-1'] = r1
    dic['ROUGE-2'] = r2
    dic['ROUGE-L'] = rl
    dic['ROUGE-SU4'] = rsu4
    dic['WeightedSum'] = 3.33*(r1/.47 + r2/.212 + rsu4/.185)
    return dic

