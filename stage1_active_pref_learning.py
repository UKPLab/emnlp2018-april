from summariser.oracle.lno_ref_values import SimulatedUser
from summariser.utils.corpus_reader import CorpusReader
from resources import PROCESSED_PATH, LEARNT_RANKER_PATH
from summariser.utils.reader import readSampleSummaries
from summariser.vector.vector_generator import Vectoriser
from summariser.utils.evaluator import evaluateReward
from summariser.querier.random_querier import RandomQuerier
from summariser.querier.gibbs_querier import GibbsQuerier
from summariser.querier.uncertainty_querier import UncQuerier

import numpy as np
import os


def writeRanker(weights,dataset,topic,model,inter_round,q_type,post_w):
    if not os.path.isdir(os.path.join(LEARNT_RANKER_PATH,dataset)):
        os.makedirs(os.path.join(LEARNT_RANKER_PATH,dataset))

    if not os.path.isdir(os.path.join(LEARNT_RANKER_PATH,dataset,topic)):
        os.makedirs(os.path.join(LEARNT_RANKER_PATH,dataset,topic))

    out_file = os.path.join(LEARNT_RANKER_PATH,dataset,topic,
                            'model{}-round{}-query{}-postWeight{}'.format(model,inter_round,q_type.upper(),post_w))

    str = ''
    for ww in weights:
        str += '{}\n'.format(ww)

    ff = open(out_file,'w')
    ff.write(str)
    ff.close()

if __name__ == '__main__':

    ### parameters
    inter_round = 100
    dataset = 'DUC2001' # DUC2001, DUC2002, DUC2004
    querier_type = 'random' # gibbs, random, unc
    posterior_weight = 0.5 # trade off between the heuristic rewards and the pref-learnt rewards
    write_learnt_reward = True

    ### read documents and ref. summaries
    reader = CorpusReader(PROCESSED_PATH)
    data = reader.get_data(dataset)

    ### store all results
    all_result_dic = {}
    topic_cnt = 0

    for topic,docs,models in data:
        print('\n=====TOPIC {}, QUERIER {}, INTER ROUND {}====='.format(topic,querier_type.upper(),inter_round))
        topic_cnt += 1
        summaries, ref_values_dic, heuristic_values_list = readSampleSummaries(dataset,topic)
        print('num of summaries read: {}'.format(len(summaries)))
        vec = Vectoriser(docs)
        summary_vectors = vec.getSummaryVectors(summaries)

        for model in models:
            model_name = model[0].split('/')[-1].strip()
            print('\n---ref. summary {}---'.format(model_name))
            rouge_values = ref_values_dic[model_name]

            oracle = SimulatedUser(rouge_values)
            if querier_type == 'gibbs':
                querier = GibbsQuerier(summary_vectors,heuristic_values_list,posterior_weight)
            elif querier_type == 'unc':
                querier = UncQuerier(summary_vectors,heuristic_values_list,posterior_weight)
            else:
                querier = RandomQuerier(summary_vectors,heuristic_values_list,posterior_weight)

            log = []

            for round in range(inter_round):
                sum1, sum2 = querier.getQuery(log)
                pref = oracle.getPref(sum1,sum2)
                log.append([[sum1,sum2],pref])
                querier.updateRanker(log)

            metrics_dic = evaluateReward(querier.getMixReward(), rouge_values)

            for metric in metrics_dic:
                print('metric {} : {}'.format(metric,metrics_dic[metric]))
                if metric in all_result_dic:
                    all_result_dic[metric].append(metrics_dic[metric])
                else:
                    all_result_dic[metric] = [metrics_dic[metric]]

            if write_learnt_reward:
                writeRanker(querier.reward_learner.weights, dataset, topic, model_name, inter_round, querier_type, posterior_weight)

        print('\n=== RESULTS UNTIL TOPIC {}, QUERIER {}, INTER ROUND {} ===\n'.format(
            topic_cnt, querier_type.upper(), inter_round
        ))
        for metric in all_result_dic:
            print('{} : {}'.format(metric, np.mean(all_result_dic[metric])))






