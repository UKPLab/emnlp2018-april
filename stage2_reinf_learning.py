from resources import PROCESSED_PATH,LEARNT_RANKER_PATH
from summariser.utils.reader import readSampleSummaries
from summariser.vector.vector_generator import Vectoriser
from summariser.rl.td_agent import *
from summariser.utils.evaluator import evaluateSummary

import os

def addResult(all_dic,result):
    for metric in result:
        if metric in all_dic:
            all_dic[metric].append(result[metric])
        else:
            all_dic[metric] = [result[metric]]

def readLearntRank(dataset,topic,model_name,setting,vectors):
    weights = []
    path = os.path.join(LEARNT_RANKER_PATH,dataset,topic,'model{}-{}'.format(model_name,setting))
    ff = open(path,'r')
    for line in ff.readlines():
        weights.append(float(line))
    ff.close()

    rewards = []
    for vec in vectors:
        rewards.append(np.dot(vec,weights))
    return rewards


if __name__ == '__main__':

    #parameters
    rl_episode = 5000
    strict = 5.0
    dataset = 'DUC2001'
    reward_type = 'heuristic' # rouge, heuristic, learnt
    base_length = 200
    block_num = 1
    learnt_rank_setting = None # for example: 'round100-queryRANDOM-postWeight0.5'

    # read documents and ref. summaries
    reader = CorpusReader(PROCESSED_PATH)
    data = reader.get_data(dataset)

    topic_cnt = 0
    all_result_dic = {}
    for topic,docs,models in data:

        topic_cnt += 1
        print('\n---TOPIC: {}---'.format(topic))

        summaries, ref_values_dic, heuristic_values_list = readSampleSummaries(dataset,topic)
        vec = Vectoriser(docs,base=base_length,block=block_num)
        rl_agent = TDAgent(vec,summaries,rl_episode,strict)

        if reward_type == 'rouge' or reward_type == 'learnt':
            for model in models:
                model_name = model[0].split('/')[-1].strip()
                print('\n---ref {}---'.format(model_name))
                if reward_type == 'rouge':
                    rewards = ref_values_dic[model_name]
                else:
                    summary_vectors = vec.getSummaryVectors(summaries)
                    assert learnt_rank_setting is not None
                    rewards = readLearntRank(dataset,topic,model_name,learnt_rank_setting,summary_vectors)
                summary = rl_agent(rewards)
                result = evaluateSummary(summary,model)
                addResult(all_result_dic,result)
                for metric in result:
                    print('{} : {}'.format(metric,result[metric]))
        elif reward_type == 'heuristic':
            summary = rl_agent(heuristic_values_list)
            for model in models:
                result = evaluateSummary(summary,model)
                addResult(all_result_dic,result)
        else:
            pass #TODO, read learnt rewards to do RL

        print('\n=== RESULTS UNTIL TOPIC {}, EPISODE {} ===\n'.format(topic_cnt, rl_episode))
        for metric in all_result_dic:
            print('{} : {}'.format(metric, np.mean(all_result_dic[metric])))


