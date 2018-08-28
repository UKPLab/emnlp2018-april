from resources import PROCESSED_PATH
from summariser.utils.reader import readSampleSummaries
from summariser.vector.vector_generator import Vectoriser
from summariser.rl.td_agent import *
from summariser.utils.evaluator import evaluateSummary


def addResult(all_dic,result):
    for metric in result:
        if metric in all_dic:
            all_dic[metric].append(result[metric])
        else:
            all_dic[metric] = [result[metric]]

if __name__ == '__main__':

    #parameters
    rl_episode = 5000
    strict = 5.0
    dataset = 'DUC2001'
    reward_type = 'rouge' # rouge, heuristic, learnt
    base_length = 300
    block_num = 1

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
        #summary_vectors = vec.getSummaryVectors(summaries)
        rl_agent = TDAgent(vec,summaries,rl_episode,strict)

        result_dic = {}
        if reward_type == 'rouge':
            for model in models:
                model_name = model[0].split('/')[-1].strip()
                print('\n---ref {}---'.format(model_name))
                summary = rl_agent(ref_values_dic[model_name])
                result = evaluateSummary(summary,model)
                addResult(result_dic,result)
                for metric in result:
                    print('{} : {}'.format(metric,result[metric]))
        elif reward_type == 'heuristic':
            summary = rl_agent(heuristic_values_list)
            for model in models:
                result = evaluateSummary(summary,model)
                addResult(result_dic,result)
        else:
            pass #TODO, read learnt rewards to do RL

        addResult(all_result_dic,result_dic)
        print('\n=== RESULTS UNTIL TOPIC {}, EPISODE {} ===\n'.format(topic_cnt, rl_episode))
        for metric in all_result_dic:
            print('{} : {}'.format(metric, np.mean(all_result_dic[metric])))


