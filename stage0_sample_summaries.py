from summariser.vector.vector_generator import Vectoriser
from summariser.utils.corpus_reader import CorpusReader
from resources import *
from summariser.utils.writer import append_to_file


def writeSample(actions,reward,path):
    if 'heuristic' in path:
        str = '\nactions:'
        for act in actions:
            str += repr(act)+','
        str = str[:-1]
        str += '\nutility:'+repr(reward)
        append_to_file(str, path)
    else:
        assert 'rouge' in path
        str = '\n'
        for j,model_name in enumerate(reward):
            str += '\nmodel {}:{}'.format(j,model_name)
            str += '\nactions:'
            for act in actions:
                str += repr(act)+','
            str = str[:-1]
            str += '\nR1:{};R2:{};R3:{};R4:{};RL:{};RSU:{}'.format(
                reward[model_name][0],reward[model_name][1],reward[model_name][2],
                reward[model_name][3],reward[model_name][4],reward[model_name][5]
            )
        append_to_file(str, path)


if __name__ == '__main__':

    dataset = 'DUC2004'
    language = 'english'
    summary_len = 100

    summary_num = 10100
    base_dir = os.path.join(SUMMARY_DB_DIR,dataset)

    reader = CorpusReader(PROCESSED_PATH)
    data = reader.get_data(dataset,summary_len)

    topic_cnt = 0
    start = 50
    end = 100
    print('dataset: {}; start: {}; end: {}'.format(dataset,start,end-1))

    for topic, docs, models in data:
        topic_cnt += 1
        dir_path = os.path.join(base_dir,topic)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        vec = Vectoriser(docs,summary_len)
        if topic_cnt >= start and topic_cnt < end:
            print('-----Generate samples for topic {}: {}-----'.format(topic_cnt,topic))
            act_list, h_rewards, r_rewards = vec.sampleRandomReviews(summary_num,True,True,models)

            assert len(act_list) == len(h_rewards) == len(r_rewards)

            for ii in range(len(act_list)):
                writeSample(act_list[ii],h_rewards[ii],os.path.join(dir_path,'heuristic'))
                writeSample(act_list[ii],r_rewards[ii],os.path.join(dir_path,'rouge'))

    print('dataset {}; total topic num: {}; start: {}; end: {}'.format(dataset,topic_cnt,start,end-1))













