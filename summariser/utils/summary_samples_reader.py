import os
import operator as op
import numpy as np
import pickle

def getExistingNum(path):
    cnt = 0
    ff = open(path,'r')
    for line in ff.readlines():
        if 'rewards:' in line:
            cnt += 1
    ff.close()
    return cnt

def normaliseList(ll,maxv=10.0):
    minv = min(ll)
    rangev = max(ll) - minv
    if rangev == 0:
        return [0.]*len(ll)

    norm_list = []
    for vv in ll:
        norm_list.append(maxv*(vv-minv)/rangev)

    return norm_list

def normaliseMatrix(mm, maxv=10.0):
    minv = mm.min()
    rangev = mm.max()-minv
    rtn_matrix = []
    for row in mm:
        new_row = []
        for ele in row:
            new_row.append(maxv*(ele-minv)/rangev)
        rtn_matrix.append(new_row)

    return np.array(rtn_matrix)


def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def ncr(n, r):
    r = min(r, n - r)
    if r == 0: return 1
    numer = reduce(op.mul, xrange(n, n - r, -1))
    denom = reduce(op.mul, xrange(1, r + 1))
    return numer // denom


def getDBSRewardsFromWeights(summary_list, weights_dir, vector_path):
    # get vectors
    _, vector_list, _ = readSyntheticSampleVectors(summary_list,vector_path,False,True,False)

    # get weights
    weights = []
    f = open(weights_dir,'r')
    for line in f.readlines():
        weights.append(float(line))
    weights = np.array(weights)
    f.close()

    # compoute dot product
    uti_list = []
    for vector in vector_list:
        uti_list.append(np.dot(vector,weights))
    norm_uti_list = normaliseList(uti_list)

    return norm_uti_list



def getRewardsFromWeights(summary_list,weights_dir,vector_path,sample_num,model_num):
    # get vectors
    _, vector_list, _ = readSyntheticSampleVectors(summary_list,vector_path,False,True,False)

    # get weights
    value_list = []
    for model_idx in range(model_num):
        weights_path = os.path.join(weights_dir,'model{}'.format(model_idx))
        weights = []
        f = open(weights_path,'r')
        for line in f.readlines():
            weights.append(float(line))
        weights = np.array(weights)
        f.close()
        for i, vec in enumerate(vector_list):
            if model_idx == 0:
                value = [np.dot(weights,np.array(vec))]
                value_list.append(value)
            else:
                value = np.dot(weights,vec)
                value_list[i].append(value)

    # normalise the values
    minvs = []
    ranges = []
    norm_value_list = []
    for idx in range(model_num):
        maxv = max([i[idx] for i in value_list])
        minv = min([i[idx] for i in value_list])
        ranges.append(maxv-minv)
        minvs.append(minv)
    for cnt in range(len(value_list)):
        temp_row = []
        for idx in range(model_num):
            temp_row.append( 10.0*(value_list[cnt][idx]-minvs[idx])/ranges[idx] )
        norm_value_list.append(temp_row)


    return norm_value_list


def readAccPretrainedRewards(acts_list,topic,data_set):
    print('\n-----Start reading pre-trained acc-rewards. Target num: {}-----'.format(len(acts_list)))
    # read in accumulated rewards
    reward_path = os.path.join('/home/gao/PycharmProjects/sampling_summaries/pretrained_rewards',data_set,topic)
    cnt = 0
    raw_reward_list = []
    ff = open(reward_path,'r')
    for line in ff.readlines():
        if 'actions:' in line:
            acts = [int(i) for i in line.split(':')[1].split(',')]
            assert acts == acts_list[cnt]
            cnt += 1
        elif 'acc-rewards:' in line:
            raw_reward_list.append(float(line.split(':')[1]))
            if cnt >= len(acts_list):
                break
        else:
            print('!!!!!ERROR in reading pretrained rewards! Count:{}\n Path: {}'.format(cnt,reward_path))
    ff.close()

    # normalise
    norm_list = normaliseList(raw_reward_list)

    print('{} pretrained acc-rewards have been read and normalised'.format(len(norm_list)))
    return norm_list


def readSumVecPretrainedRewards(summary_list,topic,data_set):
    # read vectors, vector form: NGRAM-base100-block6
    sample_num = len(summary_list)
    print('\n-----Start reading pre-trained rewards. Target num: {}-----'.format(sample_num))
    vector_strs_dir = os.path.join('/home/gao/workspace/v43-12Sep/summarizer/sampled-summaries-vectors-strs',data_set,topic)
    vector_file_path = os.path.join(vector_strs_dir,'optimal','state{}-base{}-block{}'.format('NGRAM',100,6))
    _, vector_list, _ = \
        readSyntheticSampleVectors(summary_list,vector_file_path,False,True,False)

    # read weights
    weights = []
    weights_path = os.path.join('/home/gao/workspace/v43-12Sep/summarizer/pretrained_rewards/readin_weights',topic)
    file = open(weights_path,'r')
    for line in file.readlines():
        weights.append(float(line))
    weights = np.array(weights)
    file.close()

    # compute the rewards for each summary
    reward_list = []
    for idx in range(sample_num):
        temp_value = np.dot(np.array(vector_list[idx]),weights)
        reward_list.append(temp_value)

    # normalise the rewards
    norm_reward_list = normaliseList(reward_list)

    print('{} pre-trained rewards are read and normalised.'.format(len(norm_reward_list)))
    return norm_reward_list

def reconstructFullVectorFromShort(length, str):
    ele_list = str.split(';')
    ele_idx = 0
    vector = []
    for ii in range(length):
        if ii == int(ele_list[ele_idx].split(":")[0]):
            vector.append(float(ele_list[ele_idx].split(":")[1]))
            ele_idx+=1
        else:
            vector.append(0.0)

    return np.array(vector)

def readIntermediateSampleVectors(summary_act_list,path,vec_length,inter_vectors_wanted=True,final_vector_wanted=True):
    print('\n-----Start reading vectors from: {}-----'.format(path))
    summary_list = []
    inter_vector_list = []
    final_vector_list = []
    file = open(path,'r')
    num = len(summary_act_list)
    temp_vectors_list = []
    act_num = -1

    cnt = 0
    for line in file.readlines():
        if cnt >= num:
            break
        if 'AYGactionsAYG:' in line:
            acts = [int(i) for i in line.split(':')[1].split(',')]
            if acts == summary_act_list[cnt]:
                act_num = len(acts)
                continue
            else:
                print('ERROR! Inconsistent actions!')
                print('From sampled summary {}: {}'.format(cnt,summary_act_list[cnt]))
                print('From vector strs: {}'.format(line))
                exit(1)
        elif 'AYGacc_vector:' in line:
            if inter_vectors_wanted:
                vec = line.split(':')[1]
                temp_vectors_list.append(vec)
            if final_vector_wanted and '{}'.format(act_num) in line:
                vec = line.split(':')[1]
                final_vector_list.append(vec)
            if '{}'.format(act_num-1) in line:
                inter_vector_list.append(temp_vectors_list[:])
                cnt += 1
                act_num = -1
                temp_vectors_list = []

    file.close()
    print('{} samples are read from file'.format(cnt))
    return summary_list, inter_vector_list, final_vector_list



def readShortAccSampleVectors(summary_act_list, path, vec_length,
                              str_wanted=True, acc_vector_wanted=True, final_vector_wanted=True, reward_wanted=True):
    print('\n-----Start reading vectors from: {}-----'.format(path))
    summary_list = []
    acc_vector_list = []
    final_vector_list = []
    reward_list = []
    file = open(path,'r')
    num = len(summary_act_list)

    cnt = 0
    for line in file.readlines():
        if cnt >= num:
            break
        if 'AYGactionsAYG:' in line:
            acts = [int(i) for i in line.split(':')[1].split(',')]
            if acts == summary_act_list[cnt]:
                continue
            else:
                print('ERROR! Inconsistent actions!')
                print('From sampled summary {}: {}'.format(cnt,summary_act_list[cnt]))
                print('From vector strs: {}'.format(line))
                exit(1)
        elif 'AYGacc_vectorAYG:' in line:
            if acc_vector_wanted:
                vec = reconstructFullVectorFromShort(vec_length,line[len("AYGacc_vectorAYG:"):])
                acc_vector_list.append(vec)
        elif 'AYGfinal_vectorAYG:' in line:
            if final_vector_wanted:
                vec = reconstructFullVectorFromShort(vec_length,line[len("AYGfinal_vectorAYG:"):])
                final_vector_list.append(vec)
        elif 'AYGsummary stringAYG:' in line:
            if str_wanted:
                str = line[15:-1]
                summary_list.append(str)
            cnt += 1

    file.close()
    print('{} samples are read from file'.format(cnt))
    return summary_list,acc_vector_list,final_vector_list,reward_list




def readSyntheticSampleVectors(summary_act_list, path, str_wanted=True, vector_wanted=True,reward_wanted=True):
    print('\n-----Start reading vectors from: {}-----'.format(path))
    summary_list = []
    vector_list = []
    reward_list = []
    file = open(path,'r')
    num = len(summary_act_list)

    cnt = 0
    for line in file.readlines():
        if cnt >= num:
            break
        if 'AYGactionsAYG:' in line:
            acts = [int(i) for i in line.split(':')[1].split(',')]
            if acts == summary_act_list[cnt]:
                continue
            else:
                print('ERROR! Inconsistent actions!')
                print('From sampled summary {}: {}'.format(cnt,summary_act_list[cnt]))
                print('From vector strs: {}'.format(line))
                exit(1)
        elif 'AYGvectorAYG:' in line:
            if vector_wanted:
                vec = [float(i) for i in line.split(':')[1].split(',')]
                vector_list.append(vec)
        elif 'AYGsummary stringAYG:' in line:
            if str_wanted:
                str = line[15:-1]
                summary_list.append(str)
            cnt += 1

    file.close()
    print('{} samples are read from file'.format(cnt))
    return summary_list,vector_list,reward_list


def mixingPretrainedLearntRewards(prewards, lrewards, tdf):
    new_list = []
    assert len(prewards) == len(lrewards)
    for idx in range(len(lrewards)):
        temp_row =  tdf*prewards[idx]+(1-tdf)*np.array(lrewards[idx])
        new_list.append(temp_row.tolist())
    return new_list

def mixingPretrainedLearntModelRewards(prewards, lrewards_list, tdf):
    new_list = []
    for lrewards in lrewards_list:
        model_value = mixingPretrainedLearntRewards(prewards,lrewards,tdf)
        new_list.append(model_value)

    return new_list


def readDbsOptimalSummariesAndValues(sample_num,dir,topic,only_act=False):
    print('\n-----Start reading samples in topic {}. Target sample num: {}-----'.format(topic, sample_num))

    summary_list = []
    summary_value_list = []
    type = 'optimal'
    model_idx_str = topic[topic.index('_')+1]
    with open(os.path.join(dir,topic,type),'r') as file:
        flag = False
        cnt = 0
        value = -1
        while cnt < sample_num:
            line = file.readline()
            if line == '':
                break
            if 'model '+model_idx_str not in line and not flag:
                continue
            elif 'model '+model_idx_str in line:
                flag = True
            elif 'model '+model_idx_str not in line and flag:
                if 'action' in line:
                    nums_str = line.split(':')[1].split(',')
                    acts = []
                    for nn in nums_str:
                        acts.append(int(nn))
                    summary_list.append(acts)
                    if only_act:
                        cnt += 1
                elif 'R1' in line:
                    if only_act:
                        continue
                    scores = line.split(';')
                    R1 = float(scores[0].split(':')[1])
                    R2 = float(scores[1].split(':')[1])
                    #R3 = float(scores[2].split(':')[1])
                    #R4 = float(scores[3].split(':')[1])
                    #RL = float(scores[4].split(':')[1])
                    RSU = float(scores[5].split(':')[1])
                    value = R1/0.848+R2/0.75+RSU/0.532
                elif 'action' not in line and 'R1' not in line:
                    flag = False
                    if not only_act:
                        assert value != -1
                        summary_value_list.append(value)
                        cnt += 1
                    value = -1

    if only_act:
        return summary_list

    #normalise
    norm_value_list = normaliseList(summary_value_list)

    print('{} samples are read from file'.format(len(norm_value_list)))
    return summary_list, norm_value_list

def getModelOrder(dir,topic):
    model_names = []
    ff = open(os.path.join(dir,topic,'optimal'))
    for line in ff.readlines():
        if 'model ' in line:
            model = line.split(":")[1]
            if model in model_names:
                break
            else:
                model_names.append(model)

    ff.close()
    return model_names

def readSummariesAndValues(sample_num,dir,topic,reward_type,only_act=False):
    print('\n-----Start reading samples in topic {}. Target sample num: {}-----'.format(topic, sample_num))
    summary_list = []
    summary_value_list = []
    type = reward_type
    with open(os.path.join(dir,topic,type),'r') as file:
        if reward_type == 'synthetic':
            cnt = 0
            while cnt < sample_num:
                line = file.readline()
                if line == '':
                    break
                if 'actions' not in line:
                    continue
                else:
                    cnt += 1
                    nums_str = line.split(':')[1].split(',')
                    acts = []
                    for nn in nums_str:
                        acts.append(int(nn))
                    summary_list.append(acts)
                    if not only_act:
                        value = float(file.readline().split(':')[1])
                        summary_value_list.append(value)
        elif reward_type == 'optimal':
            flag = False
            value_list = []
            idx = -1
            cnt = 0
            while cnt < sample_num:
                line = file.readline()
                if line == '':
                    break
                if 'model' not in line and not flag:
                    continue
                elif 'model' in line:
                    idx = int(line.split(':')[0].split(' ')[1])
                    flag = True
                elif 'model' not in line and flag:
                    if 'action' in line and idx == 0:
                        nums_str = line.split(':')[1].split(',')
                        acts = []
                        for nn in nums_str:
                            acts.append(int(nn))
                        summary_list.append(acts)
                        if only_act:
                            cnt += 1
                    elif 'R1' in line:
                        if only_act:
                            continue
                        scores = line.split(';')
                        R1 = float(scores[0].split(':')[1])
                        R2 = float(scores[1].split(':')[1])
                        #R3 = float(scores[2].split(':')[1])
                        #R4 = float(scores[3].split(':')[1])
                        #RL = float(scores[4].split(':')[1])
                        RSU = float(scores[5].split(':')[1])
                        if 'DUC' in dir:
                            value = 2*(R1/0.48+R2/0.212+RSU/0.195)
                        elif 'DBS' in dir:
                            value = R1/0.848+R2/0.75+RSU/0.532
                        else:
                            print('Error in readSummariesAndValues! DUC or DBS are not in the directory {}'.format(dir))
                            exit(1)
                        value_list.append(value)
                    elif 'action' not in line and 'R1' not in line:
                        flag = False
                        if not only_act:
                            summary_value_list.append(value_list[:])
                            cnt += 1
                        value_list = []

    if only_act:
        return summary_list

    #normalise
    if reward_type == 'synthetic':
        norm_value_list = normaliseList(summary_value_list)
    elif reward_type == 'optimal':
        model_num = len(summary_value_list[0])
        minvs = []
        ranges = []
        norm_value_list = []
        for idx in range(model_num):
            maxv = max([i[idx] for i in summary_value_list])
            minv = min([i[idx] for i in summary_value_list])
            ranges.append(maxv-minv)
            minvs.append(minv)
        for cnt in range(len(summary_value_list)):
            temp_row = []
            for idx in range(model_num):
                temp_row.append( 10.0*(summary_value_list[cnt][idx]-minvs[idx])/ranges[idx] )
            norm_value_list.append(temp_row)


    print('{} samples are read from file'.format(len(norm_value_list)))
    return summary_list, norm_value_list


def getValueListFromGpplModel(summary_list,vector_path,model_dir,setting,model_name):
    tt = setting.split('_')
    tt.insert(2,'model{}'.format(model_name))
    file_name = '_'.join(tt)

    if file_name not in os.listdir(model_dir):
        print('ERROR! No file in dir {} corresponds to model {}'.format(model_dir,model_name))
        exit(1)

    with open(os.path.join(model_dir,file_name),'r') as fh:
        gppl_model = pickle.load(fh)

    # get vectors
    _, vector_list, _ = readSyntheticSampleVectors(summary_list,vector_path,False,True,False)

    vector_array = np.array(vector_list)
    rewards_list, var_list = gppl_model.predict_f(None,vector_array)
    rewards_list = np.ndarray.tolist(rewards_list.reshape(1,len(rewards_list))[0])

    return normaliseList(rewards_list)


def getGpplLearntRewards(summary_list,data_set,topic,learnt_rewards_setting_name,noisy,model_names):
    if 'DUC' in data_set:
        base_length = 200
        block_num = 1
    else:
        base_length = 1000
        block_num = 1

    if noisy and 'DUC' in data_set:
        gppl_model_dir = os.path.join('/home/gao//PycharmProjects/gppl_rankers/v1_12Feb18_alQuery/learnt_model',
                                      '10PercentNoisyFeedback',data_set,topic,
                                      'stateNGRAM-base{}-block{}'.format(base_length,block_num),
                                          )
    elif not noisy and 'DUC' in data_set:
        gppl_model_dir = os.path.join('/home/gao/PycharmProjects/gppl_rankers/v1_12Feb18_alQuery/learnt_model','mix2',
                                      data_set,topic,'stateNGRAM-base{}-block{}'.format(base_length,block_num))

    else:
        exit(1) ###TODO: add DBS

    if 'DUC' in data_set:
        vector_path = os.path.join('/home/gao/PycharmProjects/sampling_summaries/2_vectors',
                                    data_set,topic,'synthetic','stateNGRAM-base{}-block{}'.format(base_length,block_num))

        new_learnt_value_list = []
        for model in model_names:
            value_list = getValueListFromGpplModel(summary_list,vector_path,gppl_model_dir,
                                                   learnt_rewards_setting_name,model[:-1])
            new_learnt_value_list.append(value_list)
    else:
        pass
        ###TODO: for DBS
        #vector_path = os.path.join('/home/gao/PycharmProjects/sampling_summaries/2_vectors',
                                   #data_set,topic,'model_{}'.format(model_idx),'synthetic',
                                   #'stateNGRAM-base{}-block{}'.format(base_length,block_num))
        #new_learnt_value_list = getDBSRewardsFromWeights(summary_list,learnt_weights_path,vector_path)


    return new_learnt_value_list




'''
def mixingPretrainedLearntModelRewards(prewards, lrewards_list, tdf):
    new_list = []
    for lrewards in lrewards_list:
        model_value = mixingPretrainedLearntRewards(prewards,lrewards,tdf)
        new_list.append(model_value)

    return new_list


def getGpplLearntRewards(summary_list,data_set,topic,learnt_rewards_setting_name,noisy,model_names):
    if 'DUC' in data_set:
        base_length = 200
        block_num = 1
    else:
        base_length = 1000
        block_num = 1

    if noisy and 'DUC' in data_set:
        gppl_model_dir = os.path.join('/home/gao/PycharmProjects/gppl_learnt_rewards_RL/learnt_model/GPPL',
                                          '10PercentNoisyFeedback',data_set,topic,'stateNGRAM-base{}-block{}'.format(base_length,block_num),
                                          learnt_rewards_setting_name)
    elif not noisy and 'DUC' in data_set:
        gppl_model_dir = os.path.join('/home/gao/PycharmProjects/gppl_learnt_rewards_RL/learnt_model/GPPL',
                                      data_set,topic,'stateNGRAM-base{}-block{}'.format(base_length,block_num),
                                      learnt_rewards_setting_name)
    else:
        exit(1) ###TODO: add DBS

    if 'DUC' in data_set:
        vector_path = os.path.join('/home/gao/PycharmProjects/sampling_summaries/2_vectors',
                                    data_set,topic,'synthetic','stateNGRAM-base{}-block{}'.format(base_length,block_num))

        new_learnt_value_list = []
        for model in model_names:
            value_list = getValueListFromGpplModel(summary_list,vector_path,gppl_model_dir,model[:-1])
            new_learnt_value_list.append(value_list)
    else:
        ###TODO: for DBS
        #vector_path = os.path.join('/home/gao/PycharmProjects/sampling_summaries/2_vectors',
                                   #data_set,topic,'model_{}'.format(model_idx),'synthetic',
                                   #'stateNGRAM-base{}-block{}'.format(base_length,block_num))
        #new_learnt_value_list = getDBSRewardsFromWeights(summary_list,learnt_weights_path,vector_path)


    return new_learnt_value_list

'''



def getLearntRewards(summary_list,data_set,topic,learnt_rewards_setting_name,noisy,model_idx=0):
    if 'DUC' in data_set:
        base_length = 200
        block_num = 1
    else:
        base_length = 1000
        block_num = 1

    if noisy and 'DUC' in data_set:
        learnt_weights_dir = os.path.join('/home/gao/PycharmProjects/sampling_summaries/al_learnt_rewards_weights',
                                          '10PercentNoisyFeedback',data_set,topic,'stateNGRAM-base{}-block{}'.format(base_length,block_num),
                                          learnt_rewards_setting_name)
    elif not noisy and 'DUC' in data_set:
        learnt_weights_dir = os.path.join('/home/gao/PycharmProjects/sampling_summaries/al_learnt_rewards_weights',
                                      data_set,topic,'stateNGRAM-base{}-block{}'.format(base_length,block_num),
                                      learnt_rewards_setting_name)
    elif noisy and 'DBS' in data_set:
        learnt_weights_path = os.path.join('/home/gao/PycharmProjects/sampling_summaries/al_learnt_rewards_weights',
                                          '10PercentNoisyFeedback',data_set,topic,'stateNGRAM-base{}-block{}'.format(base_length,block_num),
                                          learnt_rewards_setting_name,'model_{}'.format(model_idx))
    else: #not noisy and 'DBS' in data_set:
        learnt_weights_path = os.path.join('/home/gao/PycharmProjects/sampling_summaries/al_learnt_rewards_weights',
                                          data_set,topic,'stateNGRAM-base{}-block{}'.format(base_length,block_num),
                                          learnt_rewards_setting_name,'model_{}'.format(model_idx))

    if 'DUC' in data_set:
        vector_path = os.path.join('/home/gao/PycharmProjects/sampling_summaries/2_vectors',
                                    data_set,topic,'synthetic','stateNGRAM-base{}-block{}'.format(base_length,block_num))
        models = os.listdir(learnt_weights_dir)
        model_num = 0
        for model in models:
            if model[0] != '.':
                model_num += 1
        new_learnt_value_list = getRewardsFromWeights(summary_list,learnt_weights_dir,vector_path,len(summary_list),model_num)
    else:
        vector_path = os.path.join('/home/gao/PycharmProjects/sampling_summaries/2_vectors',
                                   data_set,topic,'model_{}'.format(model_idx),'synthetic',
                                   'stateNGRAM-base{}-block{}'.format(base_length,block_num))
        new_learnt_value_list = getDBSRewardsFromWeights(summary_list,learnt_weights_path,vector_path)


    return new_learnt_value_list


def getGpplMixedLearntRewards(sample_num,samples_dir,topic,data_set,setting_name,
                              model_names, noisy=False):
    #get learnt rewards
    if 'DUC' in data_set:
        act_list = readSummariesAndValues(sample_num,samples_dir,topic,'optimal',True)
    else:
        exit(1) ###TODO: DBS

    learnt_rewards_list = getGpplLearntRewards(act_list,data_set,topic,setting_name,noisy, model_names)
    pre_rewards_list = readAccPretrainedRewards(act_list,topic,data_set)

    return act_list, pre_rewards_list, learnt_rewards_list



def getMixedLearntRewards(sample_num,samples_dir,tdf,topic,data_set,acc_rewards,setting_name,
                          verbose=False, reward_type='optimal', noisy=False, model_idx=0):
    #get learnt rewards
    if 'DUC' in data_set:
        act_list = readSummariesAndValues(sample_num,samples_dir,topic,reward_type,True)
        topic_model = topic
    else:
        topic_model = os.path.join(topic, 'model_{}'.format(model_idx))
        if reward_type == 'optimal':
            act_list = readDbsOptimalSummariesAndValues(sample_num,samples_dir,topic_model,True)
        else:
            act_list = readSummariesAndValues(sample_num,samples_dir,topic_model,reward_type,True)


    learnt_rewards_list = getLearntRewards(act_list,data_set,topic,setting_name,noisy, model_idx)
    pre_rewards_list = readAccPretrainedRewards(act_list,topic_model,data_set)

    print('length of pre_rewards: {}; length of learnt rewards: {}'.format(len(pre_rewards_list),len(learnt_rewards_list)))
    mixed_rewards_list = mixingPretrainedLearntRewards(pre_rewards_list,learnt_rewards_list,tdf)

    if not verbose:
        return act_list, mixed_rewards_list
    else:
        return act_list, pre_rewards_list, learnt_rewards_list, mixed_rewards_list

def getSoftmaxList(value_list, strict):
    softmax_list = []
    for value in value_list:
        softmax_list.append(np.exp(value/strict))

    return softmax_list

def getProbSoftmaxList(value_list, strict):
    softmax_list = getSoftmaxList(value_list,strict)
    sumv = sum(softmax_list)*1.0/len(value_list)
    weights_list = []
    for vv in softmax_list:
        weights_list.append(vv/sumv)

    return weights_list

def getExpectation(value_list, strict):
    softmax_list = getSoftmaxList(value_list,strict)
    sumv = sum(softmax_list)
    exp = 0
    for (i,vv) in enumerate(value_list):
        exp += vv*(softmax_list[i]/sumv)

    return exp


