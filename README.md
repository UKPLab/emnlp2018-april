# APRIL: Active Preference ReInforcement Learning, for extractive multi-document summarisation.

APRIL has three stages:
* sample summaries (stage0): randomly generate some summaries and compute their rouge scores
* active preference learning (stage1): actively query the orale for multiple rounds, collect preferences and learn a ranking over summaries from the preferences.
* reinforcement learning (stage2): read the ranking learnt in stage1 and generate the highest-ranked summary


## Prerequisties
* We suggest you to use IDEs like PyCharm to set up this project, so that package paths are managed automatically.
* Python3 (tested with Python 3.6 on Ubuntu 16.04 LTS)
* Install all packages in requirement.txt.

        >> pip3 install -r requirements.txt

* Download ROUGE package from the [link](https://www.isi.edu/licensed-sw/see/rouge/) and place it in the rouge directory

        >> mv RELEASE-1.5.5 summariser/rouge/

* Download the Standford Parser models and jars from the [link](https://nlp.stanford.edu/software/lex-parser.shtml)
and put them to summariser/jars

		>> mv englishPCFG.ser.gz germanPCFG.ser.gz summariser/jars/
		>> mv stanford-parser.jar stanford-parser-3.6.0-models.jar summariser/jars/


## Download and Preprocess data
* Download the DUC01/02/04 data from [the DUC website](https://duc.nist.gov/data.html) and extract the data to folder data.
* Run summariser/data_processer/make_data.py. Each run of it can preprocess one dataset. You can specify the dataset you want to process by setting the variable 'corpus_name' in the main function to appropriate values (e.g., 'DUC2001', 'DUC2002' or 'DUC2004'). The preprocessed data will be put in data/processed_data.

## Stage0: Generate Sample Summaries
* Run 'stage0_sample_summaries.py' will generate the samples
* Setting 'dataset' in the main function to the dataset you want to generate summaries for
* Setting 'summary_num' in the main function to the number of samples you want to generate for a dataset. Its default value is 10001.
* The generated summaries will be put in data/sampled_summaries


## Stage1: Active Preference Learning
* Run 'stage1_active_pref_learning.py'
* Setting 'dataset' to the dataset you want to run the experiment (DUC2001, DUC2002, DUC2004)
* You can select the following active learning algorithms, by setting 'querier_type' to appropriate values:
    * random sampling + BT (set querier_type to 'random'): randomly select two summaries to query the oracle in each interaction round; use the linear BT model to obtain the ranking
    * gibbs distribution + BT (set querier_type to 'gibbs'): select a relative good and a relatively bad summary to query the user in each round; ranking is obtained using the same linear BT method.
    * uncertainty sampling + BT (set querier_type to 'unc'): estimate the uncertainty level of each summary in the sampled summary pool, select the most uncertain pair to query and use BT to obtain the ranking.

## Stage2: Reinforcement Learning
* Since TD performs much better than LSTD, in the current implementation only TD is included. 
* choose reward type: 
    * hueristic (baseline, no interaction)
    * rouge (upper bound, because reference summaries are not available during interaction)
    * read (read the rewards learnt in stage1)




