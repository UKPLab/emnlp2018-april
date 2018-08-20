# APRIL: Active Preference ReInforcement Learning, for extractive multi-document summarisation.

APRIL has three stages:
* sample summaries (stage1): randomly generate some summaries and compute their rouge scores
* active preference learning (stage2): actively query the orale for multiple rounds, collect preferences and learn a ranking over summaries from the preferences.
* reinforcement learning (stage3): read the ranking learnt in stage1 and generate the highest-ranked summary


## Prerequisties
* Python3 (tested with Python 3.6 on Ubuntu 16.04 LTS)
* install all packages in requirement.txt.

'''
pip3 install -r requirements.txt
'''

* Download ROUGE package from the [link](https://www.isi.edu/licensed-sw/see/rouge/) and place it in the rouge directory

'''
mv RELEASE-1.5.5 rouge/
'''

* Download the Standford Parser models and jars from the [link](https://nlp.stanford.edu/software/lex-parser.shtml)
and put them to summariser/jars

		>> mv englishPCFG.ser.gz germanPCFG.ser.gz summariser/jars/
		>> mv stanford-parser.jar stanford-parser-3.6.0-models.jar summariser/jars/



## Implemented algorithms in stage2 (active preference learning)
* random sampling + BT (in stage1_active_pref_learning.py, set querier_type to 'random'): randomly select two summaries to query the oracle in each interaction round; use the linear BT model to obtain the ranking
* gibbs distribution + BT (set querier_type to 'gibbs'): select a relative good and a relatively bad summary to query the user in each round; ranking is obtained using the same linear BT method.
* uncertainty sampling + BT (set querier_type to 'unc'): estimate the uncertainty level of each summary in the sampled summary pool, select the most uncertain pair to query and use BT to obtain the ranking.

## Impelmented algorithms in stage3 (reinforcement learning)
* Since TD performs much better than LSTD, in the current implementation only TD is included. 
* choose reward type: 
    * hueristic (baseline, no interaction)
    * rouge (upper bound, because reference summaries are not available during interaction)
    * read (read the rewards learnt in stage1)




