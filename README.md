# APRIL: Active Preference ReInforcement Learning, for extractive multi-document summarisation.

APRIL has three stages:
* sample summaries (stage0): randomly generate some summaries and compute their rouge scores
* active preference learning (stage1): actively query the orale for multiple rounds, collect preferences and learn a ranking over summaries from the preferences.
* reinforcement learning (stage2): read the ranking learnt in stage1 and generate the highest-ranked summary


## Installing
Use Python3 (tested with Python 3.6 on Ubuntu 16.04 LTS), install all packages in requirement.txt.

## Structure
* data: processed DUC data and sampled summaries (generated in stage0)
* summariser: contains multiple sub-directories. Names of the sub-directories indicate the files then contain.

## Implemented algorithms in stage1 (active preference learning)
* random sampling + BT (in stage1_active_pref_learning.py, set querier_type to 'random'): randomly select two summaries to query the oracle in each interaction round; use the linear BT model to obtain the ranking
* gibbs distribution + BT (set querier_type to 'gibbs'): select a relative good and a relatively bad summary to query the user in each round; ranking is obtained using the same linear BT method.
* uncertainty sampling + BT (set querier_type to 'unc'): estimate the uncertainty level of each summary in the sampled summary pool, select the most uncertain pair to query and use BT to obtain the ranking.

## Impelmented algorithms in stage2 (reinforcement learning)
* Since TD performs much better than LSTD, in the current implementation only TD is included. 
* choose reward type: 
    * hueristic (baseline, no interaction)
    * rouge (upper bound, because reference summaries are not available during interaction)
    * read (read the rewards learnt in stage1)




