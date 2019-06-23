# ReMASC Experiments
This repository describes the experimental design and Matlab code for ReMASC , a Realistic Replay Attack Corpus for Voice Controlled Systems. The data collection process can be found in the following publication:

Yuan Gong, Jian Yang, Jacob Huber, Mitchell MacKnight, Christian Poellabauer, (https://arxiv.org/abs/1904.03365)ReMASC: Realistic Replay Attack Corpus for Voice Controlled Systems*", arXiv preprint, April 2019.

## Hierarchy:

Before running the experiments, please download the dataset and split it based on the file hierarchy shown below. 

1. *metadata* directory contains metadata descriptions for all .wav files.  

.
+-- Env1_meta
|	+-- Env1_meta_aligned.csv
|	+-- Env1_meta.csv
...
+-- all_meta.csv
+-- gender.txt

2. *data* directory contains all data files in the form of .wav  

.
+-- ASVspoof2017_eval
|	+-- ****.wav
+-- ASVspoof2017_protocol
|	+-- ****.txt
|	+-- ...
+-- ASVspoof2017_train
|	+-- ****.wav
+-- ASVspoof2017_train_dev
|	+-- ****.wav
+-- Env1
|	+-- ****.wav
...

3. *src* directory stores all Matlab source code and dependencies.

**vlfeat** is a very important dependency we used in the experiments, in case that the attached dependency does not work, you can download the dependency from http://www.vlfeat.org/index.html
For each experiment indicated below, creat a directory hierarchy shown below under this path (./src). The imtermediate variables (features and GMM models) will be stored through this path. 

.
+-- intermediate
|	+-- features
|		+-- ExpID
|			+-- eval
|			+-- trn
|	+-- gmm
|		+-- ExpID

## Experiments:

We use the official [ASVspoof 2017 Challenge](http://www.asvspoof.org/index2017.html) baseline CQCC-GMM model that  uses constant Q cepstral coefficients (CQCC) features and Gaussian mixture model (GMM) classifier and apply exactly the same hyper-parameters. We further use conventional equal error rate (EER) as the metric for evaluation.

### Exp A:
In this experiment, we train the baseline CQCC-GMM model using the RedDot Replayed dataset, and then test it on the ReMASC dataset. In ExpA1, only the RedDot training set is used for training, while in Exp_A2, the RedDot training and development set are both for training.

**Run the code:**
```
baseline_A.m
```
For each experiment shown in the table below, change the Experiment ID and Environment ID according to the data path, and make sure the path defined in the code are all valid.

| ID | Training | Eval-Env1 | Eval-Env2 | Eval-Env3 | Eval-Env4 |
| --- | --- | --- | --- | --- | --- |
| A1 | ASV17_train | 49.74 | NA | 41.03 | 48.18 |
| A2 | ASV17_train+dev | 47.10 | 44.54 | 48.95 | 39.63 |


### Exp A+:
**Run the code:**
```
baseline_Ap.m	
```
Similar to Exp A, change the Env ID accordingly. 

| ID | Training | Eval-RedDot |
| --- | --- | --- |
| Ap1 | All+train+dev |  20.2 |


### Exp B:
For each environment, we split the ReMASC dataset randomly into two disjoint and speaker-independent sets of roughly same size and then train the baseline CQCC-GMM defense model (referred to as Env-Dependent in Table 3 of the paper) on one set and test on the other.

**Run the code:**
```
baseline_B.m	
```
Similar to the above experiments, make sure the path are valid, and the only thing you need to do is change the experiment ID and environment ID before running. 

| ID | Env | Train (speaker IDs) | Eval (speaker IDs) | EER |
| --- | --- | --- | --- | --- |
| B1 | Env1 | -1, 1, 3, 5, 6, 8, 11 | 2, 4, 7, 9, 10, 12 | 13.50 |
| B2 | Env2 | 11, 13, 15, 17, 20, 22, 23, 25, 28, 30, 31, 33 | 14, 16, 18, 19, 21, 24, 26, 27, 29, 32, 34 | 17.38 |
| B3 | Env3 | 0, 1, 27, 29, 31, 35, 36, 38, 41 | 26, 28, 30, 32, 37, 39, 40, 42, 49 | 21.31 |
| B4 | Env4 | 0, 1, 42, 43, 44, 45 | 11, 46, 47, 48, 49 ,50 | 22.11 |


### Exp C:
When testing in each target environment, we train the baseline CQCC-GMM model using data of three environments other than the target environment (referred to as Env-Independent in Table 3 of the paper).

**Run the code:**
```
baseline_C.m	
```
Due to the large amount of data included in this set of experiments, you can use the provided code for feature extraction and model training for each of the environment (comment out the evaluation section) and use the ‘*runExpC.m’* script  (unzip the runExpC.zip file) for fast evaluation. We strongly recommend that you use a machine with 64GB+ RAM to run this experiment. 

| ID | Training | Eval | EER |
| --- | --- | --- | --- |
| C1 | Env 2,3,4 | Env1 | 19.9 |
| C2 | Env 1,3,4 | Env2 | 39.9 |
| C3 | Env 1,2,4 | Env3 | 34.6 |
| C4 | Env 1,2,3 | Env4 | 48.9 |



