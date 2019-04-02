# ReMASC Dataset
Welcome to use ReMASC dataset. Here is the description of the data files:

## Directory structure:

1 *Metadata* directory contains metadata descriptions for all .wav files.   
2 *Data* directory contains all data files in the form of .wav  

## Annotations:
The Annotation.xlsx file records the full annotation information of the audio files.

Specially, the file contains 9 columns:

**Column 1:** Unique File ID, aligned with the file name.

**Column 2:** Speech Type ID (1 = source, 2 = genuine, 3 = replayed).

**Column 3:** Speaker ID, i.e., Unique Speaker Identifier, consistent with the UID in "Recruitment Form.xlsx".

**Column 4:** Command ID, consistent with the ID in "command.xlsx".

**Column 5:** Environment ID (1 = outdoor, 2 = indoor environment 1, 3 = indoor environment 2, 4 = vehicle environment), see the paper for details.

**Column 6**: Position ID (-1 = N/A, Env1: 1 = 0.5m 2 = 1.5m; Env4: 1 = quite 2 = moving)

**Column 7:** Source Recorder ID ( -1 = N/A, 1 = DR_05, 2 = iPod, 3 = TTS)

**Column 8:** Playback Device ID ( -1 = N/A, 1 = Audio Technica ATH-AD700X headphone, 2 = iPod Touch, 3 = Sony SRS*X11, 4 = Sony SRS*X5)

**Column 9:** Recording Device ID ( -1 = N/A, 1 = AIY, 2 = Respeaker 4_Linear, 3 = Respeaker Core V2.0, 4 = Amlogic A113X1)


## Experiment Outline:

### Exp A:
Matlab code: baseline_A.m

| ID | Training | Eval-Env1 | Eval-Env2 | Eval-Env3 | Eval-Env4 |
| --- | --- | --- | --- | --- | --- |
| A1 | ASV17_train | 49.74 |  | 41.03 | 48.18 |
| A2 | ASV17_train+dev | 47.10 |  | 48.95 | 39.63 |


### Exp A+:
Matlab code: baseline_Ap.m

| ID | Training | Eval |
| --- | --- | --- |
| Ap1 | All |  40? |
| Ap2 | All+train+dev |  <25? |
| Ap1_4 | Env4(waiting env2) | 45.68 |


### Exp B:
Matlab code: baseline_B.m

| ID | Env | Train (speaker IDs) | Eval (speaker IDs) | EER |
| --- | --- | --- | --- | --- |
| B1 | Env1 | 1, 3, 5, 6, 8, 11 | 2, 4, 7, 9, 10, 12 |  |
| B2 | Env2 |  |  |  |
| B3 | Env3 | 0, 1, 27, 29, 31, 35, 36, 38, 41 | 26, 28, 30, 32, 37, 39, 40, 42, 49 |  |
| B4 | Env4 | 0, 1, 42, 43, 44, 45 | 11, 46, 47, 48, 49 ,50 | 22.11 |


### Exp C:
Matlab code: baseline_C.m

| ID | Training | Eval | EER |
| --- | --- | --- | --- |
| C1 | Env 2,3,4 | Env1 |  |
| C2 | Env 1,3,4 | Env2 |  |
| C3 | Env 1,2,4 | Env3 |  |
| C4 | Env 1,2,3 | Env4 |  |



### Exp D:
S+R 



