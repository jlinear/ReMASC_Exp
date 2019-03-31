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
| ID | Training | Eval-Env1 | Eval-Env2 | Eval-Env3 | Eval-Env4 |
| --- | --- | --- | --- | --- | --- |
| A1 | ASV17_train | 49.74 |  | 41.03 | 48.18 |
| A2 | ASV17_train+dev | 47.10 |  | 48.95 | 39.63 |


### Exp A+:
| ID | Training | Eval |
| --- | --- | --- |
| Ap1 |  | ASV17_eval |
| Ap2 | Env2 | ASV17_eval |
| Ap3 | Env3 | ASV17_eval |
| Ap4 | Env4 | ASV17_eval |
| Ap5 | All | ASV17_eval 40? |
| Ap6 | All+train+dev | ASV17_eval <25? |

### Exp B:




### Exp C:








