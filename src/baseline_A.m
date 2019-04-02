%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ReMASC Dataset Baseline Experiment A:
% 
% ====================================================================================
% Matlab Implementation of the baseline system for replay detection.
% Feature Extractor: constant Q cepstral coefficients (CQCC) 
% Classifier: Gaussian Mixture Models (GMMs)
% Experiment Setup: Trained on RedDot Dataset (ASVspoof2017 Challenge
% Dataset), evaluated on ReMASC dataset under each environment settings.
% Training options on ASV2017 dataset: 1) train; 2) train + dev; 3) train + dev + eval
% ====================================================================================
%
% Download ReMASC dataset at: xxxxxxx
% Cite our paper:
% xxxxxxxxxxxxx
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% clear; close all; clc;
Exp_ID = 'ExpA1'
Env_ID = 'Env1'

% add required libraries to the path
addpath(genpath('utility'));
addpath(genpath('CQCC_v1.0'));
addpath(genpath('bosaris_toolkit'));
% load vlfeat (for gmm training)
vlfeat_path = fullfile('.','vlfeat-0.9.21','toolbox','vl_setup');
run(vlfeat_path);

% set paths to the wave files and protocols
pathToTrainData = fullfile('..','data','ASVspoof2017_train');
pathToEvalData = fullfile('..','data',Env_ID);
trainProtocolFile = fullfile('..','data','ASVspoof2017_protocol', 'ASVspoof2017_train.txt');
evalProtocolFile = fullfile('..','metadata',strcat(Env_ID,'_meta'),strcat(Env_ID,'_meta_aligned.csv'));

% set save path:
GmmSavePath = fullfile('.','intermediate','gmm',Exp_ID);
TrainFeatureSavePath = fullfile('.','intermediate','features',Exp_ID,'trn');
EvalFeatureSavePath = fullfile('.','intermediate','features',Exp_ID,'eval');
EerSavePath = fullfile('.','EER');

% read train protocol (ASVspoof2017)
fileID = fopen(trainProtocolFile);
protocol = textscan(fileID, '%s%s%s%s%s%s%s');
fclose(fileID);

% get file and label lists
filelist = protocol{1};
labels = protocol{2};

% get indices of genuine and spoof files
genuineIdx = find(strcmp(labels,'genuine'));
spoofIdx = find(strcmp(labels,'spoof'));

%% Feature extraction for training data

% extract features for GENUINE training data and store in cell array
disp('Extracting features for GENUINE training data...');
genuineFeatureCell = cell(size(genuineIdx));
parfor i=1:length(genuineIdx)
    filePath = fullfile(pathToTrainData,filelist{genuineIdx(i)});
    [x,fs] = audioread(filePath);

    tmp_fea = cqcc(x, fs, 96, fs/2, fs/2^10, 16, 29, 'ZsdD');
    genuineFeatureCell{i} = tmp_fea;
    
    save_name = strrep(filelist{genuineIdx(i)}, '.wav', '_cqcc.mat');
    save_path = fullfile(TrainFeatureSavePath, save_name);
    parsave(save_path, tmp_fea);
end
disp('Done!');

% extract features for SPOOF training data and store in cell array
disp('Extracting features for SPOOF training data...');
spoofFeatureCell = cell(size(spoofIdx));
parfor i=1:length(spoofIdx)
    filePath = fullfile(pathToTrainData,filelist{spoofIdx(i)});
    [x,fs] = audioread(filePath);

    tmp_fea = cqcc(x, fs, 96, fs/2, fs/2^10, 16, 29, 'ZsdD');
    spoofFeatureCell{i} = tmp_fea;
    
    save_name = strrep(filelist{spoofIdx(i)}, '.wav', '_cqcc.mat');
    save_path = fullfile(TrainFeatureSavePath, save_name);
    parsave(save_path, tmp_fea);
end
disp('Done!');

%% GMM training

% train GMM for GENUINE data
disp('Training GMM for GENUINE...');
[genuineGMM.m, genuineGMM.s, genuineGMM.w] = vl_gmm([genuineFeatureCell{:}], 512, 'verbose', 'MaxNumIterations',100);
gen_save_path = fullfile(GmmSavePath,'genuineGMM.mat');
save(gen_save_path,'-struct','genuineGMM');
disp('Done!');

% train GMM for SPOOF data
disp('Training GMM for SPOOF...');
[spoofGMM.m, spoofGMM.s, spoofGMM.w] = vl_gmm([spoofFeatureCell{:}], 512, 'verbose', 'MaxNumIterations',100);
spf_save_path = fullfile(GmmSavePath,'spoofGMM.mat');
save(spf_save_path,'-struct','spoofGMM');
disp('Done!');


%% Feature extraction and scoring of ReMASC data

% read Evaluation protocol
fileID = fopen(evalProtocolFile);
protocol = textscan(fileID, '%d,%d,%d,%d,%d,%d,%d,%d,%d');
fclose(fileID);

% get file and label lists
filelist = protocol{1};
labels = protocol{2};

% process each evaluation trial: feature extraction and scoring
scores = zeros(size(filelist));
disp('Computing scores for evaluation trials...');
h = waitbar(0,'please wait');
l = length(filelist);
for i=1:length(filelist)
%     tmp_fname = strcat(int2str(filelist(i)), '.wav');
    tmp_fname = strcat(sprintf('%06d',filelist(i)),'.wav'); %for env1 only!!!
    filePath = fullfile(pathToEvalData, tmp_fname);
%     [x,fs] = audioread(filePath);
    
    [x, fs] = ReSamp(filePath, 16000);
    % featrue extraction
    tmp_fea = cqcc(x(:,1), fs, 96, fs/2, fs/2^10, 16, 29, 'ZsdD');
    x_cqcc = tmp_fea;
    
    save_name = strcat(int2str(filelist(i)),'_cqcc.mat');
    save_path = fullfile(EvalFeatureSavePath, save_name);
%     parsave(save_path, tmp_fea);

    % score computation
    llk_genuine = mean(compute_llk(x_cqcc,genuineGMM.m,genuineGMM.s,genuineGMM.w));
    llk_spoof = mean(compute_llk(x_cqcc,spoofGMM.m,spoofGMM.s,spoofGMM.w));
    % compute log-likelihood ratio
    scores(i) = llk_genuine - llk_spoof;
    msg = ['Evaluating',num2str(i/l*100),'%'];
    waitbar(i/length(filelist),h,msg);
end
disp('Done!');


%% Compute performance (EER)
% [Pmiss,Pfa] =
% rocch(scores(strcmp(labels,'genuine')),scores(strcmp(labels,'spoof')));
[Pmiss,Pfa] = rocch(scores(labels == 2),scores(labels == 3));
EER = rocch2eer(Pmiss,Pfa) * 100;
eer_name = strcat(Exp_ID, Env_ID, '.mat');
eer_path = fullfile(EerSavePath, eer_name);
save(eer_path, 'EER');
fprintf('EER is %.2f\n', EER);


%% Other Functions
function parsave(fname, x)
    save(fname, 'x', '-v6')
end

function [x_new, tar_freq] = ReSamp(fname, tar_freq)
    [x,fs] = audioread(fname);
    [P,Q] = rat(tar_freq/fs);
    x_new = resample(x, P, Q);
end


