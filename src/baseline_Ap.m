%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ReMASC Dataset Baseline Experiment A+:
% 
% ====================================================================================
% Matlab Implementation of the baseline system for replay detection.
% Feature Extractor: constant Q cepstral coefficients (CQCC) 
% Classifier: Gaussian Mixture Models (GMMs)
% Experiment Setup: Trained on ReMASC dataset (on all environments), evaluated on 
% RedDot Dataset (ASVspoof2017 Challenge Dataset).
% ====================================================================================
%
% Download ReMASC dataset at: xxxxxxx
% Cite our paper:
% xxxxxxxxxxxxx
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% clear; close all; clc;
Exp_ID = 'ExpAp1'
Env_ID = 'Env4'

% add required libraries to the path
addpath(genpath('utility'));
addpath(genpath('CQCC_v1.0'));
addpath(genpath('bosaris_toolkit'));
% load vlfeat (for gmm training)
vlfeat_path = fullfile('.','vlfeat-0.9.21','toolbox','vl_setup');
run(vlfeat_path);

% set paths to the wave files and protocols
pathToTrainData = fullfile('..','data',Env_ID);
pathToEvalData = fullfile('..','data','ASVspoof2017_eval');
trainProtocolFile = fullfile('..','metadata',strcat(Env_ID,'_meta'),strcat(Env_ID,'_meta_aligned.csv'));
evalProtocolFile = fullfile('..','data','ASVspoof2017_protocol', 'ASVspoof2017_eval.txt');

% set save path:
GmmSavePath = fullfile('.','intermediate','gmm',Exp_ID);
TrainFeatureSavePath = fullfile('.','intermediate','features',Exp_ID,'trn');
EvalFeatureSavePath = fullfile('.','intermediate','features',Exp_ID,'eval');
EerSavePath = fullfile('.','EER');

% read train protocol (ReMASC)
fileID = fopen(trainProtocolFile);
protocol = textscan(fileID, '%d,%d,%d,%d,%d,%d,%d,%d,%d');
fclose(fileID);

% get file and label lists
filelist = protocol{1};
labels = protocol{2};

% get indices of genuine and spoof files
genuineIdx = find(labels == 2);
spoofIdx = find(labels == 3);

%% Feature extraction for training data

% extract features for GENUINE training data and store in cell array
disp('Extracting features for GENUINE training data...');
genuineFeatureCell = cell(size(genuineIdx));
parfor i=1:length(genuineIdx)
    tmp_fname = strcat(int2str(filelist(genuineIdx(i))), '.wav');
    filePath = fullfile(pathToEvalData, tmp_fname);
    
    [x, fs] = ReSamp(filePath, 16000);
    % featrue extraction
    tmp_fea = cqcc(x(:,1), fs, 96, fs/2, fs/2^10, 16, 29, 'ZsdD');
    genuineFeatureCell{i} = tmp_fea;
    
    save_name = strcat(int2str(filelist(genuineIdx(i))),'_cqcc.mat');
    save_path = fullfile(TrainFeatureSavePath, save_name);
    parsave(save_path, tmp_fea);
end
disp('Done!');

% extract features for SPOOF training data and store in cell array
disp('Extracting features for SPOOF training data...');
spoofFeatureCell = cell(size(spoofIdx));
parfor i=1:length(spoofIdx)
    tmp_fname = strcat(int2str(filelist(spoofIdx(i))), '.wav');
    filePath = fullfile(pathToEvalData, tmp_fname);
    
    [x, fs] = ReSamp(filePath, 16000);
    % featrue extraction
    tmp_fea = cqcc(x(:,1), fs, 96, fs/2, fs/2^10, 16, 29, 'ZsdD');
    spoofFeatureCell{i} = tmp_fea;
    
    save_name = strcat(int2str(filelist(spoofIdx(i))),'_cqcc.mat');
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



%% Other Functions
function parsave(fname, x)
    save(fname, 'x', '-v6')
end

function [x_new, tar_freq] = ReSamp(fname, tar_freq)
    [x,fs] = audioread(fname);
    [P,Q] = rat(tar_freq/fs);
    x_new = resample(x, P, Q);
end


