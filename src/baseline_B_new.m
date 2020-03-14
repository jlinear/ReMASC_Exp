%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ReMASC Dataset Baseline Experiment B_new:
% 
% ====================================================================================
% Matlab Implementation of the baseline system for replay detection.
% Feature Extractor: constant Q cepstral coefficients (CQCC) 
% Classifier: Gaussian Mixture Models (GMMs)
% Experiment Setup: randomly split ReMASC dataset into two parts: core and eval. 
% The evaluation is conducted on each device setting independently.
% ====================================================================================
%
% Download ReMASC dataset at: https://github.com/YuanGongND/ReMASC
% Cite our paper:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Parameters setup
clear; close all; clc;
rng(0);
EXP_ID = 'ExpBn1';
DVC_ID = 1;

% add required libraries to the path
addpath(genpath('utility'));
addpath(genpath('CQCC_v1.0'));
addpath(genpath('bosaris_toolkit'));
% load vlfeat (for gmm training)
vlfeat_path = fullfile('.','vlfeat-0.9.21','toolbox','vl_setup');
run(vlfeat_path);

% set paths to the wave files and protocols
pathToTrainData = fullfile('..','core_44100','data');
pathToEvalData = fullfile('..','eval_44100','data');
trainProtocolFile = fullfile('..','core_44100','meta.csv');
evalProtocolFile = fullfile('..','eval_44100','meta.csv');

% set save path:
GmmSavePath = fullfile('.','intermediate','gmm',EXP_ID);
TrainFeatureSavePath = fullfile('.','intermediate','features',EXP_ID,'trn');
EvalFeatureSavePath = fullfile('.','intermediate','features',EXP_ID,'eval');
EerSavePath = fullfile('.','EER');
if ~(exist(GmmSavePath, 'dir') && exist(TrainFeatureSavePath, 'dir') ...
    && exist(EvalFeatureSavePath, 'dir') && exist(EerSavePath, 'dir'))
    mkdir(GmmSavePath)
    mkdir(TrainFeatureSavePath)
    mkdir(EvalFeatureSavePath)
    mkdir(EerSavePath)
end

% read train protocol (ReMASC)
fileID = fopen(trainProtocolFile);
protocol = textscan(fileID, '%d,%d,%d,%d,%d,%d,%d,%d,%d');
fclose(fileID);

% get file and label lists
filelist = protocol{1};
labels = protocol{2};
rec_device = protocol{8};

%% Feature extraction for training data

% get indices of genuine and spoof files
genuineIdx = find((labels == 2 & rec_device == DVC_ID));
spoofIdx = find((labels == 3 & rec_device == DVC_ID));

% extract features for GENUINE training data and store in cell array
disp('Extracting features for GENUINE training data...');
genuineFeatureCell = cell(size(genuineIdx));
parfor i=1:length(genuineIdx)
%     tmp_fname = strcat(sprintf('%06d',filelist(genuineIdx(i))),'.wav'); %for env1 only!!!
    tmp_fname = strcat(int2str(filelist(genuineIdx(i))), '.wav');
    filePath = fullfile(pathToTrainData, tmp_fname);
    [x,fs] = audioread(filePath);
%     [x, fs] = ReSamp(filePath, 44100);
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
%     tmp_fname = strcat(sprintf('%06d',filelist(spoofIdx(i))),'.wav'); %for env1 only!!!
    tmp_fname = strcat(int2str(filelist(spoofIdx(i))), '.wav');
    filePath = fullfile(pathToTrainData, tmp_fname);
    [x,fs] = audioread(filePath);
%     [x, fs] = ReSamp(filePath, 44100);
    % featrue extraction
    tmp_fea = cqcc(x(:,1), fs, 96, fs/2, fs/2^10, 16, 29, 'ZsdD');
%     try
%         tmp_fea = cqcc(x(:,1), fs, 96, fs/2, fs/2^10, 16, 29, 'ZsdD');
%     catch
%         disp(tmp_fname);
%     end
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

%% Feature extraction and scoring of eval data

% read Evaluation protocol
fileID = fopen(evalProtocolFile);
protocol = textscan(fileID, '%d,%d,%d,%d,%d,%d,%d,%d,%d');
fclose(fileID);

% get file and label lists
filelist = protocol{1};
labels = protocol{2};
rec_device = protocol{8};

evalIdx = find(rec_device == DEV_ID);

% process each evaluation trial: feature extraction and scoring
scores = zeros(size(evalIdx));
disp('Computing scores for evaluation trials...');
h = waitbar(0,'please wait');
l = length(evalIdx);
for i=1:length(evalIdx)
    tmp_fname = strcat(int2str(filelist(evalIdx(i))), '.wav');
    filePath = fullfile(pathToEvalData, tmp_fname);
    [x,fs] = audioread(filePath);
    
%     [x, fs] = ReSamp(filePath, 44100);
    % featrue extraction
    tmp_fea = cqcc(x(:,1), fs, 96, fs/2, fs/2^10, 16, 29, 'ZsdD');
    x_cqcc = tmp_fea;
    
    save_name = strcat(int2str(filelist(evalIdx(i))),'_cqcc.mat');
    save_path = fullfile(EvalFeatureSavePath, save_name);
%     parsave(save_path, tmp_fea);

    % score computation
    llk_genuine = mean(compute_llk(x_cqcc,genuineGMM.m,genuineGMM.s,genuineGMM.w));
    llk_spoof = mean(compute_llk(x_cqcc,spoofGMM.m,spoofGMM.s,spoofGMM.w));
    % compute log-likelihood ratio
    scores(i) = llk_genuine - llk_spoof;
    msg = ['Evaluating',num2str(i/l*100),'%'];
    waitbar(i/length(evalIdx),h,msg);
end
disp('Done!');

%% Compute performance (EER)
% [Pmiss,Pfa] =
% rocch(scores(strcmp(labels,'genuine')),scores(strcmp(labels,'spoof')));
[Pmiss,Pfa] = rocch(scores(labels(evalIdx) == 2),scores(labels(evalIdx) == 3));
EER = rocch2eer(Pmiss,Pfa) * 100;
eer_name = strcat(Exp_ID, '_DVC_', int2str(DVC_ID), '.mat');
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
