%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ReMASC Dataset Baseline Experiment C:
% 
% ====================================================================================
% Matlab Implementation of the baseline system for replay detection.
% Feature Extractor: constant Q cepstral coefficients (CQCC) 
% Classifier: Gaussian Mixture Models (GMMs)
% Experiment Setup: Trained on any three environment settings while evaluated on the 
% remaining environment setting.
% ====================================================================================
%
% Download ReMASC dataset at: xxxxxxx
% Cite our paper:
% xxxxxxxxxxxxx
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; close all; clc;
Exp_ID = 'ExpC1'
Env_ID = 'Env1'

% add required libraries to the path
addpath(genpath('utility'));
addpath(genpath('CQCC_v1.0'));
addpath(genpath('bosaris_toolkit'));
% load vlfeat (for gmm training)
vlfeat_path = fullfile('.','vlfeat-0.9.21','toolbox','vl_setup');
run(vlfeat_path);

% split train and eval set
env = {'Env1','Env2','Env3','Env4'};
trainSet = setdiff(env, Env_ID);
evalSet = {Env_ID};



% set save path:
GmmSavePath = fullfile('.','intermediate','gmm',Exp_ID);
TrainFeatureSavePath = fullfile('.','intermediate','features',Exp_ID,'trn');
EvalFeatureSavePath = fullfile('.','intermediate','features',Exp_ID,'eval');
EerSavePath = fullfile('.','EER');





