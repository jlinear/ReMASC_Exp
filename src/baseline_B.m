%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ReMASC Dataset Baseline Experiment B:
% 
% ====================================================================================
% Matlab Implementation of the baseline system for replay detection.
% Feature Extractor: constant Q cepstral coefficients (CQCC) 
% Classifier: Gaussian Mixture Models (GMMs)
% Experiment Setup: Based on speakers' IDs, ReMASC dataset is randomly split into
% two subsets of roughly equal size, one for training, and the other for
% evaluating. The evaluation is conducted on each environment independently.
% ====================================================================================
%
% Download ReMASC dataset at: xxxxxxx
% Cite our paper:
% xxxxxxxxxxxxx
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; close all; clc;

% add required libraries to the path
addpath(genpath('utility'));
addpath(genpath('CQCC_v1.0'));
addpath(genpath('bosaris_toolkit'));
% load vlfeat (for gmm training)
run('vlfeat-0.9.21/toolbox/vl_setup');

% set paths to the wave files and protocols
pathToTrainData = fullfile('..','data','ASVspoof2017_train_dev');
pathToEvalData = fullfile('..','data','Env3');
trainProtocolFile = fullfile('..','data','ASVspoof2017_protocol', 'ASVspoof2017_train_dev.txt');
evalProtocolFile = fullfile('..','metadata','Env3_meta','Env3_meta_aligned.csv');

