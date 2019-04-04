%%%%%%%%
% This is quick implementation of Exp C
%%%%%%%%

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

% set save path:
GmmSavePath = fullfile('.','intermediate','gmm',Exp_ID);
EerSavePath = fullfile('.','EER');

FeaturePath = fullfile('.','intermediate','features','All');
metadata = fullfile('..','metadata','all_meta.csv');

% read train protocol (ReMASC)
fileID = fopen(metadata);
protocol = textscan(fileID, '%d,%d,%d,%d,%d,%d,%d,%d,%d');
fclose(fileID);

% get file and label lists
filelist = protocol{1};
labels = protocol{2};
envID = protocol{5};

% split train and eval set
env = {'Env1','Env2','Env3','Env4'};
trainSet = setdiff(env, Env_ID);

% get indices of genuine and spoof files
env_label = strcat("Env",int2str(envID));
genuineIdx = find((labels == 2 & ismember(env_label, trainSet)));
spoofIdx = find((labels == 3 & ismember(env_label, trainSet)));

%% load spoof training data
disp('Load features for GENUINE training data...');
genuineFeatureCell = cell(size(genuineIdx));
parfor i=1:length(genuineIdx)
    
    tmp_fname = strcat(int2str(filelist(genuineIdx(i))),'_cqcc.mat'); 
    filePath = fullfile(FeaturePath, tmp_fname);
    tmp_fea = load(filePath);
    genuineFeatureCell{i} = tmp_fea.x;

end
disp('Done!');

%% load spoof training data
disp('Load features for SPOOF training data...');
spoofFeatureCell = cell(size(spoofIdx));
parfor i=1:length(spoofIdx)
    
    tmp_fname = strcat(int2str(filelist(spoofIdx(i))),'_cqcc.mat'); 
    filePath = fullfile(FeaturePath, tmp_fname);
    tmp_fea = load(filePath);
    spoofFeatureCell{i} = tmp_fea.x;

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


%% Evaluation

evalIdx = find(strcmp(env_label, Env_ID));

% process each evaluation trial: feature extraction and scoring
scores = zeros(size(evalIdx));
disp('Computing scores for evaluation trials...');
h = waitbar(0,'please wait');
l = length(evalIdx);
for i=1:length(evalIdx)
    
    tmp_fname = strcat(int2str(filelist(evalIdx(i))),'_cqcc.mat'); 
    filePath = fullfile(FeaturePath, tmp_fname);
    eval_fea = load(filePath);
    x_cqcc = eval_fea.x;

    % score computation
    llk_genuine = mean(compute_llk(x_cqcc,genuineGMM.m,genuineGMM.s,genuineGMM.w));
    llk_spoof = mean(compute_llk(x_cqcc,spoofGMM.m,spoofGMM.s,spoofGMM.w));
    % compute log-likelihood ratio
    scores(i) = llk_genuine - llk_spoof;
    msg = ['Evaluating ',num2str(i/l*100),'%'];
    waitbar(i/length(filelist),h,msg);
end
disp('Done!');

%% Compute performance (EER)
% [Pmiss,Pfa] =
% rocch(scores(strcmp(labels,'genuine')),scores(strcmp(labels,'spoof')));
[Pmiss,Pfa] = rocch(scores(labels(evalIdx) == 2),scores(labels(evalIdx) == 3));
EER = rocch2eer(Pmiss,Pfa) * 100;
eer_name = strcat(Exp_ID, Env_ID, '.mat');
eer_path = fullfile(EerSavePath, eer_name);
save(eer_path, 'EER');
fprintf('EER is %.2f\n', EER);



