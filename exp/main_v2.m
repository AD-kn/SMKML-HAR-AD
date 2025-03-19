% 最终加速版本


clear; 
clc; 
profile on;

% 设定路径
data_path = fullfile(pwd, '..',  filesep, "data", filesep);
addpath(data_path);
lib_path = fullfile(pwd, '..',  filesep, "lib", filesep);
addpath(genpath(lib_path));
code_path = genpath(fullfile(pwd, '..',  filesep, 'M3KNN_3'));
addpath(code_path);

% 参数设置
nRepeat = 1;
nFold = 5; % 5折交叉验证
inlier_class_id = 1;
K_1 = 5; % 第1近邻
K_2 = 5; % 第2近邻
nMeasure = 8; % 评价指标
k = 5;
blockSize = 10000;

res_all = cell(1,1);
datasetCandi = dir(fullfile(data_path, '*.mat'));  
exp_n = 'M3KNN_3';
iSeed = 0;

for i1 =1%: length(datasetCandi) 

    data_name = datasetCandi(i1).name(1:end-4);
    split_dir = fullfile(pwd, '..', 'data_split', data_name);
    iSeed = 0; 

   
    res_te_k = zeros(K_1, K_2, nMeasure);
    res_tr_val_k = zeros(K_1, K_2, nMeasure);
    

    for k1 = 1%: K_1
        for k2 = 5%: K_2
            res = [];
            res.k1 = k1;
            res.k2 = k2;
            
            res_tr_val_repeat = zeros(nRepeat, nMeasure);
            res_te_repeat = zeros(nRepeat, nMeasure);
            iter_repeat = zeros(nRepeat, 1); 
            
            for iRepeat = 1:nRepeat
                iSeed = iSeed + 1;
       
                res_tr_val_fold = zeros(nFold, nMeasure);
                res_te_fold = zeros(nFold, nMeasure);   
                res_model_nfold = cell(nFold, 1);
                iter_fold = zeros(nFold, 1); 

                for iFold =1:nFold
                    split_name = sprintf('pos_%d_repeat_%d_fold_%d.mat', inlier_class_id, iRepeat, iFold);
                    load(fullfile(split_dir, split_name), 'X_tr_m', 'X_te_m', 'X_val_m', 'Y_tr', 'Y_te', 'Y_val');    
                    [nSmp, nFea] = size(X_tr_m);
                    

                    W_tr_tr = compute_Poly_weight_tr_tr(X_tr_m, k); 
                               
                    disp(['Data: ' num2str(i1) '/' num2str(length(datasetCandi)), ...
                        ', iJNN: ' num2str(k1) '/' num2str(K_1), ...
                        ', iKNN: ' num2str(k2) '/' num2str(K_2), ...
                        ', iRepeat: ' num2str(iRepeat) '/' num2str(nRepeat), ...
                        ', iFold: ' num2str(iFold) '/' num2str(nFold)]);

                    iSeed = iSeed + 1;
                    [res_model_nfold{iFold}, res_tr_val_fold(iFold, :), res_te_fold(iFold, :), iter] = ...
                        one_round_train_test_eval(X_tr_m, W_tr_tr, Y_tr, X_te_m, Y_te, X_val_m, Y_val, k1, k2, iSeed, blockSize,k);

                    iter_fold(iFold) = iter; 
                    clear X_tr_m X_te_m X_val_m Y_tr Y_te Y_val W_tr_tr

                end
                iter_repeat(iRepeat) = mean(iter_fold);
                res_tr_val_repeat(iRepeat, :) = mean(res_tr_val_fold, 1);
                res_te_repeat(iRepeat, :) = mean(res_te_fold, 1);    
                clear res_tr_val_fold res_te_fold iter_fold res_model_nfold

            end

            avg_iter_k1k2 = mean(iter_repeat);
            res.avg_iter = avg_iter_k1k2; 
            res_te_k(k1, k2, :) = mean(res_te_repeat, 1);
            res_tr_val_k(k1, k2, :) = mean(res_tr_val_repeat, 1);
        end
    end

    res.te = res_te_k;
    res.tr_val = res_tr_val_k;
    res.performance = {'accuracy', 'sensitivity', 'specificity', 'precision', 'recall', 'f_measure', 'gmean', 'AUPR'};
 

    save([data_name, '_result_all.mat'], 'res');
    clear res_tr_val_repeat res_te_repeat iter_repeat res
       
end

rmpath(data_path);
rmpath(lib_path);
rmpath(code_path);
%run('untitled2.m');

profile off;
profile viewer;
