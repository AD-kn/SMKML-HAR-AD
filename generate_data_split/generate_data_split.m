clear;
clc;
data_path = fullfile(pwd, '..', filesep, 'data', filesep);
addpath(data_path);
lib_path = fullfile(pwd, '..', filesep, 'lib', filesep);
addpath(lib_path);

%*******************************************
% Experiment Configuration
%*******************************************
nRepeat = 5;
nFold = 5;
inlier_class_id = 0; 
update_data_labels();
datasetCandi = dir(fullfile(data_path, '*.mat')); 
for iDataSet =1:length(datasetCandi)
    data_name = datasetCandi(iDataSet).name;
    toy_data = load(fullfile(data_path, data_name)); 
    
 
    Y = toy_data.y;
    X = double(toy_data.X); 
 
    save(fullfile(data_path, data_name), 'X', 'Y'); 
    X = zscore(X);
    
    
    % Define the directory for saving the data
    save_dir = fullfile(pwd, '..', 'data_split', data_name(1:end-4)); 
    if ~exist(save_dir, 'dir') 
        mkdir(save_dir);
    end

    %*******************************************
    % Build Inlier and Outlier
    %*******************************************
    [Xy_in, Xy_o] = generate_Inlier_Outlier(X, Y, inlier_class_id);
    [ind_pos_all, ind_neg_all] = generate_nRepeat_nFold_Split(Xy_in, Xy_o, nRepeat, nFold);
    
    for iRepeat = 1:nRepeat
        ind_pos = ind_pos_all(:, iRepeat);
        ind_neg = ind_neg_all(:, iRepeat);
        
        for iFold = 1:nFold
            % Prepare train, validation, and test sets
            [X_tr_m, Y_tr, X_te_m, Y_te, X_val_m, Y_val] = prepare_train_val_test(Xy_in, Xy_o, ind_pos, ind_neg, iFold);

            % Save the processed data to a .mat file
            save_name = sprintf('pos_%d_repeat_%d_fold_%d.mat', 1, iRepeat, iFold);
            save(fullfile(save_dir, save_name), 'X_tr_m', 'X_te_m', 'X_val_m', 'Y_tr', 'Y_te', 'Y_val');
        end
    end
end
rmpath(data_path);
rmpath(lib_path);