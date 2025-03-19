clear; clc;

output_filename = fullfile(pwd, 'AUPRC_all.mat'); 

result_files = dir('*_result_all.mat');

num_datasets = length(result_files);
if num_datasets == 0
    error('未找到任何 *_result_all.mat 文件，请检查路径！');
end

auprc_matrix = zeros(num_datasets, 25); 

for i = 1:num_datasets
    disp(['正在处理文件: ', result_files(i).name]);
    
    data = load(result_files(i).name);
    
    if ~isfield(data, 'res')
        warning(['文件 ', result_files(i).name, ' 中未找到变量 res，跳过该文件！']);
        continue;
    end

    res = data.res;  

    if ~isfield(res, 'te')
        warning(['文件 ', result_files(i).name, ' 中未找到变量 te，跳过该文件！']);
        continue;
    end


    te = res.te;
    

    if ndims(te) ~= 3 || size(te,3) < 8
        warning(['文件 ', result_files(i).name, ' 的 te 维度异常，跳过该文件！']);
        continue;
    end
    auprc_values = te(:, :, 8);  

    auprc_matrix(i, :) = reshape(auprc_values', 1, []); 
end

if any(auprc_matrix(:) ~= 0)
    save(output_filename, 'auprc_matrix');
    disp(['AUPRC 结果已成功保存至: ', output_filename]); 
else
    warning('未能成功提取任何 AUPRC 结果，未保存文件。');
end
