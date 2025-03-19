function [ind_pos_all, ind_neg_all] = generate_nRepeat_nFold_Split(X_in, X_o, nRepeat, nFold)
nInlier = size(X_in, 1);
nOutlier = size(X_o, 1);


if ~exist('nRepeat', 'var')
    nRepeat = 5;
end

if ~exist('nFold', 'var')
    nFold = 5;
end

ind_pos_all = zeros(nInlier, nRepeat);
ind_neg_all = zeros(nOutlier, nRepeat);


seed = 2024;
%rng(seed); %过时
rng(seed,'twister');
random_seeds = randi([0, 10000], 1, nRepeat);
original_rng_state = rng;

for iRepeat = 1:nRepeat
    rng(original_rng_state);
    rng(random_seeds(iRepeat));
    ind_pos_all(:, iRepeat) = crossvalind('kfold', nInlier, nFold);
    ind_neg_all(:, iRepeat) = crossvalind('kfold', nOutlier, nFold);
end
end