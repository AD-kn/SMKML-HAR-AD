function [label, score] = eval_M3KNN_model(X_te, model, nOutlier_a, blockSize,k)

X_tr = model.X_tr;

Ks = X2Ks_large_tr_te(X_tr, X_te, k, blockSize);

nKernel = numel(Ks);
Ks_trte = cell(nKernel, 1); 

for i = 1:nKernel
    Ks_trte{i} = Ks{i}';
end

output = eval_M3KNN_result(X_te, model, nOutlier_a, Ks_trte,  k, blockSize);

label = output.predictedlabel;
score = output.outlier_score;

end
