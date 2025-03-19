function out = eval_M3KNN_result(X_te, model, nOutlier_te, Ks_trte, k, blockSize)

k1 = model.k1;
k2 = model.k2;
nSmp_te = size(X_te, 1);
nSmp_tr = size(model.X_tr, 1);


Vb = X_te * model.Vs + model.bs;
Us_a = 1 ./ (1 + exp(-Vb));


W_te_tr = compute_Poly_weight_te_tr(X_te, model.X_tr, k);


Ks = X2Ks_large_20250104(X_te, k, blockSize);
nKernel = numel(Ks);

uKs_tete = sparse(nSmp_te, nSmp_te);
for iKernel = 1:nKernel
    uKi_tete = Ks{iKernel} .* Us_a(:, iKernel);  
    uKi_tete = uKi_tete .* Us_a(:, iKernel)';  
    uKs_tete = uKs_tete + 0.5 * (uKi_tete + uKi_tete');
end


uKs_te_tr = sparse(nSmp_te, nSmp_tr);
for iKernel = 1:nKernel
    uKi_te_tr = Ks_trte{iKernel} .* Us_a(:, iKernel);
    uKi_te_tr = uKi_te_tr .* model.Us(:, iKernel)';  
    uKs_te_tr = uKs_te_tr + uKi_te_tr;
end

uK_diag_tete = sum(uKs_tete .* speye(nSmp_te), 2);  
uK_diag_trtr = sum(model.uKs .* speye(nSmp_tr), 2); 

distUK_a_tr = uK_diag_tete + uK_diag_trtr' - 2 * uKs_te_tr; 
distUK_a_tr(1:nSmp_te+1:end) = Inf; 


[~, idx_n_a_tr] = sort(distUK_a_tr, 2, 'ascend');
idx_k1_a_tr = idx_n_a_tr(:, 1:k1);


Lambda = sparse(nSmp_te, k1);
idx_k2_tr_tr = model.idx_k2;

for j1 = 1:k1
    j_idx = idx_k1_a_tr(:, j1);
    ij_idx = sub2ind([nSmp_te, nSmp_tr], (1:nSmp_te)', j_idx);
    
    w_ij = W_te_tr(ij_idx);  
    f_j = distUK_a_tr(ij_idx);  
    g_jl = distUK_a_tr(sub2ind([nSmp_te, nSmp_tr], repmat((1:nSmp_te)', 1, k2), idx_k2_tr_tr(j_idx, :)));
    
    Lambda(:, j1) = w_ij .* f_j ./ sum(g_jl, 2);% case 2
    %Lambda(:, j1) =  f_j ./ sum(g_jl, 2); % case1
end


outlier_score = sum(Lambda, 2);
[~, idx] = sort(outlier_score, 'descend');
predictedlabel = ones(nSmp_te, 1);
predictedlabel(idx(1:nOutlier_te)) = 2;

out = struct();
out.outlier_score = outlier_score;
out.predictedlabel = predictedlabel;
end
