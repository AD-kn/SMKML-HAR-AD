function [grad_Vs, grad_bs] = compute_grad(Ks, Us, X, W, Lambda, idx_k1, idx_k2)

nKernel = numel(Ks); 
nSmp = size(Ks{1}, 1); 
k1 = size(idx_k1, 2);
k2 = size(idx_k2, 2);


if ~issparse(X)
    X = sparse(X);
end


X = [X, sparse(ones(nSmp, 1))];

[~, nFea] = size(X);
grad_Vs = sparse(nFea, nKernel);

for iKernel = 1:nKernel
    Ki = Ks{iKernel}; 
    k_r_ii = diag(Ki);
    ui = Us(:, iKernel);
    
    grad_1_p1_all = sparse(1, nFea);
    grad_1_p2_all = sparse(1, nFea);
    grad_1_p3_all = sparse(1, nFea);
    grad_1_p4_all = sparse(1, nFea);
    
    grad_2_p1_all = sparse(1, nFea);
    grad_2_p2_all = sparse(1, nFea);
    grad_2_p3_all = sparse(1, nFea);
    grad_2_p4_all = sparse(1, nFea);
    
    for j1 = 1:k1
        j_idx = idx_k1(:, j1);
        ij_idx = sub2ind([nSmp, nSmp], (1:nSmp)', j_idx);
        k_r_ij = Ki(ij_idx);
        uj = ui(j_idx);
        k_r_jj = k_r_ii(j_idx);
        w_ij = W(ij_idx);
        Lambda_j = Lambda(:, j1);

        p1X = bsxfun(@times, X, w_ij .* ui .* k_r_ii .* ui .* (1 - ui));
        p2X = bsxfun(@times, X, w_ij .* ui .* (1 - ui) .* k_r_ij .* uj);
        p3X = bsxfun(@times, X(j_idx, :), w_ij .* ui .* k_r_ij .* uj .* (1 - uj));
        p4X = bsxfun(@times, X(j_idx, :), w_ij .* uj .* k_r_jj .* uj .* (1 - uj));
        
        grad_1_p1_all = grad_1_p1_all + sum(p1X, 1);
        grad_1_p2_all = grad_1_p2_all + sum(p2X, 1);
        grad_1_p3_all = grad_1_p3_all + sum(p3X, 1);
        grad_1_p4_all = grad_1_p4_all + sum(p4X, 1);
        
        for j2 = 1:k2
            l_idx = idx_k2(j_idx, j2);
            jl_idx = sub2ind([nSmp, nSmp], j_idx, l_idx);
            k_r_jl = Ki(jl_idx);
            ll_idx = sub2ind([nSmp, nSmp], l_idx, l_idx);
            k_r_ll = Ki(ll_idx);
            ul = ui(l_idx);

            p1X = bsxfun(@times, X(j_idx, :), Lambda_j .* uj .* k_r_jj .* uj .* (1 - uj));
            p2X = bsxfun(@times, X(j_idx, :), Lambda_j .* uj .* (1 - uj) .* k_r_jl .* ul);
            p3X = bsxfun(@times, X(l_idx, :), Lambda_j .* uj .* k_r_jl .* ul .* (1 - ul));
            p4X = bsxfun(@times, X(l_idx, :), Lambda_j .* ul .* k_r_ll .* ul .* (1 - ul));

            grad_2_p1_all = grad_2_p1_all + sum(p1X, 1);
            grad_2_p2_all = grad_2_p2_all + sum(p2X, 1);
            grad_2_p3_all = grad_2_p3_all + sum(p3X, 1);
            grad_2_p4_all = grad_2_p4_all + sum(p4X, 1);
        end
    end

    grad_1_all = 2 * grad_1_p1_all - 2 * grad_1_p2_all - 2 * grad_1_p3_all + 2 * grad_1_p4_all;
    grad_2_all = 2 * grad_2_p1_all - 2 * grad_2_p2_all - 2 * grad_2_p3_all + 2 * grad_2_p4_all;
    
    grad_Vs(:, iKernel) = grad_1_all - grad_2_all;
end

grad_bs = grad_Vs(end, :);
grad_Vs(end, :) = [];

end
