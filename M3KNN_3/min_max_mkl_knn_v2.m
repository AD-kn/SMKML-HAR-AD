function [Vs, bs, objHistory, Lambda, idx_k1, idx_k2, uKs, Us, iter] = min_max_mkl_knn_v2(X, Ks, W, k1, k2, iSeed)

nKernel = numel(Ks);
nSmp = size(Ks{1}, 1); 
nFea = size(X, 2); 
rng('default'); 


if nargin == 6
    rng(iSeed); 
else
    rng(2024); 
end


Vs = randn(nFea, nKernel);
bs = randn(1, nKernel);


[obj, Lambda, Us, idx_k1, idx_k2, uKs] = compute_obj(Ks, X, W, Vs, bs, k1, k2);
objHistory = obj;


eta = 10; 
min_eta = 1e-6; 
tol = 1e-6; 
alpha = 0.3; 
beta = 0.4; 
max_iter = 100;
iter = 0;


best_obj = obj;
best_Vs = Vs;
best_bs = bs;

while iter < max_iter
    
    [grad_Vs, grad_bs] = compute_grad(Ks, Us, X, W, Lambda, idx_k1, idx_k2);
    grad_Vs = grad_Vs / (norm(grad_Vs(:)) + 1e-8);
    grad_bs = grad_bs / (norm(grad_bs(:)) + 1e-8);

 
    c = alpha * (dot(grad_Vs(:), grad_Vs(:)) + dot(grad_bs(:), grad_bs(:)));

   
    while true
        Vs_new = Vs - eta * grad_Vs;
        bs_new = bs - eta * grad_bs;
        [new_obj_value, Lambda, Us, idx_k1, idx_k2, uKs] = compute_obj(Ks, X, W, Vs_new, bs_new, k1, k2);
        if new_obj_value <= obj - eta * c
            if new_obj_value < best_obj
                best_obj = new_obj_value;
                best_Vs = Vs_new;
                best_bs = bs_new;
            end
            break;
        else
            eta = beta * eta; 
            if eta < min_eta
                warning('Step size eta too small, stopping line search');
                break;
            end
        end
    end


    Vs = Vs_new;
    bs = bs_new;
    obj = new_obj_value;
  
    objHistory = [objHistory; obj];
    iter = iter + 1;
    
    if iter > 5 && obj > 1.05 * min(objHistory)
        warning('Detected large obj increase, stopping early.');
        break;
    end

    if mod(iter, 10) == 0
        eta = min(eta * 1.2, 10);
    end

  
    if norm(grad_Vs(:)) < tol && norm(grad_bs(:)) < tol
        break;
    end
    
    if iter > 2 && abs((objHistory(end-1) - objHistory(end)) / objHistory(end-1)) < 1e-6
        break;
    end
end


if obj > best_obj
    obj = best_obj;
    Vs = best_Vs;
    bs = best_bs;
end

end

function [obj, Lambda, Us, idx_k1, idx_k2, uKs] = compute_obj(Ks, X, W, Vs, bs, k1, k2)
nKernel = numel(Ks); 
nSmp = size(Ks{1}, 1); 

Us = 1 ./ (1 + exp(- (X * Vs + bs))); % nSmp * nKernel


uKs = sparse(nSmp, nSmp);

for iKernel = 1:nKernel
    uKi = Ks{iKernel} .* Us(:, iKernel);  
    uKi = uKi .* Us(:, iKernel)';  
    uKs = uKs + uKi; % 累加
end


uK_diag = full(sum(uKs .* speye(nSmp), 2)); 
distUK = spdiags(uK_diag, 0, nSmp, nSmp) + spdiags(uK_diag, 0, nSmp, nSmp)' - 2 * uKs;
distUK = distUK + spdiags(1e8 * ones(nSmp,1), 0, nSmp, nSmp);


[~, idx_n] = sort(distUK, 1, 'ascend');
idx_k1 = idx_n(1:k1, :)';
idx_k2 = idx_n(1:k2, :)';

Lambda = zeros(nSmp, k1);
for j1 = 1:k1
    j_idx = idx_k1(:, j1);
    ij_idx = sub2ind([nSmp, nSmp], (1:nSmp)', j_idx);
    w_ij = W(ij_idx); 
    f_j = distUK(ij_idx); 
    
    g_jl = distUK(sub2ind([nSmp, nSmp], repmat(j_idx, 1, k2), idx_k2(j_idx, :)));

    Lambda(:, j1) = w_ij .* f_j ./ sum(g_jl, 2);   
end

obj = sum(Lambda(:));
end
