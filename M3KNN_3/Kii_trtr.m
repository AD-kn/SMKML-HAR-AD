function Kii = Kii_trtr(idx,Ks_tr,opts)


[nSmp, ~, nKernel] = size(Ks_tr);

avgK = sum(Ks_tr, 3) - 10^8*eye(nSmp);
[~, Idx] = sort(avgK, 1, 'descend'); 
Idx = Idx(1:opts.k, :);
Idx2 = Idx';

Ik = opts.lambda * eye(opts.k); 


Idxs = zeros(opts.k, nSmp, nKernel); 
for i1 = 1:nKernel
    Idxs(:, :, i1) = Idx; 
end


for i1 = 1:nKernel 
    Ki = Ks_tr(:, :, i1);
    for iSmp = 1:nSmp     
        Kii = Ki(idx, idx') + Ik;
    end
end

end