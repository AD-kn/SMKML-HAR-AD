function [model_m3knn,iter] = create_M3KNN_model(X_tr, Y_tr, W, k1, k2, iSeed,blockSize,k)
Ks = X2Ks_large_20250104(X_tr, k, blockSize);

nKernel = numel(Ks);
Ks_trtr = cell(nKernel, 1); 

for i = 1:nKernel
    Ks_trtr{i} = Ks{i}';
end
model_m3knn.k1 = k1;
model_m3knn.k2 = k2;
model_m3knn.X_tr = X_tr;
model_m3knn.Y_tr = Y_tr;


[Vs, bs, objHistory, Lambda, idx_k1, idx_k2, uKs, Us, iter] = min_max_mkl_knn_v2(X_tr, Ks_trtr, W, k1, k2, iSeed);



model_m3knn.Vs = Vs;
model_m3knn.bs = bs;
model_m3knn.objHistory = objHistory;
model_m3knn.Lambda = Lambda;
model_m3knn.idx_k1 = idx_k1;
model_m3knn.idx_k2 = idx_k2;
model_m3knn.uKs = uKs;
model_m3knn.Us = Us;
model_m3knn.gat = [Vs; bs]';

model_m3knn.nKernel = nKernel;

end