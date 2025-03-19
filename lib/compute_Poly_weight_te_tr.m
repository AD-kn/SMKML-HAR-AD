function W_te_tr = compute_Poly_weight_te_tr(X_te, X_tr, k)
    [nSmp_te, ~] = size(X_te);
    [nSmp_tr, ~] = size(X_tr);
    
    epsilon = 1e-6; 

    distance = zeros(nSmp_te, nSmp_tr);
    for i = 1:nSmp_te
        for j = 1:nSmp_tr
            distance(i, j) = sqrt(sum((X_te(i,:) - X_tr(j,:)) .^ 2));
        end
    end


    [~, dis_index_te_tr] = sort(distance, 2, 'ascend'); 
    knn_index = dis_index_te_tr(:, 2:k+1);  
    W_te_tr = zeros(nSmp_te, nSmp_tr); 

    for i = 1:nSmp_te
        for j = 1:k
            neighbor_idx = knn_index(i, j);
            poly_kernel = (X_te(i,:) * X_tr(neighbor_idx,:)' + 1)^2; 

            W_te_tr(i, neighbor_idx) = 1 / (poly_kernel + epsilon);
           %W_te_tr(i, neighbor_idx) = poly_kernel ;
        end
    end
end



