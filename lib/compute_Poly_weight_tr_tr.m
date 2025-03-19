function W = compute_Poly_weight_tr_tr(X, k)
    [nSmp, ~] = size(X);
    distance = zeros(nSmp, nSmp);
    for i = 1:nSmp
        for j = 1:nSmp
            distance(i, j) = sqrt(sum((X(i,:) - X(j,:)) .^ 2));
        end
    end

    distance = distance + 1e8 * eye(nSmp); 
    [~, dis_index] = sort(distance, 2, 'ascend'); 
    knn_index = dis_index(:, 2:k+1); 
   
    W = ones(nSmp, nSmp); 

    for i = 1:nSmp
        for j = 1:k
            neighbor_idx = knn_index(i, j);
            W(i, neighbor_idx) = (X(i,:) * X(neighbor_idx,:)' + 1)^2; % **PolyPlus 核**
        end
    end
end
