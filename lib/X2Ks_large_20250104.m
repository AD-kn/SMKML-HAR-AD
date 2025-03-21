function [Ks] = X2Ks_large_20250104(X, k, blockSize)

[nSmp, ~] = size(X);

nKernel = 12;
Ks = cell(1, nKernel);


if ~exist('blockSize', 'var')
    blockSize = 1000;
end
blockSize = min(blockSize, nSmp); 
numBlocks = ceil(nSmp / blockSize);
rowIdx_s_cell = cell(numBlocks, 1);
colIdx_s_cell = cell(numBlocks, 1);
val_s_cell = cell(numBlocks, 1);
rowIdx_d_cell = cell(numBlocks, 1);
colIdx_d_cell = cell(numBlocks, 1);
val_d_cell = cell(numBlocks, 1);

t_k = 0;
x_norms = sum(X.^2, 2); 
x_min = min(x_norms);
x_max = max(x_norms); 

s = zeros(1, numBlocks); 

for iBlock = 1:numBlocks 
    blockStart = (iBlock - 1) * blockSize + 1; 
    blockEnd = min(iBlock * blockSize, nSmp);
    blockSizeCurr = blockEnd - blockStart + 1; 
    
    t1 = tic;
    s_block = X * X(blockStart:blockEnd, :)';


    if x_min > 0.9999 && x_max < 1.001   
        s(iBlock) = 2 * nSmp *  blockSizeCurr - 2 * sum(sum(s_block));
    else  
        d_block = bsxfun(@plus, -2 * s_block, x_norms);
        d_block = bsxfun(@plus, d_block, x_norms(blockStart:blockEnd)'); 
        s(iBlock) = sum(sum(d_block));
    end
    t_k = t_k + toc(t1); 
    

 
 
    s_block(blockStart:blockEnd, :)  = s_block(blockStart:blockEnd, :) - 10^8 * eye(blockSizeCurr); 

    [K_block_value, Idx] = sort(s_block, 1, 'descend'); 
    Idx = Idx(1:k, :);  
    K_block_value = K_block_value(1:k, :); 

    rowIdx_s_cell{iBlock} = Idx(:);
    colIdx2 = repmat((1:blockSizeCurr), k, 1);  
    colIdx_s_cell{iBlock} = colIdx2(:) + blockStart - 1;
    val_s_cell{iBlock} = K_block_value(:); 
    


    if x_min > 0.9999 && x_max < 1.001
        D_block_value = 2 - 2 * K_block_value;
    else
        d_block(blockStart:blockEnd, :)  = d_block(blockStart:blockEnd, :) + 10^8 * eye(blockSizeCurr); % Subtract diagonal with a scalar efficiently
        [D_block_value, Idx] = sort(d_block, 1, 'ascend');
        Idx = Idx(1:k, :); 
        D_block_value = D_block_value(1:k, :);
    end
    rowIdx_d_cell{iBlock} = Idx(:);
    colIdx2 = repmat((1:blockSizeCurr), k, 1);
    colIdx_d_cell{iBlock} = colIdx2(:) + blockStart - 1; 
    val_d_cell{iBlock} = D_block_value(:); 
end


rowIdx = cell2mat(rowIdx_s_cell);
colIdx = cell2mat(colIdx_s_cell);
val_s = cell2mat(val_s_cell);


Ks{1} = create_K(rowIdx, colIdx, val_s, nSmp, k);
val = (1 + val_s).^2;
Ks{2} = create_K(rowIdx, colIdx, val, nSmp, k);
val = (1 + val_s).^4;
Ks{3} = create_K(rowIdx, colIdx, val, nSmp, k);
val = val_s.^2;
Ks{4} = create_K(rowIdx, colIdx, val, nSmp, k);
val = val_s.^4;
Ks{5} = create_K(rowIdx, colIdx, val, nSmp, k);


rowIdx = cell2mat(rowIdx_d_cell);
colIdx = cell2mat(colIdx_d_cell);
val_d = cell2mat(val_d_cell);
s2 = sum(s)/nSmp^2;

val = exp(-val_d / (2^-3 * s2 ));
Ks{6} = create_K(rowIdx, colIdx, val, nSmp, k);

% Kaussian 2.^-2
val = exp(-val_d / (2^-2 * s2 ));
Ks{7} = create_K(rowIdx, colIdx, val, nSmp, k);


% Kaussian 2.^-1
val = exp(-val_d / (2^-1 * s2 ));
Ks{8} = create_K(rowIdx, colIdx, val, nSmp, k);


% Kaussian 2.^-0
val = exp(-val_d / (2^0 * s2 ));
Ks{9} = create_K(rowIdx, colIdx, val, nSmp, k);

% Kaussian 2.^1
val = exp(-val_d / (2^1 * s2 ));
Ks{10} = create_K(rowIdx, colIdx, val, nSmp, k);

% Kaussian 2.^2
val = exp(-val_d / (2^2 * s2 ));
Ks{11} = create_K(rowIdx, colIdx, val, nSmp, k);

% Kaussian 2.^3
val = exp(-val_d / (2^3 * s2 ));
Ks{12} = create_K(rowIdx, colIdx, val, nSmp, k);
end



function K = create_K(rowIdx, colIdx, val, nSmp, k)
K = sparse(rowIdx, colIdx, val, nSmp, nSmp, nSmp * k);  
K = bsxfun(@rdivide, K, sum(K, 1)+eps);
K = .5 * K + .5 * K';


DSsym = 1 ./ sqrt(sum(K, 1) +eps );
K = bsxfun(@times, K, DSsym);
K = bsxfun(@times, K, DSsym');

K = K + speye(nSmp);
K = .5 * K;
K = .5 * K + .5 * K';
end