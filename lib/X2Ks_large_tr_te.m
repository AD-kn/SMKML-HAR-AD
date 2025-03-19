function [Ks] = X2Ks_large_tr_te(x_tr, x_te, k, blockSize)

[n_tr, d] = size(x_tr);
n_te = size(x_te, 1);
nKernel = 12;
Ks = cell(1, nKernel);

if ~exist('blockSize', 'var')
    blockSize = 1000;
end
blockSize = min(blockSize, n_te);

numBlocks = ceil(n_te / blockSize);
t_k = 0;

% 计算范数
x_tr_norms = sum(x_tr.^2, 2);
x_te_norms = sum(x_te.^2, 2);
x_min = min([x_tr_norms; x_te_norms]);
x_max = max([x_tr_norms; x_te_norms]);

% 初始化存储变量
rowIdx_s_cell = cell(numBlocks, 1);
colIdx_s_cell = cell(numBlocks, 1);
val_s_cell = cell(numBlocks, 1);
rowIdx_d_cell = cell(numBlocks, 1);
colIdx_d_cell = cell(numBlocks, 1);
val_d_cell = cell(numBlocks, 1);
s = zeros(1, numBlocks);

for iBlock = 1:numBlocks
    blockStart = (iBlock - 1) * blockSize + 1;
    blockEnd = min(iBlock * blockSize, n_te);
    blockSizeCurr = blockEnd - blockStart + 1;

    % 计算 x_tr 和 x_te 当前块之间的相似度
    t1 = tic;
    s_block = x_tr * x_te(blockStart:blockEnd, :)';

    % 计算欧式距离
    if x_min > 0.9999 && x_max < 1.001
        s(iBlock) = 2 * n_tr * blockSizeCurr - 2 * sum(sum(s_block));
    else
        d_block = bsxfun(@plus, -2 * s_block, x_tr_norms);
        d_block = bsxfun(@plus, d_block, x_te_norms(blockStart:blockEnd)');
        s(iBlock) = sum(sum(d_block));
    end
    t_k = t_k + toc(t1);

    % 计算最近邻索引
    [K_block_value, Idx] = sort(s_block, 1, 'descend');
    Idx = Idx(1:k, :);
    K_block_value = K_block_value(1:k, :);

    rowIdx_s_cell{iBlock} = Idx(:);
    colIdx2 = repmat((1:blockSizeCurr), k, 1);
    colIdx_s_cell{iBlock} = colIdx2(:) + blockStart - 1;
    val_s_cell{iBlock} = K_block_value(:);

    % 计算距离矩阵
    if x_min > 0.9999 && x_max < 1.001
        D_block_value = 2 - 2 * K_block_value;
    else
        d_block(blockStart:blockEnd, :) = d_block(blockStart:blockEnd, :) + 10^8 * eye(blockSizeCurr);
        [D_block_value, Idx] = sort(d_block, 1, 'ascend');
        Idx = Idx(1:k, :);
        D_block_value = D_block_value(1:k, :);
    end

    rowIdx_d_cell{iBlock} = Idx(:);
    colIdx_d_cell{iBlock} = colIdx2(:) + blockStart - 1;
    val_d_cell{iBlock} = D_block_value(:);
end

% 合并索引和值
rowIdx = cell2mat(rowIdx_s_cell);
colIdx = cell2mat(colIdx_s_cell);
val_s = cell2mat(val_s_cell);

% 计算线性核
Ks{1} = create_K(rowIdx, colIdx, val_s, n_tr, n_te, k);

% 计算 PolyPlus 核
val = (1 + val_s).^2;
Ks{2} = create_K(rowIdx, colIdx, val, n_tr, n_te, k);
val = (1 + val_s).^4;
Ks{3} = create_K(rowIdx, colIdx, val, n_tr, n_te, k);

% 计算多项式核
val = val_s.^2;
Ks{4} = create_K(rowIdx, colIdx, val, n_tr, n_te, k);
val = val_s.^4;
Ks{5} = create_K(rowIdx, colIdx, val, n_tr, n_te, k);

% 处理距离矩阵
rowIdx = cell2mat(rowIdx_d_cell);
colIdx = cell2mat(colIdx_d_cell);
val_d = cell2mat(val_d_cell);

% 计算高斯核
s2 = sum(s) / (n_tr * n_te);

for i = 6:12
    sigma = 2^(i - 9);
    val = exp(-val_d / (sigma * s2));
    Ks{i} = create_K(rowIdx, colIdx, val, n_tr, n_te, k);
end

end

% 创建核矩阵的辅助函数
function K = create_K(rowIdx, colIdx, val, n_tr, n_te, k)
K = sparse(rowIdx, colIdx, val, n_tr, n_te, n_tr * k);
K = bsxfun(@rdivide, K, sum(K, 1) + eps);


DSsym_r = 1 ./ sqrt(sum(K, 1) + eps);
DSsym_c = 1 ./ sqrt(sum(K, 2) + eps);
K = K .* DSsym_r; % 按列归一化
K = K .* DSsym_c; % 按行归一化


end
