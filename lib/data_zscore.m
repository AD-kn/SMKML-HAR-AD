function [X_tr_m, X_te_m, X_val_m] = data_zscore(X_tr, X_te, X_val)
norm_tr = mean_and_std(X_tr, 'true');
X_tr_m = normalize_data(X_tr, norm_tr);
norm_te = mean_and_std(X_te,'true');
X_te_m = normalize_data(X_te, norm_te);
norm_val = mean_and_std(X_val, 'true');
X_val_m = normalize_data(X_val, norm_val);
end