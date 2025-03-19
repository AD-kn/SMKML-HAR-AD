function [X_tr, Y_tr, X_te, Y_te, X_val, Y_val] = prepare_train_val_test(Xy_in, Xy_o, idx_inlier, idx_outlier, id_fold_test)


test_inlier_ind = (idx_inlier == id_fold_test);
train_inlier_ind =~ test_inlier_ind;
test_outlier_ind = (idx_outlier == id_fold_test);
train_outlier_ind =~ test_outlier_ind;

test_inlier = Xy_in(test_inlier_ind,:);
train_inlier = Xy_in(train_inlier_ind,:);
test_outlier = Xy_o(test_outlier_ind,:);
train_outlier = Xy_o(train_outlier_ind,:);

X_tr = train_inlier(:,1:end-1);
Y_tr = train_inlier(:,end);
X_te = cat(1,test_inlier(:,1:end-1), test_outlier(:,1:end-1));
Y_te = cat(1,test_inlier(:,end), test_outlier(:,end));
X_val = train_outlier(:,1:end-1);
Y_val = train_outlier(:,end);