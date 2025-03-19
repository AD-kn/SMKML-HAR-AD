function [model, res_tr_val_8in1, res_te_8in1, iter] = one_round_train_test_eval(X_tr_m, W_tr_tr, Y_tr, X_te_m, Y_te, X_val_m, Y_val, k1, k2,  iSeed, blockSize,k)

nOutlier_tr = sum(Y_tr == 2); 
nOutlier_te = sum(Y_te == 2); 
nOutlier_val = sum(Y_val == 2); 


[model, iter] = create_M3KNN_model(X_tr_m, Y_tr, W_tr_tr, k1, k2, iSeed,  blockSize,k);



[label_tr, score_tr] = eval_M3KNN_model(X_tr_m, model, nOutlier_tr, blockSize,k);
[label_te, score_te] = eval_M3KNN_model(X_te_m, model, nOutlier_te,  blockSize,k);
[label_val, score_val] = eval_M3KNN_model(X_val_m, model, nOutlier_val,  blockSize,k);


act_Y_val = cat(1, Y_tr, Y_val);
pred_Y_val = [label_tr; label_val];
pred_val_score = [score_tr; score_val];

[accuracy, sensitivity, specificity, precision, recall, f_measure, gmean, AUPRC] = Evaluate_SVOD(act_Y_val, pred_Y_val, pred_val_score, 1);
res_tr_val_8in1 = [accuracy, sensitivity, specificity, precision, recall, f_measure, gmean, AUPRC];

[accuracy, sensitivity, specificity, precision, recall, f_measure, gmean, AUPRC] = Evaluate_SVOD(Y_te, label_te, score_te, 1);
res_te_8in1 = [accuracy, sensitivity, specificity, precision, recall, f_measure, gmean, AUPRC];

end
