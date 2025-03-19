function [Xy_in, Xy_o] = generate_Inlier_Outlier(X, Ys, inlier_class_id)
Y = Ys;
Y( Ys ~= inlier_class_id) = 2;
Y( Ys == inlier_class_id) = 1;
Xy_in = X(Y==1,:);
Xy_in = cat(2, Xy_in, ones(size(Xy_in,1),1));
Xy_o = X(Y==2,:);
Xy_o = cat(2, Xy_o, ones(size(Xy_o,1),1).*2);
end