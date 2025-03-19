function aupr =pr_curve(output,original)
    output=output(:);
    original=original(:);

    [threshold,ind] = sort(output,'descend'); 
    roc_y = original(ind);  

    P=[1:length(roc_y)]';  
    stack_x = cumsum(roc_y == 1)/sum(roc_y == 1);
    stack_y = cumsum(roc_y == 1)./P; 
    aupr=sum((stack_x(2:length(roc_y))-stack_x(1:length(roc_y)-1)).*stack_y(2:length(roc_y)));  


end

