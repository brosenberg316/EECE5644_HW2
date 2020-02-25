function cost = linear_cost_function(w,z,labels,N)
cost = 0;
for ii = 1:N
    zxi = [1 z(:,ii)']';
    cost = cost + -w'*zxi.*(1-labels(ii))-log(1+exp(-w'*zxi));
end
cost = -1*cost;