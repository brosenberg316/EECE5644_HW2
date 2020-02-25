function cost = quadratic_cost_function(w,z,labels,N)
cost = 0;
for ii = 1:N
    zxi = [1 z(:,ii)' z(1,ii)^2 z(1,ii)*z(2,ii) z(2,ii)^2]';
    cost = cost + -w'*zxi.*(1-labels(ii))-log(1+exp(-w'*zxi));
end
cost = -1*cost;