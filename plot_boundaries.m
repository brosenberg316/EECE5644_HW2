function plot_boundaries(data,w)
clf;
z = [data.value];
x2_range = [min(z(2,:))-1 max(z(2,:))+1];
if length(w) == 3
    x1_decision = -(w(1) + w(3).*x2_range)./(w(2));
    plot(x1_decision, x2_range,'DisplayName','Decision Boundary'); 
    hold on;
elseif length(w) == 6
    fcn = @(x1,x2) w(1)+(w(2).*x1)+(w(3).*x2)+(w(4).*x1.^2)+(w(5).*x1.*x2)+(w(6).*x2.^2);
    fp = fimplicit(fcn,[min(z(1,:))-1 max(z(1,:))+1 min(z(2,:))-1 max(z(2,:))+1]); fp.DisplayName = 'Decision Boundary';
    hold on;
end
samples0 = z(:,[data.label] == 0);
samples1 = z(:,[data.label] == 1);
plot(samples0(1,:),samples0(2,:),'oc','DisplayName','L=0'); 
plot(samples1(1,:),samples1(2,:),'+m','DisplayName','L=1'); 
hold off; 
legend('location','northeastoutside');
xlabel('x_1'); ylabel('x_2');

