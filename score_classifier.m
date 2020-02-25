function pe = score_classifier(data,w)
data_vals = [data.value];
clf;
if length(w) == 3
    z = [ones(1,length(data)); [data.value]];
    decisions = double(w'*z >= 0);
    x2_range = [min(data_vals(2,:))-1 max(data_vals(2,:))+1];                      
    x1_decision = -(w(1) + w(3).*x2_range)./(w(2));
    plot(x1_decision, x2_range,'DisplayName','Decision Boundary');
    hold on;
elseif length(w) == 6
    z = [ones(1,length(data)); [data.value]; data_vals(1,:).^2; data_vals(1,:).*data_vals(2,:); data_vals(2,:).^2];
    decisions = double(w'*z >= 0);
    fcn = @(x1,x2) w(1)+(w(2).*x1)+(w(3).*x2)+(w(4).*x1.^2)+(w(5).*x1.*x2)+(w(6).*x2.^2);
    fp = fimplicit(fcn); fp.DisplayName = 'Decision Boundary';
    hold on;
end
    
idx00 = decisions == 0 & [data.label] == 0;
idx01 = decisions == 0 & [data.label] == 1;
idx10 = decisions == 1 & [data.label] == 0;
idx11 = decisions == 1 & [data.label] == 1;
pe = sum(decisions ~= [data.label])/length(data);
data = [data.value];

% Plot classifier decisions
plot(data(1,idx00),data(2,idx00),'oc','DisplayName','D=0, L=0'); 
plot(data(1,idx10),data(2,idx10),'^r','DisplayName','D=1, L=0');
plot(data(1,idx11),data(2,idx11),'+m','DisplayName','D=1, L=1');
plot(data(1,idx01),data(2,idx01),'sb','DisplayName','D=0, L=1');
hold off;
xlabel('x_1'); ylabel('x_2');
legend('location','northeastoutside');