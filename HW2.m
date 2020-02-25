%% Benjamin Rosenberg    EECE 5644    Take Home Exam 2
%% Question 1

% Create both classes as structs
class0.label = 0;
class0.prior = 0.9;
class0.mu = [-2;0];
class0.sigma = [1 -0.9;-0.9 2];

class1.label = 1;
class1.prior = 0.1;
class1.mu = [2;0];
class1.sigma = [2 0.9;0.9 1];

d10 = generate_samples(10,class0,class1);
d100 = generate_samples(100,class0,class1);
d1000 = generate_samples(1000,class0,class1);
d10K = generate_samples(10000,class0,class1);

% plot_samples(d10);
% plot_samples(d100);
% plot_samples(d1000);
%% Part 1 - MAP Classifer
% For two-label classification, in order to minimize expected risk, the classifier 
% is:
% 
% $$\frac{p(\mathbf{x}|L=1)}{p(\mathbf{x}|L=0)}{>}\frac{(\lambda_{10}-\lambda_{00})}{(\lambda_{01}-\lambda_{11})}\frac{P(L=0)}{P(L=1)}\Longrightarrow 
% D=1$ or $$\frac{p(\mathbf{x}|L=1)}{p(\mathbf{x}|L=0)}{>}\gamma$, where $\gamma 
% = \frac{(\lambda_{10}-\lambda_{00})}{(\lambda_{01}-\lambda_{11})}\frac{P(L=0)}{P(L=1)}$
% 
% Using a 0-1 loss function, we can create a MAP classifier that minimizes P(error):
% 
% $\lambda_{10}=\lambda_{01}=1, \ \lambda_{00}=\lambda_{11}=0$, and
% 
% $$\frac{p(\mathbf{x}|L=1)}{p(\mathbf{x}|L=0)}{>}\frac{P(L=0)}{P(L=1)}\Longrightarrow 
% D=1$ or $$\frac{p(\mathbf{x}|L=1)}{p(\mathbf{x}|L=0)}{>}\gamma$, where $\gamma 
% = \frac{P(L=0)}{P(L=1)}=\frac{0.9}{0.1}=9$
%% 
% Minimum probability of error classifier implementation on $D^{10K}_{validate}$:

likelihood0 = evalGaussian([d10K.value],class0.mu,class0.sigma);
likelihood1 = evalGaussian([d10K.value],class1.mu,class1.sigma);
llrt = log(likelihood1) - log(likelihood0);

% Min P(error) classifier => 0-1 Loss => MAP classifier
l10 = 1; l01 = 1; l00 = 0; l11 = 0;
map_gamma = ((l10 - l00)/(l01 - l11))*(class0.prior/class1.prior);

% Iterate through gamma for ROC curve
gamma = 0:0.01:20;
pcd = zeros(1,length(gamma));
pfa = zeros(1,length(gamma));
pe = zeros(1,length(gamma));
for ii = 1:length(gamma)
    decision = double(llrt >= log(gamma(ii)));
    pcd(ii) = sum(decision == 1 & [d10K.label] == 1)/sum([d10K.label] == 1);
    pfa(ii) = sum(decision == 1 & [d10K.label] == 0)/sum([d10K.label] == 0);
    pe(ii) = sum(decision ~= [d10K.label])/length(d10K);
end
% Get the minimum Pe
[val,idx] = min(pe);
map_idx = find(gamma == map_gamma);
% ROC Curve Plot
plot(pfa,pcd); hold on;
xlabel('P_{fa}'); ylabel('P_{cd}'); title({'Approximate ROC Curve','Minimum P(error) Classifier'});
plot(pfa(idx),pcd(idx), 'm*');
plot(pfa(map_idx),pcd(map_idx), 'go');
hold off;
legend('ROC Curve','ROC at Minimum P_e (Estimated)','ROC at Minimum P_e (MAP)');
%% 
% Threshold value for minimum probability of error, and the respective probability 
% of error at that threshold (using $D^{10K}_{validate}$) :

fprintf(['Minimum probability of error: %f\n',...
    'Gamma (threshold) value: %f\nProbability of false alarm: %f \n',...
    'Probability of correct detection: %f'], val,gamma(idx),pfa(idx),pcd(idx));
%% 
% Decision boundary overlaid on the $D^{10K}_{validate}$ dataset:

decision = double(llrt >= log(map_gamma));
idx00 = decision == 0 & [d10K.label] == 0;
idx01 = decision == 0 & [d10K.label] == 1;
idx10 = decision == 1 & [d10K.label] == 0;
idx11 = decision == 1 & [d10K.label] == 1;
d10kvals = [d10K.value];
% Plot classifier decisions
clf; figure; hold on;
plot(d10kvals(1,idx00),d10kvals(2,idx00),'oc');
plot(d10kvals(1,idx10),d10kvals(2,idx10),'^r');
plot(d10kvals(1,idx11),d10kvals(2,idx11),'+m');
plot(d10kvals(1,idx01),d10kvals(2,idx01),'sb');
xlim([-7 7]); ylim([-7 7]);
xlabel('x_1'); ylabel('x_2');
xlabel('x_1');ylabel('x_2');title('Samples Classified by Minimum Error Rate Classifier');

% Add decision boundary
h_grid = linspace(min(d10kvals(1,:)),max(d10kvals(1,:)),100);
v_grid = linspace(min(d10kvals(2,:)),max(d10kvals(2,:)),100);
[h,v] = meshgrid(h_grid,v_grid);
class0_values = class0.prior*evalGaussian([h(:)';v(:)'],class0.mu,class0.sigma);
class1_values = class1.prior*evalGaussian([h(:)';v(:)'],class1.mu,class1.sigma);
discriminant_scores = log(class1_values) - log(class0_values);
discriminant_scores = reshape(discriminant_scores,100,100);
contour(h_grid,v_grid,discriminant_scores,[0 0],'k');
legend('D=0, L=0','D=1, L=0','D=1, L=1','D=0, L=1','Decision Boundary','location','northwest');
hold off;
%% Part 2 - Logistic-Linear
% Given the logistic-linear function $h(\mathbf{x},\mathbf{w})=\frac{1}{1+exp(-\mathbf{w}^T\mathbf{z}(\mathbf{x}))}$ 
% and $\mathbf{z}(\mathbf{x})=[1, \mathbf{x}]^T$, let the class label posterior 
% for L=1 be approximated by the logistic-linear function. Then,
% 
% $p(L=1|\mathbf{x})\approx h(\mathbf{x},\mathbf{w})=\frac{1}{1+exp(-\mathbf{w}^T\mathbf{z}(\mathbf{x}))}$, 
% and $p(L=0|\mathbf{x})=1-p(L=1|\mathbf{x})=1-h(\mathbf{x},\mathbf{w}) \approx\frac{exp(-\mathbf{w}^T\mathbf{z}(\mathbf{x}))}{1+exp(-\mathbf{w}^T\mathbf{z}(\mathbf{x}))}$.
% 
% Datasets $\mathbf{x}$ have an associated labels vector $\mathbf{l}$. Thus, 
% the likelihood function can be written as $p(\mathbf{x},\mathbf{l}|\mathbf{w})=\prod_{i=1}^{N}p(x_i,l_i|\mathbf{w})$ 
% due to the i.i.d nature of the samples, where $N$ is the number of samples. 
% Using Bayes' rule, $\prod_{i=1}^{N}p(x_i,l_i|\mathbf{w})=\prod_{i=1}^{N}p(l_i|x_i,\mathbf{w})p(l_i|\mathbf{w})$. 
% The class-label priors are independent of model parameters, so  $\prod_{i=1}^{N}p(l_i|x_i,\mathbf{w})p(l_i|\mathbf{w}) 
% = \prod_{i=1}^{N}p(l_i|x_i,\mathbf{w})p(l_i)$. Taking the negative natural log 
% to find a negative-log-likelihood, $-ln(\prod_{i=1}^{N}p(l_i|x_i,\mathbf{w})p(l_i) 
% ) =-\sum_{i=1}^Nln(p(l_i|x_i,\mathbf{w})p(l_i) )$. 
% 
% For maximum likelihood parameter estimation, $\mathbf{w}_{ML}=\underset{\mathbf{w}}{argmin}(-ln(p(\mathbf{x},\mathbf{l}|\mathbf{w}))=\underset{\mathbf{w}}{argmin}(-\sum_{i=1}^Nln(p(l_i|\mathbf{x_i},\mathbf{w})p(l_i) 
% ))=\underset{\mathbf{w}}{argmin}(-\sum_{i=1}^Nln(p(l_i|\mathbf{x_i},\mathbf{w}) 
% ))$. The labels $l_i$ can either be 0 or 1, so $p(l_i|x_i,\mathbf{w})$can be 
% modeled with a Bernoulli pmf: $p(l_i|\mathbf{x_i},\mathbf{w})=p^{l_i}(1-p)^{(1-l_i)}$, 
% where $p=p(L=1|\mathbf{x_i,}\mathbf{w})$. Therefore, $\mathbf{w}_{ML}=\underset{\mathbf{w}}{argmin}\Big\{-\sum_{i=1}^Nln(p^{l_i}(1-p)^{(1-l_i)}\Big\}=\underset{\mathbf{w}}{argmin}\Big\{-\sum_{i=1}^N((l_i)ln(p)+(1-l_i)ln(1-p))\Big\}$.
% 
% Using the logistic-linear approximations above, $p=p(L=1|\mathbf{x_i},\mathbf{w}) 
% \approx h(\mathbf{x_i},\mathbf{w})=\frac{1}{1+exp(-\mathbf{w}^T\mathbf{z}(\mathbf{x_i}))}$ 
% and $1-p=p(L=0|x_i,\mathbf{w}) \approx 1-h(x_i,\mathbf{w})=\frac{exp(-\mathbf{w}^T\mathbf{z}(\mathbf{x_i}))}{1+exp(-\mathbf{w}^T\mathbf{z}(\mathbf{x_i}))}$.
% 
% So, $ln(p) = -ln(1+exp(-\mathbf{w}^T\mathbf{z}(\mathbf{x_i}))$ and $ln(1-p) 
% = -\mathbf{w}^T\mathbf{z}(\mathbf{x_i}) -ln(1+exp(-\mathbf{w}^T\mathbf{z}(\mathbf{x_i}))$, 
% and 
% 
% $\mathbf{w}_{ML} = \underset{\mathbf{w}}{argmin}\Big\{-\sum_{i=1}^{N}\{-(l_i)ln(1+exp(-\mathbf{w}^T\mathbf{z}(\mathbf{x_i}))+(1-l_i)( 
% -\mathbf{w}^T\mathbf{z}(\mathbf{x_i})-ln(1+exp(-\mathbf{w}^T\mathbf{z}(\mathbf{x_i})))\}\Big\}$, 
% which simplifies to  $\mathbf{w}_{ML} = \underset{\mathbf{w}}{argmin}\Big\{-\sum_{i=1}^{N}\{-(1-l_i)\mathbf{w}^T\mathbf{z}(\mathbf{x_i})-ln(1+exp(-\mathbf{w}^T\mathbf{z}(\mathbf{x_i}))\}\Big\}$.
% 
% 
% 
% To classify a sample, the decision boundary occurs when $p(L=1|\mathbf{x}) 
% = p(L=0|\mathbf{x})$. Using our logistic-linear approximations, this is $h(\mathbf{x},\mathbf{w})=1-h(\mathbf{x},\mathbf{w})$, 
% or $\frac{1}{1+exp(-\mathbf{w}^T\mathbf{z}(\mathbf{x}))}= \frac{exp(-\mathbf{w}^T\mathbf{z}(\mathbf{x}))}{1+exp(-\mathbf{w}^T\mathbf{z}(\mathbf{x}))}$, 
% so $exp(\mathbf{-w}^T\mathbf{z}(\mathbf{x}))=1 \Longrightarrow -\mathbf{w}^T\mathbf{z}(\mathbf{x}) 
% = 0$. Therefore, the decision boundary is $\mathbf{w}^T\mathbf{z}(\mathbf{x}) 
% \geq 0 \Longrightarrow D = 1$
% D10 Classifier

init_w = zeros(3,1);
% Train using D10 dataset
z = [d10.value];
labels = [d10.label];
N = length(d10);
[w10_lin, cost10] = fminsearch(@(w)(linear_cost_function(w,z,labels,N)), init_w);
plot_boundaries(d10,w10_lin);
title('Decision Boundary of D_{10} Classifier');
% Find P(error) for D10 dataset classifer
pe10_lin = score_classifier(d10K,w10_lin);
title('D_{10K} Samples Classified by D_{10} Training Set Classifier');
fprintf('P(error) for D10 training set classifier on D10K validation set: %f', pe10_lin);
% D100 Classifier

% Train using D100 dataset
z = [d100.value];
labels = [d100.label];
N = length(d100);
[w100_lin, cost100] = fminsearch(@(w)(linear_cost_function(w,z,labels,N)), init_w);
plot_boundaries(d100,w100_lin);
title('Decision Boundary of D_{100} Classifier');
pe100_lin = score_classifier(d10K,w100_lin);
title('D_{10K} Samples Classified by D_{100} Training Set Classifier');
fprintf('P(error) for D100 training set classifier on D10K validation set: %f', pe100_lin);
% D1000 Classifer

% Train using D1000 dataset
z = [d1000.value];
labels = [d1000.label];
N = length(d1000);
[w1000_lin, cost1000] = fminsearch(@(w)(linear_cost_function(w,z,labels,N)), init_w);
% Draw boundary line for D1000 dataset
plot_boundaries(d1000,w1000_lin);
title('Decision Boundary of D_{1000} Classifier');
pe1000_lin = score_classifier(d10K,w1000_lin);
title('D_{10K} Samples Classified by D_{1000} Training Set Classifier');
fprintf('P(error) for D1000 training set classifier on D10K validation set: %f', pe1000_lin);
%% Part 3 - Logistic-Quadratic
% The logistic-quadratic-function-based approximation of class-label posteriors 
% has the same maximum-likelihood parameter estimate as the logistic-linear case:
% 
% $\mathbf{w}_{ML} = \underset{\mathbf{w}}{argmin}\Big\{-\sum_{i=1}^{N}\{-(1-l_i)\mathbf{w}^T\mathbf{z}(\mathbf{x_i})-ln(1+exp(-\mathbf{w}^T\mathbf{z}(\mathbf{x_i}))\}\Big\}$, 
% but now $\mathbf{z}(\mathbf{x})=[1,x_1,x_2,x_1^2,x_1x_2,x_2^2]^T$
% D10 Classifier

init_w = [w1000_lin; zeros(3,1)];
% Train using D10 dataset
z = [d10.value];
labels = [d10.label];
N = length(d10);
[w10_quad, ~] = fminsearch(@(w)(quadratic_cost_function(w,z,labels,N)), init_w);
plot_boundaries(d10,w10_quad);
title('Decision Boundary of D_{10} Classifier');
% Find P(error) for D10 dataset classifer
pe10_quad = score_classifier(d10K,w10_quad);
title('D_{10K} Samples Classified by D_{10} Training Set Classifier');
fprintf('P(error) for D10 training set classifier on D10K validation set: %f', pe10_quad);
% D100 Classifier

% Train using D100 dataset
z = [d100.value];
labels = [d100.label];
N = length(d100);
[w100_quad, ~] = fminsearch(@(w)(quadratic_cost_function(w,z,labels,N)), init_w);
plot_boundaries(d100,w100_quad);
title('Decision Boundary of D_{100} Classifier');
% Find P(error) for D100 dataset classifer
pe100_quad = score_classifier(d10K,w100_quad);
title('D_{10K} Samples Classified by D_{100} Training Set Classifier');
fprintf('P(error) for D100 training set classifier on D10K validation set: %f', pe100_quad);
% D1000 Classifier

% Train using D1000 dataset
z = [d1000.value];
labels = [d1000.label];
N = length(d1000);
[w1000_quad, ~] = fminsearch(@(w)(quadratic_cost_function(w,z,labels,N)), init_w);
plot_boundaries(d1000,w1000_quad);
title('Decision Boundary of D_{1000} Classifier');
% Find P(error) for D100 dataset classifer
pe1000_quad = score_classifier(d10K,w1000_quad);
title('D_{10K} Samples Classified by D_{1000} Training Set Classifier');
fprintf('P(error) for D1000 training set classifier on D10K validation set: %f', pe1000_quad);
%% Question 2
% Given a two-dimensional dataset $D=\{(x_1,y_1),...(X_N,y_N)\}$ generated occording 
% to the relationship $y=ax^3+bx^2+cx+d+v$, where $\mathbf{w}_{true} = [a,b,c,d]^T$ 
% are the true parameters of the x-y relationship, and $v\sim N(0,\sigma^2)$,  
% the MAP estimate for the parameter vector can be determined, using Bayes rule 
% and the independence of the dataset samples, by:
% 
% $$\mathbf{w}_{MAP}=\underset{\mathbf{w}}{argmax}\ p(\mathbf{w}|\mathbf{D}) 
% =\underset{\mathbf{w}}{argmax}\ p(\mathbf{D}|\mathbf{w})p(\mathbf{w})= \underset{\mathbf{w}}{argmax}\ 
% \prod_{i=1}^N p(x_i,y_i|\mathbf{w})p(\mathbf{w})$$
% 
% $$=\underset{\mathbf{w}}{argmax}\ \prod_{i=1}^N p(y_i|x_i,\mathbf{w})p(x_i|\mathbf{w})p(\mathbf{w})=\underset{\mathbf{w}}{argmax}\ 
% \prod_{i=1}^N p(y_i|x_i,\mathbf{w})p(\mathbf{w})$$
% 
% We are given the parameter prior $p(\mathbf{w}) \sim N(\mathbf{0},\gamma^2\mathbf{I})$, 
% and from the polynomial relationship between x and y, we create a function $y_i 
% \approx h(x_i,\mathbf{w})+v$ that approximates the relationship. Hence, $p(y_i|x_i,\mathbf{w}) 
% \sim N(h(x_i,\mathbf{w}),\sigma^2)$. Plugging these distributions into the above 
% equation, and taking the natural log, the optimization problem becomes:
% 
% $$\mathbf{w}_{MAP}=\underset{\mathbf{w}}{argmax}\ \sum_{i=1}^N ln(p(y_i|x_i,\mathbf{w})) 
% + ln(p(\mathbf{w}))=\underset{\mathbf{w}}{argmax}\ \sum_{i=1}^N ln\Big(\frac{1}{\sqrt{2\pi\sigma}}e^{-\frac{1}{2}\frac{(y_i-h(x_i,\mathbf{w}))^2}{\sigma^2}}\Big)+ln\{(2\pi)^{-n/2}|\gamma\mathbf{I}|^{-1/2}e^{-\frac{1}{2}\mathbf{w}^T(\gamma\mathbf{I})^{-1}\mathbf{w}}\}$$
% 
% Removing terms that do not depend on $\mathbf{w}$ and simplifying, 
% 
% $$\mathbf{w}_{MAP}=\underset{\mathbf{w}}{argmax}\ -\frac{1}{2\sigma^2}\sum_{i=1}^N(y_i-h(x_i,\mathbf{w}))^2-\frac{\mathbf{w}^T\mathbf{w}}{2\gamma}$$ 
% 
% which can be rewritten to see the relationship between $\sigma^2,\ \gamma,$ 
% and the term $\mathbf{w}^T\mathbf{w}$ as:
% 
% $$\mathbf{w}_{MAP}=\underset{\mathbf{w}}{argmin}\ \sum_{i=1}^N(y_i-h(x_i,\mathbf{w}))^2+\frac{\sigma^2}{\gamma}\mathbf{w}^T\mathbf{w}}$$
% 
% and if we let $h(x_i,\mathbf{w}) = \mathbf{w}^T\mathbf{z}(x_i)$, where $\mathbf{z}(x_i)=[x_i^3, 
% x_i^2,x_i,1]^T$,
% 
% $$\mathbf{w}_{MAP}=\underset{\mathbf{w}}{argmin}\ \sum_{i=1}^N(y_i-\mathbf{w}^T\mathbf{z}(x_i))^2+\frac{\sigma^2}{\gamma}\mathbf{w}^T\mathbf{w}}$$
% 
% By taking the gradient $\nabla_\mathbf{w}$, setting the equation to 0 and, 
% and removing terms that do not depend on $\mathbf{w}$, we get:
% 
% $0 = -2\sum_{i=1}^Ny_i\mathbf{z}(x_i)+2(\sum_{i=1}^N\mathbf{z}(x_i)\mathbf{z}(x_i)^T)\mathbf{w}_{MAP}+2\frac{\sigma^2}{\gamma}\mathbf{w}_{MAP}$, 
% and solving for $\mathbf{w}_{MAP}$,
% 
% $$\mathbf{w}_{MAP}=\Big(\sum_{i=1}^{N}\mathbf{z}(x_i)\mathbf{z}(x_i)^T+\frac{\sigma^2}{\gamma}I\Big)^{-1}\sum_{i=1}^Ny_i\mathbf{z}(x_i)$$
% 
% Generation of samples according to the cubic polynomial relationship with 
% additive noise, and 200 parameter vector MAP estimate experiments per gamma:

N = 10;
% Generate parameters for w_true
% true_roots = -1 + (1+1)*rand(1,3);
true_roots = [0.5 -0.5 0.9];
w_true = poly(true_roots)';
sig = 1;
gamma = logspace(-4,4,100);
error_vals = zeros(5,length(gamma));
for ii = 1:length(gamma)
    l2_dist = zeros(1,100);
    for jj = 1:200
        % Generate 10 iid samples x~U[-1,1]
        x = (-1 + (1+1)*rand(N,1));
        z = [x.^3 x.^2 x ones(length(x),1)]';
        % z = [ones(length(x),1) x x.^2 x.^3]';
        % Generate additive noise v~N(0,sig)
        v = randGaussian(N,0,sig);
        y = w_true'*z + v;
        w_map = inv(z*z' + (sig/gamma(ii))*eye(4,4))*(z*y');
        % y_map = w_map'*z;
        l2_dist(jj) = norm(w_true-w_map)^2;
    end
    % Store min, 25%, 50%, median, 75%, and max l2 distances for each gamma
    error_vals(:,ii) = [min(l2_dist); prctile(l2_dist,[25 50 75])'; max(l2_dist)];
end
clf;
semilogx(gamma, error_vals(1:4,:)); xlabel('\gamma'); ylabel('L_2 Error');
title({'L_2 Errors for Each Gamma','200 Experiments'});
legend('Minimum','25 Percentile','50 Percentile','75 Percentile');
semilogx(gamma, error_vals(5,:)); xlabel('\gamma'); ylabel('L_2 Error');
title({'Maximum L_2 Errors for Each Gamma','200 Experiments'});
%% Question 3
% Step 1: GMM data generation
% The true GMM parameters were selected to be:
% 
% $$\mu_1 = \left[\matrix{ -8\cr 0} \right]\ \mu_2= \left[\matrix{ 1\cr 4} \right]\ 
% \mu_3 = \left[\matrix{10\cr 0} \right]\ \mu_4 = \left[\matrix{ 0\cr -6} \right]\ 
% $$ 
% 
% $$\Sigma_1=\frac{1}{2}\left[\matrix{ 3& 1 \cr 1 &20} \right],\ \Sigma_2=\left[\matrix{ 
% 7& 1 \cr 1 &2} \right],\ \Sigma_3=\frac{1}{6}\left[\matrix{ 4& 1 \cr 1 &16} 
% \right],\ \Sigma_4=\frac{1}{2}\left[\matrix{ 7& 1\cr 1 &2} \right]$$
% 
% $$\alpha=\left[\matrix{0.2&0.3&0.3&0.2}\right]$$

% Specify component parameters
mu_true = [-8 1 10 0;0 4 0 -6];
sigma_true(:,:,1) = [3 1;1 20]/2;
sigma_true(:,:,2) = [7 1;1 2];
sigma_true(:,:,3) = [4 1;1 16]/6;
sigma_true(:,:,4) = [7 1;1 2]/2;
% Probability of each mixture component
alpha_true = [0.2,0.3,0.3,0.2];

d10 = generate_gaussian_mixture(10,mu_true,sigma_true,alpha_true);
d100 = generate_gaussian_mixture(100,mu_true,sigma_true,alpha_true);
d1000 = generate_gaussian_mixture(1000,mu_true,sigma_true,alpha_true);

% Plot with class labels
clf;
plot(d1000(1,:),d1000(2,:),'co');
title('True GMM Samples');
xlabel('x_1'); ylabel('x_2');
% Step 2 - Computation

% Tolerance for EM convergence criterion
delta = 1e-5;
%% 
% Model order selection for 15 sample bootstrapped datasets, based off the 10 
% sample initial GMM dataset:

train_and_validate(100,d10,15,d10,15,delta);
%% 
% Model order selection for 45 sample bootstrapped datasets, based off the 1000 
% sample initial GMM dataset:

train_and_validate(100,d1000,45,d1000,45,delta);
%% 
% Model order selection for 85 sample bootstrapped datasets, based off the 100 
% sample initial GMM dataset:

train_and_validate(100,d100,85,d100,85,delta);
%% 
% Model order selection for 400 sample bootstrapped datasets, based off the 
% 100 sample initial GMM dataset:

train_and_validate(100,d100,400,d100,400,delta);
%% 
% Model order selection for 10,000 sample bootstrapped datasets, based off the 
% 100 sample initial GMM dataset:

train_and_validate(100,d1000,300,d1000,300,delta);
%% 
% Model order selection for 10,000 sample bootstrapped datasets, based off the 
% 10 sample initial GMM dataset:

train_and_validate(100,d10,10000,d10,10000,delta);
%%
train_and_validate(100,d1000,900,d1000,900,delta);
%% 
% Model order selection for 5,000 sample bootstrapped datasets, based off the 
% 1000 sample initial GMM dataset:

train_and_validate(100,d1000,5000,d1000,5000,delta);
%% 
% Model order selection for 7,000 sample bootstrapped datasets, based off the 
% 1000 sample initial GMM dataset:

train_and_validate(100,d1000,7000,d1000,7000,delta);
%% 
% Model order selection for 10,000 sample bootstrapped datasets, based off the 
% 1000 sample initial GMM dataset:

train_and_validate(100,d1000,10000,d1000,10000,delta);
%% 
% Model order selection for 15,000 sample bootstrapped datasets, based off the 
% 1000 sample initial GMM dataset:

train_and_validate(100,d1000,15000,d1000,15000,delta);
% Step 3 - Report
% EM Algorithm Initialization
% My expecation maximization algorithm was derived from the "EMforGMM.m" file 
% provided by Prof. Erdogmuz for this course. I modified the initialization criteria 
% to implement several changes. The first is the k-means++ algorithm as described 
% in the paper "k-means++: The Advantages of Careful Seeding" by Arthur and Vassilvitskii. 
% I found that the original initialization procedure from "EMforGMM.m" often led 
% to mean and covariance estimates far from optimized values produced by the EM-GMM 
% algorithm. The discrepancy between initialized parameters and optimized parameters 
% was causing slow or nonexistent convergences. The k-means++ algorithm starts 
% with uniformly chosen mean vectors, as before, but then picks new mean values 
% using a probability distribution based on squared distances from the uniformly 
% chose mean vectors. Arthur and Vassilvitskii use k-means++ to seed the k-means 
% algorithm, which is a variant of the EM algorithm. When I applied k-means++ 
% to the EM-GMM algorithm, I found faster convergence times and improved model 
% order estimates. I also modified the initial GMM component weights from the 
% uniform proportions as in "EMforGMM.m", to instead be based off the number of 
% data points associated with each k-means++ derived mean vector. The reason for 
% this was, again, to improve model order estimates and convergence times.
% EM Algorithm Convergence
% The "EMforGMM.m" convergence criteriion was based off differences between 
% iterations of estimated GMM parameters. I found slow or nonexistent convergences 
% from various datasets using this criterion. I modifed the convergence criteria 
% to be based off $\Delta(\theta|\theta_n) = L(\theta) - L(\theta_n)$, the difference 
% between the current log likelihood and log likelihood for a prior iteration 
% of EM-GMM model parameter estimates. This convergence approach is described 
% in section 3.2 of "The Expectation Maximization Algorithm" by Sean Borman. I 
% set $\Delta(\theta|\theta_n) < 1\times10^{-5}$ as a convergence criterion.This 
% change resulted in faster convergence times.
% Experiment Design
% All experiments were run with various bootstrapped datasets derived from the 
% initial 10, 100, and 1000 sample sets. Each experiment consisted of 10 iterations 
% of:
% 
% 1. Generating bootstrapped training and validation sets of a specified size 
% from a specified dataset. Bootstrapped datasets were generated by sampling with 
% replacement from an initial sample set.
% 
% 2. Iterating through the Expectation-Maximization GMM Algorithm for GMM models 
% with k = 1,...,6 components, usng the training set, until convergence criterion 
% reached. This step produced 6 GMM models with mean, covariance, and GMM weight 
% parameters for each.
% 
% 3. Calculating the log likelihood of the EM-derived Maximum Likelihood parameters 
% for each GMM mode orderl (with components k = 1,...,6)  using the training dataset, 
% and storing these values as a performance measure for later comparison.
% 
% Each experiment was repeated 100 times, and an average performance value was 
% calculated from these 100 trials for each model order. A maximum of these average 
% performance values was found, and the corresponding model order was defined 
% as the model order selected by the EM-GMM algorithm.
% Results
% Histograms for model selections from various datasets are included in Section 
% 2. From these histograms, we can see that true model order selection was achieved 
% by training and validating on large datasets based off the 1000 sample initial 
% dataset (see the 10,000 and 15,000 sample training/validation datasets based 
% off the 1000 sample dataset). The trend overall, is closer model order selection 
% to k=4 components with increasing dataset size. Smaller datasets, and even larger 
% datasets bootstrapped from smaller initial datasets, did not yield correct model 
% order selection in the cross-validation process. For example, the plot for a 
% 10,000 sample training/validation set experiment shows the model order selection 
% to be 6. It appears the EM algorithm is overfitting on a bootstrapped dataset 
% in this case. Datasets below approximately 10,000 samples yielded a model order 
% selection of 1; absent a high quantity of unique data points, the EM algorithm 
% is unable to discern individual components, and converges to a single-component 
% GMM. Parameter estimation for GMMs with several components requires a large 
% amount of data; my experiment results agree with this.