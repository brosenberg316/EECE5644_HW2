function [alpha,mu,Sigma] = gmm_expectation_maximization(M,x,delta)
% Adopted from Prof. Erdogmuz EMforGMM.m file, with edits for convergence
% criteria and k-means++ algorithm initialization, and initial GMM
% component weights

% Regulation weight for covariance matrices
regWeight = 1e-10;
% Dimensionality of data
d = size(x,1);
% Number of samples in data
N = size(x,2);
% Assume uniform component mixture probability
% alpha = ones(1,M)/M;

% Pick M random samples as initial mean estimates
shuffledIndices = randperm(N);
mu = x(:,shuffledIndices(1:M));

% C = x(:,1+round(rand*(size(X,2)-1)));
L = ones(1,size(x,2));
% Use k-means++ algorithm to improve mean estimates
for i = 2:M
    % Each sample is weighted by distance from current mu
    D = x-mu(:,L);
    D = cumsum(sqrt(dot(D,D,1)));
    if D(end) == 0
        mu(:,i:k) = x(:,ones(1,M-i+1)); 
        return; 
    end
    % Pick new mu, sampled from these distribution weights 
    mu(:,i) = x(:,find(rand < D/D(end),1));
    [~,L] = max(bsxfun(@minus,2*real(mu'*x),dot(mu,mu,1).'));
end


% Initialize distance array with infinite values
% D = inf(1,N);
% for i = 2:M
%     % For each model order
%     % Each sample is weighted by distance from current mu
%     D = min(D,sum((x-mu(:,i-1)).^2,1));
%     % Pick new mu, sampled from these distribution weights 
%     idx = find(rand < cumsum(D/sum(D)),1);
%     mu(:,i) = x(:,idx);
% end
% Assign each sample to the nearest mu
[~,assignedCentroidLabels] = min(pdist2(mu',x'),[],1); 

% Use sample covariances of k-means++ mu's as initial covariance estimates
for m = 1:M
    Sigma(:,:,m) = cov(x(:,find(assignedCentroidLabels==m))') + regWeight*eye(d,d);
end

% Use number of points assigned to each cluster as starting GMM weight
alpha = zeros(1,M);
for ii = 1:M    
    alpha(ii) = sum(assignedCentroidLabels == ii)/N;
end

% Begin EM Algorithm
converged = false;
while ~converged
    for l = 1:M
        temp(l,:) = repmat(alpha(l),1,N).*evalGaussian(x,mu(:,l),Sigma(:,:,l));
    end
    plgivenx = temp./sum(temp,1);
    alphaNew = mean(plgivenx,2);
    w = plgivenx./repmat(sum(plgivenx,2),1,N);
    muNew = x*w';
    for l = 1:M
        v = x-repmat(muNew(:,l),1,N);
        u = repmat(w(l,:),d,1).*v;
        SigmaNew(:,:,l) = u*v' + regWeight*eye(d,d);
    end
    % Calculate log likelihoods for old and new parameter estimates
    Lold = sum(log(evalGMM(x,alpha,mu,Sigma)));
    Lnew = sum(log(evalGMM(x,alphaNew,muNew,SigmaNew)));
    % Convergence criteria based on difference between new and old log
    % likelihoods, adopted from section 3.2 in paper 
    % "The Expectation Maximization Algorithm" by Sean Borman
    converged = abs(Lnew-Lold) < delta;
    % Assign updated parameters
    alpha = alphaNew; mu = muNew; Sigma = SigmaNew;
end
end

function gmm = evalGMM(x,alpha,mu,Sigma)
gmm = zeros(1,size(x,2));
for m = 1:length(alpha) % evaluate the GMM on the grid
    gmm = gmm + alpha(m)*evalGaussian(x,mu(:,m),Sigma(:,:,m));
end
end