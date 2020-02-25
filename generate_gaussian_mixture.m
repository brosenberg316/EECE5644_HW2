function samples = generate_gaussian_mixture(num_samples,mu,sigma,p)
samples = zeros(size(mu,1), num_samples);
cum_p = [0,cumsum(p)];
u = rand(1,num_samples); 
for m = 1:length(p)
    ind = find(cum_p(m)<u & u<=cum_p(m+1)); 
    samples(:,ind) = randGaussian(length(ind),mu(:,m),sigma(:,:,m));
end