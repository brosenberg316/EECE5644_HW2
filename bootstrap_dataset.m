function data = bootstrap_dataset(x,num_samples)
b_idx = randi([1 length(x)],1,num_samples);
data = x(:,b_idx);
v = randGaussian(length(data),[0;0],1e-8*eye(2,2));
data = data + v;