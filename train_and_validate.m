function model_order_wins = train_and_validate(num_experiments, d_train,n_train,d_validate,n_validate,delta)
% Initialize array of model order wins
model_order_wins = zeros(1,6);

for E = 1:num_experiments
    % Run the whole experiment many times
    perf_vals = zeros(10,6);
    for B = 1:10
        % Each experiment contains 10 trials
        % Create training and validation datasets
        d_train = bootstrap_dataset(d_train,n_train);
        d_validate = bootstrap_dataset(d_validate,n_validate);
        parfor M = 1:6
            % Iterate through model orders
            [alpha,mu,Sigma] = gmm_expectation_maximization(M,d_train,delta);
            perf_vals(B,M) = measure_performance(alpha,mu,Sigma,d_validate);
        end
    end
    % Determine model order with maximum performance value
    [~,max_idx] = max(mean(perf_vals,1));
    model_order_wins(max_idx) = model_order_wins(max_idx) + 1;
    % fprintf("Experiment %d Complete\n",E);
end
bar(model_order_wins); xlabel('Model Order'); ylabel('Number of Times Selected in Validation');
bar_title = sprintf('Number of times Each Model Order Selected\nD_{train} = %d samples, D_{validate} = %d samples',...
    length(d_train), length(d_validate)); title(bar_title);
end