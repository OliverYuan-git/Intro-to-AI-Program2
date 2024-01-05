% Load the Iris dataset
load fisheriris.mat;

% Prepare the dataset 
data = meas;

% Set the number of clusters (k) and the maximum number of iterations
k = 3;
max_iterations = 10000;

% Run the modified k-means algorithm
[cluster_indices, cluster_centers, obj_func_values] = kmeans_obj_func(data, k, max_iterations);

% Plot the objective function value as a function of the iteration
figure;
plot(1:length(obj_func_values), obj_func_values, 'LineWidth', 2);
xlabel('Iteration');
ylabel('Objective Function Value');
title('K-means Objective Function Value vs. Iteration');


function [cluster_indices, cluster_centers, obj_func_values] = kmeans_obj_func(data, k, max_iterations)
    % Initialize cluster centers
    cluster_centers = datasample(data, k, 'Replace', false);
    % Initialize cluster indices and objective function values
    cluster_indices = zeros(size(data, 1), 1);
    obj_func_values = zeros(max_iterations, 1);

    for iter = 1:max_iterations
        % Assign data points to the closest cluster center
        for i = 1:size(data, 1)
            [~, cluster_indices(i)] = min(vecnorm(data(i, :) - cluster_centers, 2, 2));
        end

        % Update cluster centers
        for j = 1:k
            cluster_points = data(cluster_indices == j, :);
            cluster_centers(j, :) = mean(cluster_points);
        end

        % Compute the objective function value
        obj_func_value = 0;
        for n = 1:size(data, 1)
            for kk = 1:k
                r_nk = (cluster_indices(n) == kk);
                obj_func_value = obj_func_value + r_nk * norm(data(n, :) - cluster_centers(kk, :))^2;
            end
        end
        obj_func_values(iter) = obj_func_value;

        % Check for convergence
        if iter > 1 && obj_func_values(iter) == obj_func_values(iter-1)
            obj_func_values = obj_func_values(1:iter);
            break;
        end
    end
end