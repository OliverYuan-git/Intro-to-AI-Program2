
% Load iris dataset
load fisheriris.mat
data = meas;

% Set parameters
k_values = [2, 3];
max_iterations = 100;

% Run k-means clustering and plot results
figure;
for k = k_values
    [centroids, cluster_labels] = kmeans_clustering(data, k, max_iterations);

    subplot(1, 2, find(k_values == k));
    gscatter(data(:, 1), data(:, 2), cluster_labels, 'brgcmyk', '.', 10);
    hold on;
    plot(centroids(:, 1), centroids(:, 2), 'kx', 'MarkerSize', 10, 'LineWidth', 2);
    hold off;
    title(['k-means clustering with k = ' num2str(k)]);
    xlabel('Sepal Length');
    ylabel('Sepal Width');
end

function [centroids, cluster_labels] = kmeans_clustering(data, k, max_iterations)
    % Normalize the data
    data = (data - min(data)) ./ (max(data) - min(data));

    % Initialize centroids
    centroids = data(randperm(size(data, 1), k), :);

    for iter = 1:max_iterations
        % Assign each data point to the nearest centroid
        distances = pdist2(data, centroids);
        [~, cluster_labels] = min(distances, [], 2);

        % Update centroids
        new_centroids = zeros(k, size(data, 2));
        for i = 1:k
            new_centroids(i, :) = mean(data(cluster_labels == i, :));
        end

        % Check for convergence
        if isequal(centroids, new_centroids)
            break;
        end
        centroids = new_centroids;
    end
end