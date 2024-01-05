% Load iris dataset
load fisheriris.mat
data = meas;

% Set parameters
k_values = [2, 3];
max_iterations = 100;

figure;
for k = k_values
    [centroids, cluster_labels] = kmeans_clustering(data, k, max_iterations);

    subplot(1, 2, find(k_values == k));
    plot_decision_boundaries(data, k, centroids, cluster_labels);
end

% Define the kmeans_clustering function
function [centroids, cluster_labels] = kmeans_clustering(data, k, max_iterations)
    % Normalize the data
    data = (data - min(data, [], 1)) ./ (max(data, [], 1) - min(data, [], 1));

    % Initialize centroids
    centroids = data(randperm(size(data, 1), k), :);

    for iter = 1:max_iterations
        % Assign each data point to the nearest centroid
        distances = pdist2(data, centroids);
        [~, cluster_labels] = min(distances, [], 2);

        % Update centroids
        new_centroids = zeros(k, size(data, 2));
        for i = 1:k
            if ~isempty(data(cluster_labels == i, :))
                new_centroids(i, :) = mean(data(cluster_labels == i, :));
            end
        end

        % Check for convergence
        if isequal(centroids, new_centroids)
            break;
        end
        centroids = new_centroids;
    end
end

% Define the plot_decision_boundaries function
function plot_decision_boundaries(data, k, centroids, cluster_labels)
    % Generate a dense grid of points
    x_min = min(data(:, 1)) - 0.1;
    x_max = max(data(:, 1)) + 0.1;
    y_min = min(data(:, 2)) - 0.1;
    y_max = max(data(:, 2)) + 0.1;
    [x, y] = meshgrid(linspace(x_min, x_max, 100), linspace(y_min, y_max, 100));
    grid_points = [x(:), y(:)];

    % Assign each grid point to the nearest centroid
    distances = pdist2(grid_points, centroids);
    [~, grid_labels] = min(distances, [], 2);

    % Plot the grid points with different colors for each centroid
    gscatter(grid_points(:, 1), grid_points(:, 2), grid_labels, 'brgcmyk', '.', 1);

    % Overlay the original dataset and the centroids
    hold on;
    gscatter(data(:, 1), data(:, 2), cluster_labels, 'brgcmyk', '.', 10);
    plot(centroids(:, 1), centroids(:, 2), 'kx', 'MarkerSize', 10, 'LineWidth', 2);
    hold off;
    title(['Decision boundaries for k = ' num2str(k)]);
    xlabel('Sepal Length');
    ylabel('Sepal Width');
end
