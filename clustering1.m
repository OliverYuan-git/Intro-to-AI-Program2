% Load the Iris dataset
load fisheriris.mat;

% Prepare the dataset
data = meas;

% Set the number of clusters (k) and the maximum number of iterations
k = 3;
max_iterations = 10000;

% Run the k-means algorithm
[cluster_indices, cluster_centers] = kmeans(data, k, 'MaxIter', max_iterations);

% Visualize the results
colors = ['r', 'g', 'b'];
figure;
for i = 1:k
    cluster_points = data(cluster_indices == i, :);
    plot(cluster_points(:, 1), cluster_points(:, 2), '.', 'Color', colors(i), 'MarkerSize', 12);
    hold on;
    plot(cluster_centers(i, 1), cluster_centers(i, 2), 'x', 'Color', colors(i), 'MarkerSize', 15, 'LineWidth', 3);
end
xlabel('Sepal length');
ylabel('Sepal width');
title('K-means Clustering for Iris Dataset');
legend('Cluster 1', 'Cluster 2', 'Cluster 3', 'Centroid 1', 'Centroid 2', 'Centroid 3', 'Location', 'northwest');
hold off;