% Load Iris dataset
load fisheriris;
X = meas; % Iris dataset features
X = normalize(X); % Normalize the data

% Run k-means clustering for k = 2 and k = 3
k_values = [2, 3];
for k = k_values
    % Initialize k-means
    [initialCentroids, initialIdx] = kmeansInit(X, k);
    
    % Run k-means clustering algorithm
    [finalCentroids, finalIdx, allCentroids] = kmeansClustering(X, k, initialCentroids);
    
    % Plot results
    plotKmeans(X, initialCentroids, allCentroids, finalCentroids, finalIdx, k);
    
    % Plot decision boundaries
    plotDecisionBoundaries(X, finalCentroids, k);
end

% Implement k-means clustering algorithm
function [initialCentroids, initialIdx] = kmeansInit(X, k)
    initialIdx = randi(k, size(X, 1), 1);
    initialCentroids = arrayfun(@(i) mean(X(initialIdx == i, :), 1), 1:k, 'UniformOutput', false);
    initialCentroids = vertcat(initialCentroids{:});
end

function [finalCentroids, finalIdx, allCentroids] = kmeansClustering(X, k, initialCentroids)
    maxIter = 100;
    prevCentroids = initialCentroids;
    allCentroids = cell(1, maxIter);
    
    for i = 1:maxIter
        % Assign data points to closest centroid
        [~, finalIdx] = min(pdist2(X, prevCentroids), [], 2);
        
        % Update centroids
        newCentroids = arrayfun(@(i) mean(X(finalIdx == i, :), 1), 1:k, 'UniformOutput', false);
        newCentroids = vertcat(newCentroids{:});
        
        % Check convergence
        if all(all(abs(newCentroids - prevCentroids) < 1e-4))
            break;
        end
        
        allCentroids{i} = prevCentroids;
        prevCentroids = newCentroids;
    end
    allCentroids = allCentroids(1:i);
    finalCentroids = newCentroids;
end

function plotKmeans(X, initialCentroids, allCentroids, finalCentroids, finalIdx, k)
    figure;
    hold on;
    
    % Plot initial centroids
    plot(initialCentroids(:, 1), initialCentroids(:, 2), 'kx', 'MarkerSize', 15, 'LineWidth', 3);
    
    % Plot intermediate centroids
    for i = 1:numel(allCentroids)
        centroids = allCentroids{i};
        plot(centroids(:, 1), centroids(:, 2), 'mx', 'MarkerSize', 10, 'LineWidth', 2);
    end
    
    % Plot converged centroids
    plot(finalCentroids(:, 1), finalCentroids(:, 2), 'rx', 'MarkerSize', 15, 'LineWidth', 3);
    
    % Plot data points
       colors = lines(k);
    for i = 1:k
        plot(X(finalIdx == i, 1), X(finalIdx == i, 2), '.', 'Color', colors(i, :), 'MarkerSize', 10);
    end

    title(sprintf('K-means clustering for k = %d', k));
    xlabel('Feature 1');
    ylabel('Feature 2');
    hold off;
end

function plotDecisionBoundaries(X, finalCentroids, k)
    % Create a meshgrid to cover the feature space
    [x1, x2] = meshgrid(linspace(min(X(:, 1)), max(X(:, 1)), 100), ...
                        linspace(min(X(:, 2)), max(X(:, 2)), 100));
    meshGrid = [x1(:), x2(:)];

    % Assign each point in the meshgrid to the closest centroid
    [~, meshGridIdx] = min(pdist2(meshGrid, finalCentroids), [], 2);

    % Reshape meshGridIdx back to the size of the meshgrid
    decisionBoundary = reshape(meshGridIdx, size(x1));

    figure;
    hold on;

    % Plot decision boundaries
    contour(x1, x2, decisionBoundary, 'LineWidth', 1);

    % Plot data points and final centroids
    colors = lines(k);
    for i = 1:k
        plot(X(:, 1), X(:, 2), '.', 'Color', colors(i, :), 'MarkerSize', 10);
    end
    plot(finalCentroids(:, 1), finalCentroids(:, 2), 'rx', 'MarkerSize', 15, 'LineWidth', 3);

    title(sprintf('Decision boundaries for k = %d', k));
    xlabel('Feature 1');
    ylabel('Feature 2');
    hold off;
end




