% Load the Iris dataset
load fisheriris.mat;

% Preprocess the data
data = meas(51:end, 3:4); % Select only 2nd and 3rd iris classes and 3rd and 4th features
classes = species(51:end);

% Convert categorical classes to numerical
classes_num = grp2idx(classes) - 1;

% Normalize data
data = (data - mean(data, 1)) ./ std(data, 1);

% Define the sigmoid function
sigmoid = @(x) 1./(1 + exp(-x));

% Neural network function
simple_nn = @(x, w, b) sigmoid(x * w + b);

% Mean squared error function
mse = @(pred, true) mean((pred - true).^2, 'all');

% One-hot encoding for the true classes
true_classes = [classes_num == 1, classes_num == 2];

% First setting of the neural network parameters
weights1 = [1, -1]';
bias1 = 0;
predictions1 = simple_nn(data, weights1, bias1);
error1 = mse(predictions1, true_classes);

% Second setting of the neural network parameters
weights2 = [-3, 4]';
bias2 = 2;
predictions2 = simple_nn(data, weights2, bias2);
error2 = mse(predictions2, true_classes);

fprintf('Mean-squared error for the first setting: %.4f\n', error1);
fprintf('Mean-squared error for the second setting: %.4f\n', error2);

% Plot the decision boundaries
figure;
hold on;

% Scatter plot using plot function
class2 = data(classes_num == 1, :);
class3 = data(classes_num == 2, :);
plot(class2(:, 1), class2(:, 2), 'bo');
plot(class3(:, 1), class3(:, 2), 'go');

xlabel('Feature 3 (normalized)');
ylabel('Feature 4 (normalized)');
title('Iris Dataset (2nd and 3rd Classes) with Decision Boundaries');

% First decision boundary
x1 = linspace(min(data(:, 1)), max(data(:, 1)), 100);
y1 = -(weights1(1) * x1 + bias1) / weights1(2);
plot(x1, y1, 'r', 'LineWidth', 2);

% Second decision boundary
x2 = linspace(min(data(:, 1)), max(data(:, 1)), 100);
y2 = -(weights2(1) * x2 + bias2) / weights2(2);
plot(x2, y2, 'm', 'LineWidth', 2);

legend('Class 2', 'Class 3', 'Decision Boundary 1', 'Decision Boundary 2');
hold off;
