% Load the Iris dataset
load fisheriris.mat;

% Preprocess the data
data = meas;
classes = species;

% Convert categorical classes to numerical
classes_num = grp2idx(classes);

% Normalize data
data = (data - mean(data, 1)) ./ std(data, 1);

% Define the sigmoid function
sigmoid = @(x) 1./(1 + exp(-x));

% Define the neural network parameters
inputSize = size(data, 2);
outputSize = length(unique(classes_num));
weights = randn(inputSize, outputSize);
bias = randn(1, outputSize);

% Neural network function
simple_nn = @(x, w, b) sigmoid(x * w + b);

% Mean squared error function
mse = @(pred, true) mean((pred - true).^2, 'all');

% Calculate the neural network predictions
predictions = simple_nn(data, weights, bias);

% One-hot encoding for the true classes
true_classes = full(ind2vec(classes_num'))';

% Calculate the mean-squared error
error = mse(predictions, true_classes);
fprintf('Mean-squared error: %.4f\n', error);