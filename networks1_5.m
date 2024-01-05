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

% Derivative of the sigmoid function
sigmoid_derivative = @(x) sigmoid(x) .* (1 - sigmoid(x));

% Define the neural network parameters
inputSize = size(data, 2);
outputSize = length(unique(classes_num));
weights = randn(inputSize, outputSize) * 0.01;
bias = randn(1, outputSize) * 0.01;

% Neural network function
simple_nn = @(x, w, b) sigmoid(x * w + b);

% Mean squared error function
mse = @(pred, true) mean((pred - true).^2, 'all');

% One-hot encoding for the true classes
true_classes = full(ind2vec(classes_num'))';

% Training parameters
learning_rate = 0.1;
num_epochs = 5000;

% Training loop with gradient descent
for epoch = 1:num_epochs
    % Forward pass
    predictions = simple_nn(data, weights, bias);

    % Calculate the error
    error = mse(predictions, true_classes);

    % Backward pass
    derror_dpred = 2 * (predictions - true_classes);
    dpred_dnet = sigmoid_derivative(data * weights + bias);
    dnet_dw = data;

    % Calculate gradients
    derror_dw = (derror_dpred .* dpred_dnet)' * dnet_dw;
    derror_db = sum(derror_dpred .* dpred_dnet, 1);

    % Update weights and biases
    weights = weights - learning_rate * derror_dw';
    bias = bias - learning_rate * derror_db;

    % Print progress
    if mod(epoch, 100) == 0
        fprintf('Epoch: %d, Mean-squared error: %.4f\n', epoch, error);
    end
end

% Calculate the final neural network predictions
predictions = simple_nn(data, weights, bias);

% Calculate the final mean-squared error
final_error = mse(predictions, true_classes);
fprintf('Final mean-squared error: %.4f\n', final_error);
