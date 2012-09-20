function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%let's compute all the node values first
a1 = X'; % 400 x 5000, representing input
a1 = [ones(1, size(a1, 2)); a1]; % 401 x 5000, added the bias node

z2 = Theta1 * a1; % 25 x 5000
a2 = sigmoid(z2); % 25 x 5000
a2 = [ones(1, size(a2, 2)); a2]; % 26 x 5000, added the bias node

z3 = Theta2 * a2; % 10 x 5000
a3 = sigmoid(z3); % 10 x 5000, representing final predictions for all 10 digits, for all 5000 training examples


logicalY = zeros(num_labels, m); % 10 x 5000 of zeros

for counter = 1:m
    logicalY(y(counter), counter) = 1; % set which label the training example actually corresponds to
end

regularization = (lambda / (2 * m)) * (sum(sumsq(Theta1(:, 2:end))) + sum(sumsq(Theta2(:, 2:end))));

J = (1 / m) *   sum(sum((-logicalY .* log(a3)) - ((1 - logicalY) .* log(1 - a3)))) + regularization;

%delta3 = a3 - logicalY; % 10 x 5000
%delta2 = (Theta2' * delta3) .* sigmoidGradient(z2); % 26 x 5000

%Delta1 = 

X = [ones(size(X, 1), 1), X];
Delta1 = Delta2 = 0;

for counter = 1:m
	a1 = X(counter, :)'; % 401 x 1

	z2 = Theta1 * a1; % 25 x 1
	a2 = sigmoid(z2); % 25 x 1 
	a2 = [ones(1, size(a2, 2)); a2]; % 26 x 1

	z3 = Theta2 * a2; % 10 x 1
	a3 = sigmoid(z3); % 10 x 1

	delta3 = a3 - logicalY(:, counter); % 10 x 1, differences between predicted and actual, for each label

	delta2 = (Theta2' * delta3) .* [0 ; sigmoidGradient(z2)]; % 26 x 1
	delta2 = delta2(2:end); % 25 x 1

	Delta1 = Delta1 + delta2 * a1'; % 25 x 401
	Delta2 = Delta2 + delta3 * a2'; % 10 x 26
end

Theta1_grad = Delta1 / m;
Theta2_grad = Delta2 / m;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
