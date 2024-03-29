function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.1;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

possibleValues = [.001, .03, .1, .3, 1, 3, 10, 30];
length = length(possibleValues);

errors = zeros(length, length);

for i = 1:length
	for j = 1:length
		model = svmTrain(X, y, possibleValues(i), @(x1, x2) gaussianKernel(x1, x2, possibleValues(j)));
		predictions = svmPredict(model, Xval);
		errors(i, j) = mean(double(predictions ~= yval));
	end
end

[minRow, minRowPosition] = min(errors);
[minCol, minColPosition] = min(minRow);
i = minRowPosition(minColPosition);
j = minColPosition;
C = possibleValues(i)
sigma = possibleValues(j)





% =========================================================================

end
