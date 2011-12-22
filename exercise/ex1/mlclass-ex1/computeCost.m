function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% J(0) = delta = (1/m)*for(i to m){(h_x - y) * x}
%h_x = theta'*X;

%matrix = X'*y*(1/(2*m));
%J = theta - matrix;
%J = (1/m)*((h_x-y)*X);
h_x = (X*theta);
J = (1/(2*m))*sum((h_x-y).^2);
%J = theta;
	
% =========================================================================

end
