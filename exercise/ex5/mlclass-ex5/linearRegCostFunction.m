function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


h_x = X*theta;
inner_summation_J = (h_x - y).^2;
fitting = sum(inner_summation_J); 
theta_except_0 = theta;
theta_except_0(1,:)=[];
regularization = (lambda/(2*m))*sum(theta_except_0.^2);
J = ((1/(2*m))*fitting)+regularization;


grad_0 = (1/m)*(X'*(h_x-y));
grad_1_m = (1/m)*(X'*(h_x-y));
grad_1_m(1,:) = [];
reg = (lambda/m)*theta_except_0;
grad_1_m_reg = grad_1_m + reg;
j = size(theta)-1;
for i=1:j
	grad((i+1),:) = grad_1_m_reg(i,:);
end;
grad(1,:) = grad_0(1,:);

% =========================================================================

grad = grad(:);

end
