function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
for iter = 1:num_iters
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

	% RAWR Hear me roar! Meow meow
	%theta = theta - alpha * delta 
	%theta(iter) = theta(iter) - alpha *
    h_x = (X*theta);
	%delta = (1/m)*sum((h_x-y).*X);
	delta = ((1/m)*(h_x - y))'*X;
	%delta = (1/m)*(h_x - y);
	
	theta = theta - (alpha * delta)';
	%theta = theta - alpha*delta;
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
	J_debug = J_history(iter);
end
end
