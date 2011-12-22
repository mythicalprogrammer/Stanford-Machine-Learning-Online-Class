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

%disp(size(Theta1));
%disp(size(Theta2));
%disp(size(X));

y = eye(num_labels)(y,:); %5000 10
a_1 = [ones(m, 1), X]; % Add a column of ones to x
z_2 = a_1*Theta1';
a_2 = sigmoid(z_2);
m2 = size(a_2, 1);
a_2 = [ones(m2, 1), a_2];
z_3 = a_2*Theta2';
a_3 = sigmoid(z_3);
h_x = a_3; % 5000 10

inner_summation=0; 
for i = 1:m,
	for j=1:num_labels, 
		inner_summation = inner_summation+(-y(i,j)*log(h_x(i,j))-(1-y(i,j))*log(1-h_x(i,j))); 
	end; 
end;

J = (1/m)*inner_summation;

% regularization
% -------------------------------------------------------------


j_theta_1 = size(Theta1,1); % 25
k_theta_1 = input_layer_size + 1; %400 because theta1 is actually 401 skip one bias

summation_theta_1 = 0;
for j = 1:j_theta_1,
	for k = 2:k_theta_1, % skip bias
		summation_theta_1 = summation_theta_1 + Theta1(j,k)*Theta1(j,k);
	end; 
end;

j_theta_2 = num_labels; % 10
k_theta_2 = hidden_layer_size + 1; % 25 because theta2 is actually 26 skip one bias

summation_theta_2 = 0;
for j = 1:j_theta_2,
	for k = 2:k_theta_2, % skip bias
		summation_theta_2 = summation_theta_2 + Theta2(j,k)*Theta2(j,k);
	end; 
end;

regularization = (lambda/(2*m))*(summation_theta_1+summation_theta_2);

J = J + regularization;

% Backpropagation
% -------------------------------------------------------------

%y = eye(num_labels)(y,:); %5000 10
%a_1 = [ones(m, 1), X]; % Add a column of ones to x
%z_2 = a_1*Theta1'; 5000 x 25
%a_2 = sigmoid(z_2); 5000 x 26 
%m2 = size(a_2, 1);
%a_2 = [ones(m2, 1), a_2];
%z_3 = a_2*Theta2';
%a_3 = sigmoid(z_3);
%h_x = a_3; % 5000 10
delta_3 = [zeros(num_labels,1)]; % 10 x 1
Theta2(:,1) = []; % kill the bias % 10 x 25
%a_2(:,1) = []; % kill the bias 5000 x 25
%disp('y(t,:)');
%disp(size(y(1,:)));
%disp(y(1,:));
%disp('a_3(t,:)');
%disp(size(a_3(1,:)));
%disp(a_3(1,:));
%disp('delta_2');
%disp(size(delta_2));
%disp('delta_3');
%disp(size(delta_3));
%disp('a_2');
%disp(size(a_2));
%disp('Theta2');
%disp(size(Theta2));
%disp('z_2');
%disp(size(z_2));
%j_theta_1 = size(Theta1,1); % 25
%for t = 1:m,
	delta_3 = a_3 - y;
	%for k = 1:num_labels,
	%		delta_3 = delta_3 + (a_3(t,k) - y(t,k));
	%end;
%end;

%disp(size(delta_3));


%disp('meow');
%disp(size(Theta1_grad));
%disp(size(Theta2_grad));
%delta_2 = sigmoidGradient(z_2) * (Theta2' * delta_3 )'; % 5000 x 1
%delta_2 = sigmoidGradient(z_2) * (Theta2' * delta_3)'; % 5000 x 1
delta_2 = delta_3*Theta2 .* sigmoidGradient(z_2); 
big_delta_1 = 0;
big_delta_2 = 0;
big_delta_1 = big_delta_1 + delta_2'*a_1;
big_delta_2 = big_delta_2 + delta_3'*a_2;
%delta_2 = Theta2' * delta_3 .* sigmoidGradient(z_2); % 5000 x 1
%disp('delta2');
%disp(size(delta_2));
Theta1_grad = (1/m)*big_delta_1; 
Theta2_grad = (1/m)*big_delta_2;


%regularization
Theta1_grad_without_bias = Theta1_grad;
Theta1_grad_without_bias(:,1) = [];
Theta2_grad_without_bias = Theta2_grad;
Theta2_grad_without_bias(:,1) = [];
Theta1(:,1) = []; % kill the bias % 25 x 400
%Theta1_grad = Theta1_grad(2,end) + (lambda/m)*Theta1; 
%Theta2_grad = Theta2_grad(2,end) + (lambda/m)*Theta2; 
Theta1_grad_without_bias = Theta1_grad_without_bias + (lambda/m)*Theta1; 
Theta2_grad_without_bias = Theta2_grad_without_bias + (lambda/m)*Theta2; 
Theta1_grad = [Theta1_grad(:, 1), Theta1_grad_without_bias];
Theta2_grad = [Theta2_grad(:, 1), Theta2_grad_without_bias];
%Theta2_grad_without_bias 
% dead code for cost function >___<-------------------------------------------------------------

%cur_index = 0;
%cur_high = 0;
%temp = 0;
%inner_summation = [zeros(num_labels,1)];
%disp(inner_summation);
%for i = 1:m,
%	temp = 0;
%	for j = 1:num_labels,
%		if y(i,:) == j,
%			temp = -y(i,:)'*log(sigmoid(h_x(i,j)))-(1-y(i,:))'*log(1-sigmoid(h_x(i,j)));
%			inner_summation(j,1) = inner_summation(j,1) + temp;
%		end;
	%	if y(i,:) == j,
	%		cur_index = j;
	%		temp = -y(i,:)'*log(sigmoid(h_x(i,cur_index)))-(1-y(i,:))'*log(1-sigmoid(h_x(i,cur_index)));
	%		if i == 1,
	%			save_temp = temp;
	%		end;
	%	end;
		%value = h_x(i,j);
		%if j == 1,
		%	cur_high= value;
		%	cur_index = j;
		%end;
		%if value > cur_high,
		%	cur_high = value;
		%	cur_index = j;
		%end;
%	end;
	%temp = -y(i,:)'*log(sigmoid(h_x(i,cur_index)))-(1-y(i,:))'*log(1-sigmoid(h_x(i,cur_index)));
	%inner_summation(cur_index,1) = inner_summation(cur_index,1) + temp;
%end;
%disp(save_temp);
%disp(inner_summation);
%disp(size(inner_summation));
%disp(inner_summation);
%disp(y(1,:));
%J = (1/m)*sum(inner_summation);


%m2 = size(summation_theta1, 1);
%summation_theta1 = sigmoid(summation_theta1);
%summation_theta1 = [ones(m2, 1), summation_theta1];
%disp(size(summation_theta1));
%disp(size(Theta2));
%h_x_2 = summation_theta1*Theta2';
%disp(size(h_x_2));
%summation_theta2 = -y'*log(sigmoid(h_x_2))-(1-y')*log(1-sigmoid(h_x_2));
%summation_theta2 = -y'*log(sigmoid(h_x_2))-(1-y')*log(1-sigmoid(h_x_2));
%innersummation_1_to_k = summation_theta1 + summation_theta2;
%fitting = sum(inner_summation_1_to_k); 

%J = ((1/m)*fitting);









% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
