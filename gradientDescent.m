function [theta, J_history] = gradientDescent(X, ...
											  y, ...
											  lambda, ...
											  ctype, ...
											  alpha, ...
											  num_iters)
%-------------------------------------------------------------------------
% gradientDescent performs the gradient descent
%   [theta] = the final theta calculatd out
%   [J_history] = the cost value history throughout the gradient descent,
%       used to check if the cost converges
%
%   [X]: is the m*n matrix contains m samples with n features
%   [y]: is the m*1 vecter contain value of training samples
%   [theta]: is the (n+1)*1 vector contains the current/initial theta value. 
%   [lambda]: is the regularization param
%   [ctype]: "linear" or "logistic"
%   [alpha]: the step to go for each descent
%   [num_iters]: number of iterations to perform
%  
%   Note: if the dimension of X,y,theta is not correct (X,Y dim-inversed), we will correct it
%-------------------------------------------------------------------------

	if(size(X,1) < size(X,2))
		X = X';
	end

	theta = zeros(size(X,2)+1, 1);

	for iter = 1:num_iters
		[J_history(iter), grad] = computeCost(X, y, theta, lambda, ctype);
		theta = theta - alpha * grad;
	end
end
