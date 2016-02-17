function [ theta, J_history ] = quickDescent( X, ...
											  y, ...
											  ctype, ...
											  lambda, ...
											  maxIter)
%-------------------------------------------------------------------------
% quickDescent performs the advanced descent (fminunc)
%   [theta] = the final theta calculatd out
%   [cost] = the cost value with the given theta
%
%   [X]: is the m*n matrix contains m samples with n features
%   [y]: is the m*1 vecter contain value of training samples
%   [initial_theta]: is the (n+1)*1 vector contains the current/initial theta value. 
%   [lambda]: is the regularization param
%   [ctype]: "linear" or "logistic"
%   [maxIter]: The max iteration to perform
%  
%   Note: if the dimension of X,y,theta is not correct (X,Y dim-inversed), we will correct it
%-------------------------------------------------------------------------

	if(size(X,1) < size(X,2))
		X = X';
	end
    
    options = optimset('GradObj', 'on', 'MaxIter', maxIter); %options = optimset('GradObj', 'on', 'MaxIter', 400);
    [theta, J_history] = fmincg(@(t)computeCost(X, y, t, lambda,ctype),zeros(size(X,2)+1, 1),options)  %or replace fmincg with fminunc
end

