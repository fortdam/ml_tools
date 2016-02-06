% Align/Correct the parameters for linear/logistic regression.
% 
%  X should be a m*(n+1) matrix
%  y should be a m*1 vector
%  theta should be a (n+1)*1 vector

function [X,y,theta] = paramAlign(X, y, theta)
	if size(X,1) < size(X,2)
		X = X';
	end

	% Make sure Y is m*1 vector
	if size(y,1) < size(y,2)
		y = y';
	end

	%Make sure the theta is 
	if size(theta,1) < size(theta,2)
		theta = theta'; 
	end

	% add one column on the left, for X0 (if necessary)
	if X(1,1) ~= 1
	  X = [ones(length(X),1), X];
	end

	%Make sure the size of X,y,theta is correct
	if(size(X,1) ~= size(y,1)) 
		throw(MException('AcctError:Incomplete', 'The size of X and y are not same'));
	else if (size(X,2) ~= size(theta,1)) 
		throw(MException('AcctError:Incomplete','The size of X and theta are not same'));
	end
end