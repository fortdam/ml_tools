%-------------------------------------------------------------------------------------
%computeCostLinear Compute cost for linear regression with single/multiple variables
%   J = computeCostLinear(X, y, theta) computes the cost of using theta as the
%   	parameter for linear regression to fit the data points in X and y
%   X: is the m*n matrix contains m samples with n features
%   y: is the m*1 vecter contain value of training samples
%   theta: is the (n+1)*1 vector contains the current theta value.
%  
%   Note: if the dimension of X,y,theta is not correct (X,Y dim-inversed), we will correct it
%-------------------------------------------------------------------------------------

function J = computeCostLinear(X, y, theta)

	try 
		[X,y,theta] = paramAlign(X,y,theta);

		m = length(y);
		J = sum(((X*theta) - y) .^ 2) / (2*m);
	catch err
		J = -1;
		throw(err);
end