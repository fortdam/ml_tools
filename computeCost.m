function [J, grad] = computeCost(X, y, theta, lambda, type)
%-------------------------------------------------------------------------------------
% ComputeCost Compute cost for linear/logistic regression with single/multiple variables
%   [J] = computeCostLinear(X, y, theta,type) computes the cost of using theta as the
%   	parameter for linear regression to fit the data points in X and y
%   [grad] = the gradient(partial direvatives) of the func
%
%   [X]: is the m*n matrix contains m samples with n features
%   [y]: is the m*1 vecter contain value of training samples
%   [theta]: is the (n+1)*1 vector contains the current/initial theta value. 
%   [lambda]: is the regularization param
%   [type]: "linear" or "logistic"
%  
%   Note: if the dimension of X,y,theta is not correct (X,Y dim-inversed),
%   we will correct it.
%-------------------------------------------------------------------------------------
    
    %Align the input data first
    [X,y,theta] = paramAlign(X,y,theta);

    m = length(y);

    if strcmpi(type, 'logistic')
        hypothesis = sigmoid(X*theta);
        J =  sum(-1*(y.*log(hypothesis) + (1-y).*log(1-hypothesis)))/m;
    else 
        hypothesis = X*theta;  %Linear regresssion
        J = sum((hypothesis - y) .^ 2) / (2*m);
    end

    grad = (X' * (hypothesis - y))/m;

    %Regularization
    regTheta = [0; theta(2:end,:)];  %Regularization theta, ignore the first value

    J = J + lambda*sum(regTheta .^ 2)/(2*m);
    grad = grad + (lambda/m)*regTheta;

end