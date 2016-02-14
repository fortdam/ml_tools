function [ theta, cost ] = quickDescent( X, y, initial_theta, type, lambda)
%-------------------------------------------------------------------------
% quickDescent performs the advanced descent (fminunc)
%   [theta] = the final theta calculatd out
%   [cost] = the cost value with the given theta
%
%   [X]: is the m*n matrix contains m samples with n features
%   [y]: is the m*1 vecter contain value of training samples
%   [initial_theta]: is the (n+1)*1 vector contains the current/initial theta value. 
%   [lambda]: is the regularization param
%   [type]: "linear" or "logistic"
%  
%   Note: if the dimension of X,y,theta is not correct (X,Y dim-inversed), we will correct it
%-------------------------------------------------------------------------
    
    options = optimset('GradObj', 'on'); %options = optimset('GradObj', 'on', 'MaxIter', 400);
    [theta, cost] = fminunc(@(t)computeCost(X, y, t, lambda,type),initial_theta,options)
end

