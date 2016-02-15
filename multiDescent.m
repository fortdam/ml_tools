function [ predictFunc, cost ] = multiDescent( X, y, lambda, maxIter)
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


	if(size(X,1) < size(X,2))
		X = X';
	end

	if(size(y,1) < size(y,2))
		y = y';
	end

	classes = unique(y);

	theta = zeros(size(X,2)+1, length(classes));
	cost = zeros(length(classes));


	for iter=1:length(classes)
		disp(['Train classifier ', int2str(iter)]);
		[tt, cc] = quickDescent(X, y==classes(iter), 'logistic', lambda, maxIter);
		theta(:,iter) = tt;
		cost(iter) = min(cc);
	end
    
    predictFunc = @(xx)classifyNorm(xx, 0, 1, theta);
end