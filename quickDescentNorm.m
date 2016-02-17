function [ predictFunc, J_history ] = quickDescentNorm( X, ...
														y, ...
														ctype, ...
														lambda, ...
														maxIter )
%-------------------------------------------------------------------------
% quickDescentNorm is similar to quickDescent(), however it will
% perform feature normalization for the training samples. Instead of
% returning the theta vector, it returns a predict function, in which it
% will convert the input sample(to predict) to normalized format and
% predict the result
%-------------------------------------------------------------------------

    [X, mu, sigma] = featureNormalize(X);

    options = optimset('GradObj', 'on', 'MaxIter', maxIter); %options = optimset('GradObj', 'on', 'MaxIter', 400);
    [theta, J_history] = fmincg(@(t)computeCost(X, y, t, lambda, ctype),zeros(size(X,2)+1,1),options);  %Might use fminunc to replace fmincg
    
    predictFunc = @(xx)predictNorm(xx, mu, sigma, theta, ctype);
end

