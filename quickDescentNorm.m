function [ predictFunc, cost ] = quickDescentNorm( X, y, initial_theta, type, lambda)
%-------------------------------------------------------------------------
% quickDescentNorm is similar to quickDescent(), however it will
% perform feature normalization for the training samples. Instead of
% returning the theta vector, it returns a predict function, in which it
% will convert the input sample(to predict) to normalized format and
% predict the result
%-------------------------------------------------------------------------

    [X, mu, sigma] = featureNormalize(X);
    
    options = optimset('GradObj', 'on'); %options = optimset('GradObj', 'on', 'MaxIter', 400);
    [theta, cost] = fminunc(@(t)computeCost(X, y, t, lambda,type),initial_theta,options);
    
    predictFunc = @(xx)predictNorm(xx, mu, sigma, theta, type);
end

