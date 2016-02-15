function [predictFunc, J_history] = gradientDescentNorm(X, y, lambda, type, alpha, num_iters)
%-------------------------------------------------------------------------
% gradientDescentNorm is similar to gradientDescent(), however it will
% perform feature normalization for the training samples. Instead of
% returning the theta vector, it returns a predict function, in which it
% will convert the input sample(to predict) to normalized format and
% predict the result
%-------------------------------------------------------------------------
  
  [X, mu, sigma] = featureNormalize(X);
  
  [theta, J_history] = gradientDescent(X, y, zeros(size(X,2)+1,1), lambda, type, alpha, num_iters);
  
  predictFunc = @(xx)predictNorm(xx, mu, sigma, theta, type);
end

