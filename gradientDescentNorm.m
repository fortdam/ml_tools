function [predictFunc, J_history] = gradientDescentNorm(X, y, theta, lambda, type, alpha, num_iters)
%gradientDescentNorm Summary of this function goes here
%   Detailed explanation goes here

  if size(X,1)<size(X,2)
      X= X';
  end
  
  [X, mu, sigma] = featureNormalize(X);
  
  [c_theta, J_history] = gradientDescent(X, y, theta, lambda, type, alpha, num_iters);
  
  function result = predict(x)
     if size(x,1) > size(x,2)
         x = x'; %should be a horizontal vector
     end
     x = (x-mu)./sigma;
     x = [1,x(1:end)];
     
     if strcmpi(type, 'logistic')
         result = sigmoid(x*c_theta);
     else
         result = x*c_theta;
     end
  end
  
  predictFunc = @predict;
end

