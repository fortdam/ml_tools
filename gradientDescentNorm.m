function [predictFunc, J_history] = gradientDescentNorm(X, y, theta, lambda, type, alpha, num_iters)
%-------------------------------------------------------------------------
% gradientDescentNorm is similar to gradientDescent(), however it will
% perform feature normalization for the training samples. Instead of
% returning the theta vector, it returns a predict function, in which it
% will convert the input sample(to predict) to normalized format and
% predict the result
%-------------------------------------------------------------------------


  if size(X,1)<size(X,2)
      X= X';
  end
  
  [X, mu, sigma] = featureNormalize(X);
  
  [c_theta, J_history] = gradientDescent(X, y, theta, lambda, type, alpha, num_iters);
  
  function result = predict(x)
     if size(x,1) > size(x,2)
         x = x'; %should be a horizontal vector
     end
     
     if length(x) == (length(c_theta)-1)
         x = [1, x(1:end)];  %Adding an initial 1 for bias
     elseif length(x) == length(c_theta)
         x(1) = 1;  %auto-correct the bias factor
     else
         throw(MException('AcctError:Incomplete', 'The size of x(features) is not correct'));
     end
     
     x(2:end) = (x(2:end)-mu)./sigma;
     
     if strcmpi(type, 'logistic')
         result = sigmoid(x*c_theta);
     else
         result = x*c_theta;
     end
  end
  
  predictFunc = @predict;
end

