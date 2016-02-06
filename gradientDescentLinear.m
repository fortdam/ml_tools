function [theta, J_history] = gradientDescentLinear(X, y, theta, alpha, num_iters)

  [X, y, theta] = paramAlign(X,y,theta);
  m = length(y);

  for iter = 1:num_iters
    theta = theta - (alpha * (X' * ((X * theta) - y))/m);
    J_history(iter) = computeCostLinear(X, y, theta);
  end
end
