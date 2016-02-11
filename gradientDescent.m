function [theta, J_history] = gradientDescent(X, y, theta, lambda, type, alpha, num_iters)

  [X, y, theta] = paramAlign(X,y,theta);
  m = length(y);

  for iter = 1:num_iters
    [J_history(iter), grad] = computeCost(X, y, theta, lambda, type);
    theta = theta - alpha * grad;
  end
end
