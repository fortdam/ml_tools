function result = classifyNorm( x, ...
								mu, ...
								sigma, ...
								theta )
%-------------------------------------------------------------------------
% classifyNorm classify the value with trained weights+bias(theta) and
% normalized factors
%   [x] = the sample to predict
%   [mu] = the mean value for the feature (calc by featureNormalize)
%   [sigma] = the std variance for the feature (calc by featureNormalize)
%   [theta] = the theta/classifier trained
%
%   [result]: the value vector predited
%-------------------------------------------------------------------------

    result = zeros(1, size(theta,2));
    
    for iter=1:size(theta,2)
        result(iter) = predictNorm(x, mu, sigma, theta(:,iter), 'logistic');
    end
end

