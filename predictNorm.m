function result = predictNorm( x, ...
                               mu, ...
                               sigma, ...
                               theta, ...
                               ctype )
%-------------------------------------------------------------------------
% predictNorm predict the value with trained weights+bias(theta) and
% normalized factors
%   [x] = the sample to predict
%   [mu] = the mean value for the feature (calc by featureNormalize)
%   [sigma] = the std variance for the feature (calc by featureNormalize)
%   [theta] = the theta/classifier trained
%   [ctype] = 'linear' or 'logistic'
%
%   [result]: the value predited
%-------------------------------------------------------------------------

    if size(x,1) > size(x,2)
        x = x'; %should be a horizontal vector
    end

    if length(x) == (length(theta)-1)
        x = [1, x(1:end)];  %Adding an initial 1 for bias
    elseif length(x) == length(theta)
        x(1) = 1;  %auto-correct the bias factor
    else
        throw(MException('AcctError:Incomplete', 'The size of x(features) is not correct'));
    end

    x(2:end) = (x(2:end)-mu)./sigma;

    if strcmpi(ctype, 'logistic')
        result = sigmoid(x*theta);
    else
        result = x*theta;
    end
  
end

