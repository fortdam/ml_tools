function [X_norm, mu, sigma] = featureNormalize(X)
%-------------------------------------------------------------------------------------
%featureNormalize Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.
%
%   [X_norm] = the normalized features,it is a m*n array, in which every
%       value is (x-mu)/sigma
%   [mu] = the mean value of the features, it is a 1*n vector
%   [sigma] = the sigma(standard difference value) of the features, it is a 1*n vector
%
%   [X]: the input features of the training samples
%-------------------------------------------------------------------------------------


	trans = false;

	if size(X,1) < size(X,2)
		X = X';
		trans = true;
	end

	X_norm = X;

	mu = mean(X); 
	sigma = std(X);

	X_norm = X - repmat(mu, size(X,1), 1);
	X_norm = X_norm ./ repmat(sigma,size(X,1), 1); %Using std deviation instead of MAX-MIN

	if trans
		X_norm = X_norm';
		mu = mu';
		sigma = sigma';
	end
end