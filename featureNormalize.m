%-------------------------------------------------------------------------------------
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.
%-------------------------------------------------------------------------------------


function [X_norm, mu, sigma] = featureNormalize(X)

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