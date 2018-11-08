function [Ynorm, mu] = normalizeRatings(Y, R)
%NORMALIZERATINGS Preprocess data by subtracting global average for every 
%rating
%   [Ynorm, mu] = normalizeRatings(Y, R) normalized Y so that everty rating
%   has a value of 0 on average, and returns the mean rating in mu.
%

mu = sum(Y(:)) / sum(R(:));
Ynorm = Y - mu;

end
