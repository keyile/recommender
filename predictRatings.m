function P = predictRatings(X, Theta, mu)
%PREDICTRATINGS Predict the ratings using trained model
%   P = predictRatings(X, Theta, mu) predicts the new ratings to a matrix P,
%   with the input parameter X, Theta and mu.


P = X(:, 2:end) * Theta(:, 2:end)' + X(:, 1) + Theta(:, 1)' + mu;


end
