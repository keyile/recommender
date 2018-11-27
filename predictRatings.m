function P = predictRatings(X, Theta, mu)
%PREDICTRATINGS Predict the ratings using trained model
%   P = predictRatings(X, Theta, mu) predicts the new ratings to a matrix P,
%   with the input parameter X, Theta and mu.


P = X(:, 2:end) * Theta(:, 2:end)' ...  % compute q_i * p_u
    + X(:, 1)                      ...  % add b_i on every row
    + Theta(:, 1)'                 ...  % add b_u on every column
    + mu;                               % add global bias mu


end
