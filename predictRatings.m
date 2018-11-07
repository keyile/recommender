function P = predictRatings(X, Theta, mu)
%PREDICTRATINGS Compute Root Mean Square Error.XXXXXXXXXX
%   P = predictRatings(X, Theta, mu) computes the RMSE of predict ratings P,
%   real data Y, and marker matix R.


P = X(:, 2:end) * Theta(:, 2:end)' + X(:, 1) + Theta(:, 1)' + mu;


end
