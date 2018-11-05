function [X, Theta, J_history] = SGDTrain(X, Theta, Y, R, lambda, alpha, ...
                               num_iters)
%SGDTrain Performs stochastic gradient descent to learn X and Theta
%   [X, Theta, J_history] = SGDTrain(X, Theta, Y, R, lambda, alpha, num_iters) 
%   updates X and Theta by taking num_iters gradient steps with learning rate
%   alpha. lambda is the regulization parameter.

% Initialize some useful values
num_users = size(Theta, 1);
num_movies = size(X, 1);
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== Loop body ======================
    % Perform a single gradient step on the parameter X and Theta. 
    %

    % compute the error matrix
    E = (X * Theta' - Y) .* R;

    % random choose some user and movie
    rand_user = randi(num_users);
    rand_movie = randi(num_movies);

    % update X
    X_grad = E(:, rand_user) * Theta(rand_user, :) + lambda * X;
    X = X - alpha * X_grad;

    % update Theta
    Theta_grad = E(rand_movie, :)' * X(rand_movie, :) + lambda * Theta;
    Theta = Theta - alpha * Theta_grad;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = sum((E .^ 2)(:)) ...
        + lambda * ( sum((Theta .^ 2)(:)) + sum((X .^ 2)(:)) );

end

end
