function [X, Theta, J_history] = sgdTrain(X, Theta, Y, R, lambda, alpha, ...
                               num_iters)
%sgdTrain Performs stochastic gradient descent to learn X and Theta
%   [X, Theta, J_history] = sgdTrain(X, Theta, Y, R, lambda, alpha, num_iters) 
%   updates X and Theta by taking num_iters gradient steps with learning rate
%   alpha. lambda is the regulization parameter.

% Initialize some useful values
num_users = size(Theta, 1);
num_movies = size(X, 1);
J_history = zeros(num_iters, 1);

% Initialize temporary variables to save the gradient
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

for iter = 1:num_iters

    % ====================== Loop body ======================
    % Perform a single gradient step on the parameter X and Theta. 
    %

    % Compute the error matrix
    % have added bu and bi on the first column, so use (2:end) 
    % to retrieve pu and qi matrix
    E = (X(:, 2:end) * Theta(:, 2:end)' + X(:, 1) + Theta(:, 1)' - Y) .* R;


    % random choose some user and movie
    rand_user = randi(num_users);
    rand_movie = randi(num_movies);

    % update X
    X_grad(:, 2:end) = E(:, rand_user) * Theta(rand_user, 2:end) ...
                        + lambda * X(:, 2:end);
    X_grad(:, 1) = E(:, rand_user) + lambda * X(:, 1);
    X = X - alpha * X_grad;

    % update Theta
    Theta_grad(:, 2:end) = E(rand_movie, :)' * X(rand_movie, 2:end) ...
                        + lambda * Theta(:, 2:end);
    Theta_grad(:, 1) = E(rand_movie, :)' + lambda * Theta(:, 1);
    Theta = Theta - alpha * Theta_grad;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = sum((E .^ 2)(:)) ...
        + lambda * ( sum((Theta .^ 2)(:)) + sum((X .^ 2)(:)) );

end

end
