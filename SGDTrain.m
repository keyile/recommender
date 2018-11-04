function [X, Theta, J_history] = SGDTrain(X, Theta, Y, R, lambda, gamma, ...
                               num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
num_users = size(Theta, 1);
num_movies = size(X, 1);
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== Loop body ======================
    % Perform a single gradient step on the parameter vector theta. 
    %

    E = (X * Theta' - Y) .* R;
    rand_user = randi(num_users);
    rand_movie = randi(num_movies);

    for i = 1:size(X, 1)
        X_grad = E(i, rand_user) * Theta(rand_user, :) + lambda * X(i, :);
        X(i, :) -= gamma * X_grad;
    end

    for j = 1:size(Theta, 1)
        Theta_grad = E(rand_movie, j) * X(rand_movie, :) + lambda * Theta(j, :);
        Theta(j, :) -= gamma * Theta_grad; 
    end

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = sum((E .^ 2)(:)) ...
        + lambda * ( sum((Theta .^ 2)(:)) + sum((X .^ 2)(:)) );

end

end
