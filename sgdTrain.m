function [X, Theta, J_history] = sgdTrain(X, Theta, Y, R, lambda, alpha, ...
                               num_iters)
%SGDTrain Performs stochastic gradient descent to learn X and Theta
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

% random choose some users and movies
% Note that we choose random number only once, then just increse the number
% in every iteration.
rand_user = randi(num_users);
rand_movie = randi(num_movies);

for iter = 1:num_iters

    % ====================== Loop body ======================
    % Perform a single gradient step on the parameter X and Theta. 
    %

    % Compute the error matrix
    % have added bu and bi on the first column, so use (2:end) 
    % to retrieve pu and qi matrix
    E = (
        X(:, 2:end) * Theta(:, 2:end)' % compute q_i * p_u
        + X(:, 1)                      % add b_i on every row
        + Theta(:, 1)'                 % add b_u on every column
        - Y                            % difference from real ratings
        ).* R;                         % cast the mask


    % compute the gradient of X and Theta
    % gradient of x_i
    X_grad(:, 2:end) = E(:, rand_user) * Theta(rand_user, 2:end) ...
                        + lambda * X(:, 2:end);

    % gradient of x_1
    X_grad(:, 1) = E(:, rand_user) + lambda * X(:, 1);

    % gradient of theta_i
    Theta_grad(:, 2:end) = E(rand_movie, :)' * X(rand_movie, 2:end) ...
                        + lambda * Theta(:, 2:end);

    % gradient of theta_1
    Theta_grad(:, 1) = E(rand_movie, :)' + lambda * Theta(:, 1);


    % update X and Theta
    X     = X     - alpha * X_grad;
    Theta = Theta - alpha * Theta_grad;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = sum((E .^ 2)(:)) ...
        + lambda * ( sum((Theta .^ 2)(:)) + sum((X .^ 2)(:)) );

    % increase the random number and keep the number between 1 and total number
    rand_user  = mod(rand_user, num_users)   + 1;
    rand_movie = mod(rand_movie, num_movies) + 1;
    % -- mod (X, Y)
    % Compute the modulo of X and Y.
end

end
