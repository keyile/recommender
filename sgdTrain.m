function [X, Theta, J_history] = sgdTrain(X, Theta, Y, R, lambda, alpha, ...
                                          num_iters)
%SGDTrain Performs stochastic gradient descent to learn X and Theta
%   [X, Theta, J_history] = sgdTrain(X, Theta, Y, R, lambda, alpha, num_iters) 
%   updates X and Theta by taking num_iters gradient steps with learning rate
%   alpha. lambda is the regulization parameter.


% Initialize some useful values
num_users = size(Theta, 1);
num_movies = size(X, 1);
num_records = sum(R(:));

% Variables related to probing
SAMPLING = 10000;   % How often when we compute the cost function
J_history = zeros(num_iters * fix(num_records / SAMPLING), 1);
J_iter = 1;

% Initialize temporary variables to save the gradient
dX_i     = zeros(1, size(X, 2));
dTheta_i = zeros(1, size(Theta, 2));

% Find the ratings records in the matrix, and save the coordinates in [i, j]
[i, j] = find(R);       % size(i) == (num_records, 1)

for iter = 1:num_iters
    % Randomly choose an index vector
    idx_vec = randperm(num_records);

    for idx = 1:num_records
        % Get the next pair of rand_movie and rand_user
        rand_movie = i(idx_vec(idx));
        rand_user =  j(idx_vec(idx));

        % ========================= Updater =========================
        % Perform a single gradient step on the parameter X and Theta.
        % Use the chosen rand_movie and rand_movie.

        % First, compute the error on this training case
        err = X(rand_movie, 2:end) * Theta(rand_user, 2:end)' ...
                + X(rand_movie, 1) ...
                + Theta(rand_user, 1) ...
                - Y(rand_movie, rand_user);


        % Second, compute the gradient of X and Theta
        % gradient of x_i
        dX_i(2:end) = err * Theta(rand_user, 2:end) ...
                            + lambda * X(rand_movie, 2:end);

        % gradient of x_1
        dX_i(1) = err + lambda * X(rand_movie, 1);

        % gradient of theta_i
        dTheta_i(2:end) = err * X(rand_movie, 2:end) ...
                            + lambda * Theta(rand_user, 2:end);

        % gradient of theta_1
        dTheta_i(1) = err + lambda * Theta(rand_user, 1);


        % Then, update X and Theta
        X(rand_movie, :)    = X(rand_movie, :)    - alpha * dX_i;
        Theta(rand_user, :) = Theta(rand_user, :) - alpha * dTheta_i;

        % ==================== Updater End ==========================

        % Compute the cost function for probing use
        if mod(idx, SAMPLING) == 0
            % Compute the error matrix
            E = (
            X(:, 2:end) * Theta(:, 2:end)' % compute q_i * p_u
            + X(:, 1)                      % add b_i on every row
            + Theta(:, 1)'                 % add b_u on every column
            - Y                            % difference from real ratings
            ).* R;                         % cast the mask

            % Save the cost J in this iteration
            J_history(J_iter) = sum((E .^ 2)(:)) ...
                + lambda * ( sum((Theta .^ 2)(:)) + sum((X .^ 2)(:)) );

            % Yes, increase the iterator
            J_iter = J_iter + 1;
        end

    end

end

end
