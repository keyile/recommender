%% Recommender alogrithm based on matrix factorization


%% Initialization
clear ; close all; clc


%% =============== Loading movie ratings dataset ================
fprintf('Loading movie ratings dataset.\n\n');

%  Load data
data = load('dataset/ml-100k/u.data');
##  u.data     -- The full u data set, 100000 ratings by 943 users on 1682 items.
##                Each user has rated at least 20 movies.  Users and items are
##                numbered consecutively from 1.  The data is randomly
##                ordered. This is a tab separated list of 
##                      user id | item id | rating | timestamp. 
##                The time stamps are unix seconds since 1/1/1970 UTC   



%% =============== Converting rating records to matrix ================
%  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies on 
%  943 users
%
%  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
%  rating to movie i
fprintf('Converting rating records to matrix.\n\n');

Y = zeros(1682, 943); R = zeros(1682, 943);
for iter = 1:size(data, 1)
  r = data(iter, :);
  u = r(1); i = r(2);
  Y(i, u) = r(3);
  R(i, u) = 1;
endfor

%% ================== Learning Movie Ratings ====================
fprintf('\nTraining collaborative filtering...\n');
%  Normalize Ratings
[Ynorm, Ymean] = normalizeRatings(Y, R);

%  Useful Values
num_users = size(Y, 2);
num_movies = size(Y, 1);
num_features = 10;

% Set Initial Parameters (Theta, X)
X = randn(num_movies, num_features);
Theta = randn(num_users, num_features);

initial_parameters = [X(:); Theta(:)];

% Set options for fmincg
options = optimset('GradObj', 'on', 'MaxIter', 100);

% Set Regularization
lambda = 10;
theta = fmincg (@(t)(cofiCostFunc(t, Ynorm, R, num_users, num_movies, ...
                                num_features, lambda)), ...
                initial_parameters, options);

% Unfold the returned theta back into U and W
X = reshape(theta(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(theta(num_movies*num_features+1:end), ...
                num_users, num_features);

fprintf('Recommender system learning completed.\n');

%% ================== Validation on RMSE ====================
prediction = X * Theta' + Ymean;
rmse = sqrt(sum((prediction .* R - Y)(:) .^ 2) / (num_movies*num_features));
fprintf("RMSE of the training set is: %f\n", rmse);