%% Recommender alogrithm based on matrix factorization


%% Initialization
clear ; close all; clc


%% =============== Loading movie ratings dataset ================
fprintf('Loading movie ratings dataset...\n');

%  Load data
data = load('dataset/ml-100k/u.data');
num_records = size(data, 1);
##  u.data     -- The full u data set, 100000 ratings by 943 users on 1682 items.
##                Each user has rated at least 20 movies.  Users and items are
##                numbered consecutively from 1.  The data is randomly
##                ordered. This is a tab separated list of 
##                      user id | item id | rating | timestamp. 
##                The time stamps are unix seconds since 1/1/1970 UTC   

fprintf('First five of the data:\n');
display(data(1:5, :));
fprintf('Number of ratings records: %d.\n\n', num_records);

%% =============== Converting rating records to matrix ================
%  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies on 
%  943 users
%
%  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
%  rating to movie i
fprintf('\nConverting rating records to matrix...\n');

Y = zeros(1682, 943); R = zeros(1682, 943);
for iter = 1:size(data, 1)
    r = data(iter, :);
    u = r(1); i = r(2);
    Y(i, u) = r(3);
    R(i, u) = 1;
endfor
fprintf('Matrix is ready, with the size of %d x %d.\n\n', size(Y,1), size(Y,2));

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

% Set Regularization
lambda = 10;
gamma = 0.003;
num_iters = 100;
[X, Theta, J_history] = SGDTrain(X, Theta, Y, R, lambda, gamma, num_iters);
plot(1:num_iters, J_history, 1:num_iters, J_history / 2);

fprintf('Recommender system learning completed.\n');

%% ================== Validation on RMSE ====================
prediction = X * Theta' + Ymean;
rmse = sqrt( sum( ((prediction - Y).*R)(:) .^ 2 ) /  num_records );
fprintf("RMSE of the training set is: %f\n", rmse);