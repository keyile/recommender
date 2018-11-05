%% Recommender alogrithm based on matrix factorization


%% Initialization
clear ; close all; clc


%% =============== Loading movie ratings dataset ================
fprintf('Loading movie ratings dataset...\n');

%  Load data
data = load('dataset/ml-100k/u.data');
##  u.data     -- The full u data set, 100000 ratings by 943 users on 1682 items.
##                Each user has rated at least 20 movies.  Users and items are
##                numbered consecutively from 1.  The data is randomly
##                ordered. This is a tab separated list of 
##                      user id | item id | rating | timestamp. 
##                The time stamps are unix seconds since 1/1/1970 UTC   

fprintf('First five of the data:\n');
display(data(1:5, :));
fprintf('Number of ratings records: %d.\n', size(data, 1));

%% =============== Converting rating records to matrix ================
%  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies on 
%  943 users
%
%  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
%  rating to movie i
fprintf('\nConverting rating records to matrix...\n');

%  Useful Values
num_users = 943;
num_movies = 1682;

% Training set and validating set use the common matrix Y, and use R and R_val to
% mark if some rating exsits.
train_ratio = 0.8;
[Y, R, R_val] = divideDataset(data, num_users, num_movies, train_ratio);

fprintf('Dataset is divided into training set and validating set,\n');
fprintf('size: %d / %d.\n', sum(R(:)), sum(R_val(:)) );

%% ================== Learning Movie Ratings ====================
fprintf('\nTraining collaborative filtering...\n');
%  Normalize Ratings
[Ynorm, Ymean] = normalizeRatings(Y, R);

% Set parameters
num_features = 10;
lambda = 10;
alpha = 0.003;
num_iters = 100;

% Set Initial Parameters (Theta, X)
X = randn(num_movies, num_features);
Theta = randn(num_users, num_features);

[X, Theta, J_history] = SGDTrain(X, Theta, Ynorm, R, lambda, alpha, num_iters);
plot(1:num_iters, J_history);

fprintf('Recommender system learning completed.\n');

%% ================== Validation by RMSE ====================
fprintf('\nValidation by RMSE...\n');

P = X * Theta' + Ymean;
fprintf("RMSE on the training   set is: %f, \n", RMSE(P, Y, R));
fprintf("     on the validating set is: %f\n", RMSE(P, Y, R_val));
