%% Recommender system alogrithm based on matrix factorization

%% This is the main executable specified for the *ml-1m* dataset,
%% and contains the follow parameters:
%%     train_ratio  : the devision ratio of training and validating set
%%     num_features : the dimension of the latent factor
%%     lambda       : the regularization term
%%     alpha        : the learning rate
%%     num_iters    : the number of epoches, as you guess

%% Initialization
clear ; close all; clc


%% =============== Loading movie ratings dataset ================
fprintf('Loading movie ratings dataset...\n');

% set dataset info
filename = 'dataset/ml-1m/ratings.dat';
num_users = 6040;
num_movies = 3952;
% and how you'd like to divide the dataset
train_ratio = 0.8;

%  Load data
data = dlmread(filename, ':');
data = data(:, [1, 3, 5, 7]);
%% All ratings are contained in the file "ratings.dat" and are in the
%% following format:
%% 
%% UserID::MovieID::Rating::Timestamp
%% 
%% - UserIDs range between 1 and 6040 
%% - MovieIDs range between 1 and 3952
%% - Ratings are made on a 5-star scale (whole-star ratings only)
%% - Timestamp is represented in seconds since the epoch as returned by time(2)
%% - Each user has at least 20 ratings

fprintf('First five of the data:\n');
display(data(1:5, :));
fprintf('Number of ratings records: %d.\n', size(data, 1));

%% =============== Converting rating records to matrix ================
%  Y is a 3952x6040 matrix, containing ratings (1-5) of 3952 movies on 
%  6040 users
%
%  R is a 3952x6040 matrix, where R(i,j) = 1 if and only if in the training set
%  user j gave a rating to movie i
%
%  R_val is a 3952x6040 matrix, where R_val(i,j) = 1 if and only if in the validating set
%  user j gave a rating to movie i

fprintf('\nConverting rating records to matrix...\n');

% Training set and validating set use the common matrix Y, and use R and R_val to
% mark if some rating exsits.

[Y, R, R_val] = divideDataset(data, num_users, num_movies, train_ratio);

fprintf('Dataset is divided into training set and validating set,\n');
fprintf('size: %d / %d.\n', sum(R(:)), sum(R_val(:)) );

%% ================== Learning Movie Ratings ====================
fprintf('\nTraining collaborative filtering...\n');
%  Normalize Ratings
%  mu -- global average bias
[Ynorm, mu] = normalizeRatings(Y, R);

% Set parameters
num_features = 30;
lambda = 0.20;
alpha = 0.05;
num_iters = 1;
fprintf('Learning parameters:\n');
fprintf('     num_features = %d\n', num_features);
fprintf('     lambda       = %.3f\n', lambda);
fprintf('     alpha        = %.3f\n', alpha);
fprintf('     num_iters    = %d\n', num_iters);

% Set Initial Parameters (Theta, X)
% first column is bi and bu -- item bias and user bias
X = randn(num_movies, num_features + 1);
Theta = randn(num_users, num_features + 1);

fprintf('...\n');
[X, Theta, J_history] = sgdTrain(X, Theta, Ynorm, R, lambda, alpha, num_iters);
plot(1:length(J_history), J_history);

fprintf('Recommender system learning completed.\n');

%% ================== Validation by RMSE ====================
fprintf('\nValidation by RMSE...\n');

P = predictRatings(X, Theta, mu);
fprintf("RMSE on the training   set is: %f, \n", rmse(P, Y, R));
fprintf("     on the validating set is: %f\n", rmse(P, Y, R_val));
