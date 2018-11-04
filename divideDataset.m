function [Y, R, Y_val, R_val] = divideDataset(data, num_users, ...
                                                num_movies, train_ratio)
%RMSE Compute RMSE function
%   rmse = RMSE(z) computes the RMSE of prediction P, real ratings Y, and R.


% ====================== divideDataset ========================

% Useful values
num_records = size(data, 1);
num_train = int32(train_ratio * num_records);

% Initilize the training and validating set
Y = zeros(num_movies, num_users);
R = zeros(num_movies, num_users);
Y_val = zeros(num_movies, num_users);
R_val = zeros(num_movies, num_users);

idx = randperm(num_records);

for i = 1:num_train
    r = data(idx(i), :);
    ##  user id | item id | rating | timestamp. 
    Y( r(2), r(1) ) = r(3);
    R( r(2), r(1) ) = 1;
end

for i = num_train+1:num_records
    r = data(idx(i), :);
    Y_val( r(2), r(1) ) = r(3);
    R_val( r(2), r(1) ) = 1;
end
% =============================================================

end
