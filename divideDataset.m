function [Y, R1, R2] = divideDataset(data, num_users, ...
                                                num_movies, train_ratio)
%RMSE Compute RMSE function
%   rmse = RMSE(z) computes the RMSE of prediction P, real ratings Y, and R1.


% ====================== divideDataset ========================

% Useful values
num_records = size(data, 1);
num_train = int32(train_ratio * num_records);

% Initilize the training and validating set
Y = zeros(num_movies, num_users);
R1 = zeros(num_movies, num_users);
R2 = zeros(num_movies, num_users);

idx = randperm(num_records);

for i = 1:num_train
    r = data(idx(i), :);
    ##  user id | item id | rating | timestamp. 
    Y( r(2), r(1) ) = r(3);
    R1( r(2), r(1) ) = 1;
end

for i = num_train+1:num_records
    r = data(idx(i), :);
    Y( r(2), r(1) ) = r(3);
    R2( r(2), r(1) ) = 1;
end
% =============================================================

end
