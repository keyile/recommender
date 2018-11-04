function rmse = RMSE(P, Y, R)
%RMSE Compute RMSE function
%   rmse = RMSE(z) computes the RMSE of prediction P, real ratings Y, and R.


% ====================== RMSE =================================

num_records = sum(R(:));

rmse = sqrt( sum( ((P - Y) .* R)(:) .^ 2 ) /  num_records );

% =============================================================

end
