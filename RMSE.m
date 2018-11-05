function rms = RMSE(P, Y, R)
%RMSE Compute RMSE function
%   rms = RMSE(P, Y, R) computes the RMSE of prediction P, 
%   real data Y, and marker matix R.

% ====================== RMSE =================================

num_records = sum(R(:));

rms = sqrt( sum( ((P - Y) .* R)(:) .^ 2 ) /  num_records );

% =============================================================

end
