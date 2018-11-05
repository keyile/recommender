function rms = RMSE(P, Y, R)
%RMSE Compute Root Mean Square Error.
%   rms = RMSE(P, Y, R) computes the RMSE of predict ratings P, 
%   real data Y, and marker matix R.

% ====================== RMSE =================================

num_records = sum(R(:));

rms = sqrt( sum( ((P - Y) .* R)(:) .^ 2 ) /  num_records );

% =============================================================

end
