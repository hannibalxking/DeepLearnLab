function [ C ] = convcut( A, B )
%CONVCUT Summary of this function goes here
%   Detailed explanation goes here

A_dim = size(A);
B_dim = size(B);

C = imfilter(A,B);

% todo generalize
if length(B_dim) == 2
    C(A_dim(1) - B_dim(1) + 2:end, :,:) = [];
    C(:, A_dim(2) - B_dim(2) + 2:end,:) = [];
else
    C(A_dim(1) - B_dim(1) + 2:end, :,:,:) = [];
    C(:, A_dim(2) - B_dim(2) + 2:end,:,:) = [];
    C(:, :,A_dim(3) - B_dim(3) + 2:end,:) = [];
end


end

