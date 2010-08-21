function [ h_expected ] = rbmHExpectation( rbm, visible )
% rbmHExpectation( rbm, visible ) computes the expectation of the hidden 
% layer given the visible layer
% 
% INPUTS:
%   rbm.........: a rbm model instantiated by createRBM
%   visible.....: an (#examples x #visible) matrix that holds a batch of
%                   visible samples
%
% OUTPUTS:
%   h_expected..: a (#examples x #hidden) matrix of the expectation of the
%                 hidden layer  

% Compute the expectation of the hidden layer
h_expected = sigmoid((rbm.W * visible')' + repmat(rbm.h_b, size(visible,1), 1));

end

