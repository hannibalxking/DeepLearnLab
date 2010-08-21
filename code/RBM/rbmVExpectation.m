function [ v_expected ] = rbmVExpectation( rbm, hidden )
% rbmVExpectation( rbm, hidden ) computes the expectation of the visible
% layer given the hidden layer
% 
% INPUTS:
%   rbm.........: a rbm model instantiated by createRBM
%   hidden......: an (#examples x #hidden) matrix that holds a batch of
%                   hidden samples
%
% OUTPUTS:
%   v_expected..: a (#examples x #visible) matrix of the expectation of the
%                 visible layer  

% Compute the expectation of the visisble layer
%if rbm.binary
%    v_expected = sigmoid(hidden * rbm.W + repmat(rbm.v_b, size(hidden,1), 1));
%else
    v_expected = hidden * rbm.W; %+ repmat(rbm.v_b, size(hidden,1),1);
%end

end

