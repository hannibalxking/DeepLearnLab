function [ h_0, v_k, h_k ] = rbmSampleK(rbm, visible, k)
%rbmSampleK(rbm, visible, k) runs k interations of alternately sampling the hidden and
% visible layer
%   
% INPUTS:
%   rbm.........: a rbm model instantiated by createRBM 
%   visible.....: a (#examples x #visible) matrix of a batch of examples
%   k...........: number of iterations to perform alternate sampling
%                
% OUTPUTS:
%   h_0.....: a (#examples x #hidden) matrix that contains the
%             initial sample of the hidden layer
%   v_k.....: a (#examples x #visible) matrix that contains the
%             kth samples of the visible layer
%   h_k.....: a (#examples x #hidden) matrix that contains the
%             kth sample of the hidden layer  

% Compute the initial hidden sample
h_0 = bernoulli_sample(rbmHExpectation(rbm,visible));

% Alternately sample the visible and hidden layers k  times 
h_k = h_0;
for i=1:k
    v_k = rbmVExpectation(rbm,h_k);
    h_k = rbmHExpectation(rbm, v_k);
end

end

