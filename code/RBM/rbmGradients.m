function [ w_g, v_g, h_g, q_current ] = rbmGradients( rbm, batch, k, q_old)
% rbmGradients( rbm, batch, k, q_old) computes the gradients on a batch of
% examples used for learning by constrative divergence
%
% INPUTS:
%   rbm.....: a rbm model instantiated by createRBM
%   batch...: an (#examples x #features) matrix that holds a batch of
%             examples
%   k.......: the number of iterations to run constrastive divergence 
%   q_old...: a (1 x nfeatures) vector of the previous sparsity estimate
%             of the hidden states
%
% OUTPUTS:
%   w_g.........: a (#hidden x #visisble) matrix of the weight gradients
%   v_g.........: a (1 x#visible) vector of the visible bias gradients
%   h_g.........: a (1 x #hidden) vector of the hidden bias gradients
%   q_current...: a (1 x #hidden) vector of the updated hidden bias
%                 estimate


batch_size = size(batch,1);

% Compute the hidden and visible samples needed for constrastive divergence
[h_0, v_k, h_k] = rbmSampleK(rbm, batch, k);


E_0 = zeros(rbm.n_h, rbm.n_v);
E_k = zeros(rbm.n_h, rbm.n_v);

% Sum the expectations of the hidden and visible layer 
% Note:
%   try arrayfun and evaluate if it is faster
for i=1:batch_size
    E_0 = E_0 + h_0(i,:)' * batch(i,:);
    E_k = E_k + h_k(i,:)' * v_k(i,:);
end

% Calculate the current hidden bias estimate
% We are only updating the hidden biases.
% Note:
%   Hinton's practical guide suggests to look into updating W as well 
q_current = sum(h_0, 1)/batch_size;


% Check to see if there was a previous hidden bias estimate 
if sum(q_old) < 0
    q_old = q_current;
end

sparsity_grad = 0;
if rbm.sparsity > 0
    % q_current is estimated from sample of hidden units. Could be weighted
    % sum of their expectation.
    sparsity_grad = rbm.sparsity - (rbm.sparsity_decay * q_old ...
                    + (1 - rbm.sparsity_decay)*q_current);
end

% Compute the average gradients of the batch
w_g = (E_0 - E_k)./batch_size;
v_g = (sum(batch,1) - sum(v_k,1))/batch_size;
h_g = (sum(h_0,1) - sum(h_k,1))/batch_size + sparsity_grad;

end

