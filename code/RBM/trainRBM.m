function [rbm ] = trainRBM( rbm, data, batch_size, nepochs, learn_rate)
% trainRBM( rbm, data, batch_size, nepochs, learn_rate) trains an
% rbm based on the given parameters
%
% INPUTS:
%   rbm.........: an rbm created by createRBM
%   data........: (n x p) data matrix
%   batch_size..: number of examples in each batch
%   nepochs.....: number of times to go through the data
%   learn_rate..: the learning rate of the training.
%
% OUTPUTS:
%   rbm.........: the trained rbm


ndata = length(data);

% Create a random permutation of the data
perm = randperm(ndata);
data = data(perm,:);

nbatches = ceil(ndata/batch_size);

q_old = -1;

for e=1:nepochs
    tic
    for b=0:nbatches-1
        
        [w,v,h, q_old] = rbmGradients(rbm, data(b*batch_size+1:...
                            min((b+1)*batch_size,ndata),:), 1, q_old);
                              

        rbm.W = rbm.W + learn_rate * w;
        if rbm.binary
            rbm.v_b = rbm.v_b + learn_rate * v;
        end
        rbm.h_b = rbm.h_b + learn_rate * h;

    end
    toc
    
    

end


end

