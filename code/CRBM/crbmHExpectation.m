function [ h_expected ] = crbmHExpectation( crbm, visible )

% V is (1xn) a cell array with each cell holding an input. n input in all
s = size(visible);
n = s(end);

% MATLAB suggests this to reduce overhead in parfor.
W = crbm.W;
hbias = crbm.h_b;

h_expected = zeros([crbm.h_dim n]); 


if length(size(W)) == 3
    for k=1:crbm.n_maps
        h_expected(:,:,k,:) =convn(visible, kthMap(W,k)', 'valid') + hbias(1,k);
    end
else
    parfor k=1:crbm.n_maps
        h_expected(:,:,:,k,:) = convn(visible, kthMap(W,k)', 'valid') + hbias(1,k); 
    end
end

h_expected = sigmoid(h_expected);

