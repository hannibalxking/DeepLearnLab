function [ W_k ] = kthMap(W,k)

if length(size(W)) == 3
    W_k = W(:,:,k);
else
    W_k = W(:,:,:,k);
end

end

