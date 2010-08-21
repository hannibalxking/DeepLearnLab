function [ crbm ] = trainCRBMMNIST(data_loc, crbm)

mnist = load(data_loc);

data_ = zeros(0, 784);

for i=0:9
    data_ = vertcat(data_, getfield(mnist, ['train' int2str(i)]));
end

data_ = round(double(data_)/255);
ndata = length(data_);

data = zeros(28,28, ndata);

for i=1:ndata
    data(:,:,i) = reshape(data_(i,:), 28,28)';
end

if nargin < 2
    crbm = createCRBM(40,'binary', .01, [28 28],[12 12], [2 2], .1 , .5);
end

% Create a random permutation of the data
perm = randperm(ndata);
data = data(:,:, perm);

batch_size = 20;
learn_rate = .1;
nepochs = 15;
nbatches = ceil(ndata/batch_size);

q_old = -1;

for e=1:nepochs
    tic
    for b=0:nbatches-1
        
        [w,v,h, q_old] = crbmGradients(crbm, data(:,:,b*batch_size+1:...
                            min((b+1)*batch_size,ndata)), 1, q_old);
                  
        crbm.W = crbm.W + learn_rate * w;
        crbm.v_b = crbm.v_b + learn_rate * v;
        crbm.h_b = crbm.h_b + learn_rate * h;


    end
    toc;
    %{
    figure(e);
    im = crbmVisualize(crbm, 28, 28, 10,10,1);
    imshow(im);
    %}

end



end

