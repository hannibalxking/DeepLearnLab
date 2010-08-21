function [ rbm ] = trainRBMMNIST(data_loc)

mnist = load(data_loc);

data = zeros(0, 784);

for i=0:9
    data = vertcat(data, getfield(mnist, ['train' int2str(i)]));
end

data = double(data>128);

ndata = length(data);
rbm = createRBM(784, 100, 'binary', .001, .001 , .1);

% Create a random permutation of the data
perm = randperm(ndata);
data = data(perm,:);

batch_size = 20;
learn_rate = .0001;
nepochs = 20;
nbatches = ceil(ndata/batch_size);

q_old = -1;

for e=1:nepochs
    tic
    for b=0:nbatches-1
        
        [w,v,h, q_old] = rbmGradients(rbm, data(b*batch_size+1:...
                            min((b+1)*batch_size,ndata),:), 1, q_old);
                              
        
        rbm.W = rbm.W + learn_rate * w;
        rbm.v_b = rbm.v_b + learn_rate * v;
        rbm.h_b = rbm.h_b + learn_rate * h;

    end
    toc
    
    figure(e);
    im = rbmVisualize(rbm, 28, 28, 10,10,1);
    imshow(im);
    

end



end

