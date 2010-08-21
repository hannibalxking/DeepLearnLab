function [ im ] = crbmVisualize(e,crbm, n_high, n_wide, start_h )

h = crbm.w_dim(1);
w = crbm.w_dim(2);

%im = zeros(n_high * h, n_wide * w);
W = (crbm.W - min(min(min(crbm.W))))/(max(max(max(crbm.W))) - min(min(min(crbm.W))));

k=1;
figure(e);
for m=1:n_high
    for r=1:n_wide
        subplot(n_high, n_wide, k);
        imshow(W(:,:,k));
        k=k+1;
    end
end
end




