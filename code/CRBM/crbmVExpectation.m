function [ v_expected ] = crbmVExpectation( crbm, hidden )


s = size(hidden);

W = crbm.W;



k_dim = length(size(W));



%v_expected_k = cell(1, crbm.n_maps);


%for k=1:crbm.n_maps

v_expected = zeros([crbm.v_dim s(end)]);


if k_dim == 3
    parfor k=1:crbm.n_maps
        v_expected = v_expected + squeeze(convn(hidden(:,:,k,:), kthMap(W,k), 'full'));
    end
else
    parfor k=1:crbm.n_maps
        v_expected = v_expected + squeeze(convn(hidden(:,:,:,k,:), kthMap(W,k), 'full'));
    end
end
%{
%for k=1:crbm.n_maps
parfor k=1:crbm.n_maps
    v_expected =v_expected+ v_expected_k{k};
end
%}
v_expected = sigmoid(v_expected + crbm.v_b);
