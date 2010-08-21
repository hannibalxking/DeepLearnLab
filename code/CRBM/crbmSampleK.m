function [ h_0, v_k, h_k ] = crbmSampleK( crbm, visible, k )

h_0 = bernoulli_sample(crbmHExpectation(crbm, visible));
h_k = h_0;

for i=1:k
    v_k = crbmVExpectation(crbm, h_k);
    h_k = crbmHExpectation(crbm, v_k);
end


end

