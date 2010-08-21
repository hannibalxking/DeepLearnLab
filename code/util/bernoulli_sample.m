function [ sample ] = bernoulli_sample( p )
    sample = rand(size(p)) < p;
end

