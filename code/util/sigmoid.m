function s = sigmoid( x )
%SIGMOID apply logistic function to input
s = 1.0 ./ (1 + exp(-x));
end

