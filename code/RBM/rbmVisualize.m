function [ im ] = rbmVisualize(rbm, h, w, n_high, n_wide, start_h )

im = zeros(n_high * h, n_wide * w);
W = (rbm.W - min(min(rbm.W)))/(max(max(rbm.W)) - min(min(rbm.W)));

hidden = 1;
for m=0:n_high-1
    for r=0:n_wide-1
        visible = 1;
        for i=1:h
            for j=1:w
                if visible > rbm.n_v
                    break;
                end
                im(m*h+1+i, r*w+1+j) = W(hidden, visible);
                visible = visible+1;
            end
        end
        hidden = hidden+1;
    end
end


end

