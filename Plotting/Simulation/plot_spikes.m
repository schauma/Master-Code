function plot_spikes(t,spikes,dotsize,distance,offset,color)

[rows,cols] = find(spikes);
urows = unique(rows);
nrows = length(urows);
for i = 1:nrows
c = cols(rows ==urows(i));
scatter(t(c),distance*(i-1-nrows)-offset,dotsize,color(urows(i),:),"filled");
hold on

end
% [r,c] = find(spikes);
% scatter(t(c),distance*(r-1),dotsize,r,"filled")
end
