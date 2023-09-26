function bounds = plot_spikes(t,spikes,dotsize,distance,offset,color)

[rows,cols] = find(spikes);
urows = unique(rows);

nrows = nnz(sum(spikes,2));
urows = find(sum(spikes,2));

[rows,idxs] = sort(rows);
cols = cols(idxs);
colors=[];
%colors = zeros(nnz(spikes),3);
s = 1;
for i = 1:nrows
current_row = rows ==urows(i);
c = cols(current_row);
colors(s:(s+nnz(current_row)-1),:) = repmat(color(urows(i),:),nnz(current_row),1); 
%scatter(t(c),distance*(i-1-nrows)-offset,dotsize,color(urows(i),:),"filled");
s = s+nnz(current_row);
end
dots = distance*(rows-1-nrows)-offset;
bounds = [min(dots),max(dots)];
scatter(t(cols),dots,dotsize,colors,"filled")
xlim([t(1),t(end)]);
% [r,c] = find(spikes);
% scatter(t(c),distance*(r-1),dotsize,r,"filled")
end
