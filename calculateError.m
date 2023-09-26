function [err_table, l1_timestep,l2_timestep,L1,L2] = calculateError(target,approx,dt)

max_t = min(size(target, 2), size(approx, 2));

assert(size(target, 1) == size(approx, 1),"The two vectors do not have the same number of rows");

l1_timestep = abs(target(:,1:max_t)-approx(:,1:max_t));
l2_timestep = sqrt((target(:,1:max_t)-approx(:,1:max_t)).^2);

L1 = dt*sum(l1_timestep,2);
L2 = sqrt(dt*sum(l2_timestep,2));


N_state = size(target,1);
states = string(1:1:N_state);

err_table = array2table([L1,L2],'VariableNames',{'L1','L2'},'RowName',states); 
disp(err_table)
end
